//! Overflow recovery via snapshot reconciliation.
//!
//! When the notification back-end reports that events were lost
//! ([`Event::need_rescan`]), the debouncer can optionally hand control to a
//! [`ResyncCoordinator`] (see [`crate::Debouncer::set_resync_coordinator`]):
//!
//! * normal batch publishing is paused and incoming events are buffered,
//!   tagged with the current scan generation,
//! * an injected [`DirScanner`] takes a snapshot of the watched roots,
//! * the snapshot is compared against the last confirmed snapshot and a
//!   resync batch is emitted, covering only per-path additions, removals and
//!   content fingerprint changes (renames are intentionally not guessed),
//! * events buffered during the scan are replayed through the normal
//!   debounce pipeline and re-checked against the resync batch, so events
//!   already covered by the batch are not emitted twice,
//! * scan results from stale generations are discarded and can never
//!   overwrite newer state.

use std::{
    collections::HashSet,
    fmt,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use notify::{
    Error, ErrorKind, Event, EventKind, RecursiveMode,
    event::{CreateKind, DataChange, ModifyKind, RemoveKind},
};
use rustc_hash::FxHashMap as HashMap;
use walkdir::WalkDir;

use crate::{DebouncedEvent, time::now};

/// Default upper bound for the number of events buffered while a resync scan is in flight.
pub const DEFAULT_MAX_BUFFERED_EVENTS: usize = 1024;

/// A content fingerprint for a single path.
pub type Fingerprint = u64;

/// A snapshot of the watched directory tree, mapping each path to its content fingerprint.
pub type Snapshot = HashMap<PathBuf, Fingerprint>;

/// The interface of a directory scanner used for overflow recovery.
///
/// The scanner is called with the currently watched root paths and must
/// return a snapshot of the directory tree. The snapshot is reconciled
/// against the last confirmed snapshot by the [`ResyncCoordinator`].
///
/// # Example implementation
///
/// ```rust
/// # use std::path::PathBuf;
/// # use notify::{Error, RecursiveMode};
/// # use notify_debouncer_full::{DirScanner, FingerprintScanner, Snapshot};
/// struct MyScanner;
///
/// impl DirScanner for MyScanner {
///     fn scan(&mut self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
///         // Delegate to the built-in content fingerprint scanner.
///         FingerprintScanner::new().scan(roots)
///     }
/// }
/// ```
pub trait DirScanner: Send + 'static {
    /// Scan the given root paths and return a snapshot of the directory tree.
    fn scan(&mut self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error>;
}

impl<F> DirScanner for F
where
    F: FnMut(&[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> + Send + 'static,
{
    fn scan(&mut self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
        (self)(roots)
    }
}

/// A [`DirScanner`] that walks the watched roots and fingerprints file contents.
///
/// Files are fingerprinted by hashing their contents, directories get a
/// constant marker fingerprint. Entries that cannot be read are skipped,
/// unless the root path itself is unreadable, in which case the scan fails.
#[derive(Debug, Default, Clone, Copy)]
pub struct FingerprintScanner;

impl FingerprintScanner {
    /// Construct a new scanner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DirScanner for FingerprintScanner {
    fn scan(&mut self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
        let mut snapshot = Snapshot::default();

        for (root, recursive_mode) in roots {
            let max_depth = match recursive_mode {
                RecursiveMode::Recursive => usize::MAX,
                RecursiveMode::NonRecursive => 1,
            };

            for (index, entry) in WalkDir::new(root)
                .follow_links(true)
                .max_depth(max_depth)
                .into_iter()
                .enumerate()
            {
                let entry = match entry {
                    Ok(entry) => entry,
                    // A failure to read the root itself makes the whole
                    // snapshot unreliable, so abort the scan.
                    Err(error) if index == 0 => {
                        return Err(Error::generic(&format!(
                            "failed to scan {}: {error}",
                            root.display()
                        )));
                    }
                    Err(_) => continue,
                };

                if let Some(fingerprint) = fingerprint_path(entry.path()) {
                    snapshot.insert(entry.into_path(), fingerprint);
                }
            }
        }

        Ok(snapshot)
    }
}

fn fingerprint_path(path: &Path) -> Option<Fingerprint> {
    let metadata = std::fs::metadata(path).ok()?;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if metadata.is_dir() {
        true.hash(&mut hasher);
    } else {
        false.hash(&mut hasher);
        std::fs::read(path).ok()?.hash(&mut hasher);
    }

    Some(hasher.finish())
}

pub(crate) type SharedScanner = Arc<Mutex<Box<dyn DirScanner>>>;

/// The result of completing a scan generation, see [`ResyncCoordinator::complete_scan`].
#[derive(Debug, Default)]
#[must_use]
pub struct ResyncOutcome {
    /// The resync batch to emit immediately, covering per-path additions,
    /// removals and content fingerprint changes.
    pub batch: Vec<DebouncedEvent>,
    /// Events that were buffered while the scan was in flight and must be
    /// replayed through the normal debounce pipeline.
    pub replay: Vec<Event>,
    /// The scan error, if the scan failed.
    pub error: Option<Error>,
    /// Whether the scan result was discarded because its generation is stale.
    pub stale: bool,
    /// Whether a follow-up scan generation was scheduled because more events
    /// were lost while this scan was in flight.
    pub rescan_scheduled: bool,
}

/// An overflow recovery coordinator.
///
/// The coordinator is driven by the debouncer: when the back-end reports
/// lost events, a new scan generation is started and normal batch publishing
/// is paused. The injected [`DirScanner`] takes a snapshot which is
/// reconciled against the last confirmed snapshot. Reconciliation only
/// promises per-path additions, removals and content fingerprint changes; it
/// never guesses renames.
///
/// Scan generations guard against races: results that arrive late for an old
/// generation are discarded and can never overwrite newer state.
pub struct ResyncCoordinator {
    scanner: SharedScanner,
    /// The last confirmed snapshot, used as the baseline for reconciliation.
    confirmed: Snapshot,
    /// The latest scan generation.
    generation: u64,
    /// The generation of the in-flight scan, if any.
    in_flight: Option<u64>,
    /// Whether another overflow was signaled while a scan was in flight.
    overflow_pending: bool,
    /// Whether the in-flight scan still needs to be picked up by the worker.
    scan_pending: bool,
    /// Events buffered while a scan is in flight, tagged by generation.
    buffered: Vec<(u64, Event)>,
    /// Upper bound for buffered events.
    max_buffered: usize,
    /// Whether a buffer overflow was already reported for the current scan.
    buffer_overflowed: bool,
}

impl fmt::Debug for ResyncCoordinator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResyncCoordinator")
            .field("confirmed", &self.confirmed)
            .field("generation", &self.generation)
            .field("in_flight", &self.in_flight)
            .field("overflow_pending", &self.overflow_pending)
            .field("scan_pending", &self.scan_pending)
            .field("buffered", &self.buffered.len())
            .field("max_buffered", &self.max_buffered)
            .finish_non_exhaustive()
    }
}

impl ResyncCoordinator {
    /// Construct a new coordinator using the given directory scanner.
    pub fn new(scanner: impl DirScanner) -> Self {
        Self {
            scanner: Arc::new(Mutex::new(Box::new(scanner))),
            confirmed: Snapshot::default(),
            generation: 0,
            in_flight: None,
            overflow_pending: false,
            scan_pending: false,
            buffered: Vec::new(),
            max_buffered: DEFAULT_MAX_BUFFERED_EVENTS,
            buffer_overflowed: false,
        }
    }

    /// Set the upper bound for events buffered while a scan is in flight.
    ///
    /// When the limit is exceeded, an error is reported and a follow-up scan
    /// is forced, so no state is lost permanently.
    #[must_use]
    pub fn with_max_buffered_events(mut self, max_buffered: usize) -> Self {
        self.max_buffered = max_buffered;
        self
    }

    /// The last confirmed snapshot, used as the baseline for reconciliation.
    #[must_use]
    pub fn confirmed_snapshot(&self) -> &Snapshot {
        &self.confirmed
    }

    /// The latest scan generation.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether a scan is currently in flight.
    #[must_use]
    pub fn is_scanning(&self) -> bool {
        self.in_flight.is_some()
    }

    /// The number of events currently buffered while a scan is in flight.
    #[must_use]
    pub fn buffered_event_count(&self) -> usize {
        self.buffered.len()
    }

    /// Signal that events were lost and a resync is required.
    ///
    /// Starts a new scan generation and returns its id. If a scan is already
    /// in flight, no new generation is started; instead the in-flight scan is
    /// marked as outdated so its result will be discarded and a follow-up
    /// scan is scheduled once it completes.
    pub fn begin_overflow(&mut self) -> u64 {
        if self.is_scanning() {
            self.overflow_pending = true;
            self.in_flight
                .expect("is_scanning implies an in-flight scan")
        } else {
            self.start_scan()
        }
    }

    /// Buffer an event that arrived while a scan is in flight.
    ///
    /// The event is tagged with the current scan generation and replayed once
    /// the scan completes. Returns an error if the buffer limit is exceeded;
    /// in that case the event is dropped and a follow-up scan is forced.
    pub fn buffer_event(&mut self, event: Event) -> Result<(), Error> {
        let Some(generation) = self.in_flight else {
            return Ok(());
        };

        if self.buffered.len() >= self.max_buffered {
            // The buffer is bounded; once it overflows we can no longer rely
            // on replaying individual events, so force another scan
            // generation and report the loss once per scan.
            self.overflow_pending = true;

            if self.buffer_overflowed {
                return Ok(());
            }
            self.buffer_overflowed = true;

            return Err(Error::new(ErrorKind::Generic(format!(
                "resync event buffer limit of {} exceeded; events are dropped and a follow-up scan was scheduled",
                self.max_buffered
            ))));
        }

        self.buffered.push((generation, event));
        Ok(())
    }

    /// Complete a scan generation with its result.
    ///
    /// Results for stale generations are discarded, so an old scan can never
    /// overwrite newer state. If more events were lost while the scan was in
    /// flight, the result is outdated as well: it is discarded and a
    /// follow-up scan generation is started instead.
    ///
    /// Otherwise the snapshot becomes the new confirmed snapshot and the
    /// returned [`ResyncOutcome`] contains the resync batch (the diff against
    /// the previous confirmed snapshot) and the buffered events to replay.
    pub fn complete_scan(
        &mut self,
        generation: u64,
        result: Result<Snapshot, Error>,
    ) -> ResyncOutcome {
        if self.in_flight != Some(generation) {
            // Stale scan result from an old (or unknown) generation.
            return ResyncOutcome {
                stale: true,
                ..ResyncOutcome::default()
            };
        }
        self.in_flight = None;

        if self.overflow_pending {
            // More events were lost while this scan was running, so its
            // result is already outdated. Discard it and scan again.
            self.overflow_pending = false;
            self.start_scan();
            return ResyncOutcome {
                rescan_scheduled: true,
                ..ResyncOutcome::default()
            };
        }

        let (batch, error) = match result {
            Ok(snapshot) => {
                let batch = diff_snapshots(&self.confirmed, &snapshot);
                self.confirmed = snapshot;
                (batch, None)
            }
            Err(error) => (Vec::new(), Some(error)),
        };

        let replay = self.drain_buffered(generation, &batch);

        ResyncOutcome {
            batch,
            replay,
            error,
            stale: false,
            rescan_scheduled: false,
        }
    }

    fn start_scan(&mut self) -> u64 {
        self.generation += 1;
        self.in_flight = Some(self.generation);
        self.scan_pending = true;
        self.buffer_overflowed = false;
        self.generation
    }

    /// Drain buffered events up to (and including) the given generation,
    /// dropping the ones already covered by the resync batch.
    fn drain_buffered(&mut self, generation: u64, batch: &[DebouncedEvent]) -> Vec<Event> {
        let covered = batch
            .iter()
            .filter_map(|event| {
                let path = event.paths.first()?;
                let class = KindClass::of(&event.kind)?;
                Some((path.clone(), class))
            })
            .collect::<HashSet<_>>();

        let (replay, kept) = std::mem::take(&mut self.buffered)
            .into_iter()
            .partition(|(tag, _)| *tag <= generation);
        self.buffered = kept;

        replay
            .into_iter()
            .filter_map(|(_, event)| {
                let covered_by_batch = event
                    .paths
                    .first()
                    .and_then(|path| Some((path.clone(), KindClass::of(&event.kind)?)))
                    .is_some_and(|key| covered.contains(&key));

                // Re-check: events already covered by the resync batch are
                // not emitted twice.
                (!covered_by_batch).then_some(event)
            })
            .collect()
    }

    pub(crate) fn scan_job_pending(&self) -> bool {
        self.scan_pending
    }

    pub(crate) fn take_scan_job(
        &mut self,
        roots: Vec<(PathBuf, RecursiveMode)>,
    ) -> Option<(u64, SharedScanner, Vec<(PathBuf, RecursiveMode)>)> {
        if !self.scan_pending {
            return None;
        }
        self.scan_pending = false;

        let generation = self
            .in_flight
            .expect("a pending scan job implies an in-flight scan");

        Some((generation, Arc::clone(&self.scanner), roots))
    }

    pub(crate) fn scan(&self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
        self.scanner.lock().unwrap().scan(roots)
    }

    pub(crate) fn merge_baseline(&mut self, snapshot: Snapshot) {
        self.confirmed.extend(snapshot);
    }

    pub(crate) fn remove_baseline(&mut self, path: &Path) {
        self.confirmed.retain(|p, _| !p.starts_with(path));
    }
}

/// The class of an event kind for the purpose of re-checking buffered events
/// against a resync batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KindClass {
    Create,
    Remove,
}

impl KindClass {
    fn of(kind: &EventKind) -> Option<Self> {
        match kind {
            EventKind::Create(_) => Some(Self::Create),
            EventKind::Remove(_) => Some(Self::Remove),
            _ => None,
        }
    }
}

/// Reconcile two snapshots.
///
/// Only per-path additions, removals and content fingerprint changes are
/// reported; renames are intentionally not guessed and surface as a removal
/// plus an addition.
fn diff_snapshots(old: &Snapshot, new: &Snapshot) -> Vec<DebouncedEvent> {
    let time = now();

    let mut paths = old.keys().chain(new.keys()).collect::<Vec<_>>();
    paths.sort();
    paths.dedup();

    let mut batch = Vec::new();
    for path in paths {
        match (old.get(path), new.get(path)) {
            (None, Some(_)) => {
                batch.push(resync_event(EventKind::Create(CreateKind::Any), path, time))
            }
            (Some(_), None) => {
                batch.push(resync_event(EventKind::Remove(RemoveKind::Any), path, time))
            }
            (Some(old_fingerprint), Some(new_fingerprint))
                if old_fingerprint != new_fingerprint =>
            {
                batch.push(resync_event(
                    EventKind::Modify(ModifyKind::Data(DataChange::Any)),
                    path,
                    time,
                ));
            }
            _ => {}
        }
    }

    batch
}

fn resync_event(kind: EventKind, path: &Path, time: std::time::Instant) -> DebouncedEvent {
    DebouncedEvent::new(
        Event::new(kind)
            .add_path(path.to_path_buf())
            .set_info("resync"),
        time,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        fs,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        time::Duration,
    };

    use notify::{Watcher, WatcherKind, event::Flag};
    use tempfile::tempdir;

    use crate::{NoCache, new_debouncer_opt};

    fn snapshot(entries: &[(&str, u64)]) -> Snapshot {
        entries
            .iter()
            .map(|(path, fingerprint)| (PathBuf::from(path), *fingerprint))
            .collect()
    }

    fn rescan_event() -> Event {
        Event::new(EventKind::Other).set_flag(Flag::Rescan)
    }

    /// A watcher that lets the test push events into the debouncer.
    #[derive(Clone, Default)]
    struct MockWatcher {
        handler: Arc<Mutex<Option<Box<dyn FnMut(notify::Result<Event>) + Send>>>>,
    }

    impl fmt::Debug for MockWatcher {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockWatcher").finish_non_exhaustive()
        }
    }

    impl MockWatcher {
        fn send(&self, event: Event) {
            let mut handler = self.handler.lock().unwrap();
            if let Some(handler) = handler.as_mut() {
                handler(Ok(event));
            }
        }
    }

    impl Watcher for MockWatcher {
        fn new<F: notify::EventHandler>(
            mut event_handler: F,
            _config: notify::Config,
        ) -> notify::Result<Self> {
            let watcher = Self::default();
            *watcher.handler.lock().unwrap() = Some(Box::new(move |event| {
                notify::EventHandler::handle_event(&mut event_handler, event)
            }));
            Ok(watcher)
        }

        fn watch(&mut self, _path: &Path, _recursive_mode: RecursiveMode) -> notify::Result<()> {
            Ok(())
        }

        fn unwatch(&mut self, _path: &Path) -> notify::Result<()> {
            Ok(())
        }

        fn kind() -> WatcherKind {
            WatcherKind::NullWatcher
        }
    }

    #[test]
    fn diff_reports_only_additions_removals_and_fingerprint_changes() {
        let mut coordinator = ResyncCoordinator::new(FingerprintScanner::new());
        coordinator.merge_baseline(snapshot(&[
            ("/watch/a", 1),
            ("/watch/b", 2),
            ("/watch/c", 3),
        ]));

        let generation = coordinator.begin_overflow();

        // `d` has the same fingerprint as the removed `a` (e.g. a rename),
        // but reconciliation must not guess renames: expect remove(a) and
        // create(d) instead.
        let outcome = coordinator.complete_scan(
            generation,
            Ok(snapshot(&[
                ("/watch/b", 2),
                ("/watch/c", 9),
                ("/watch/d", 1),
            ])),
        );

        assert!(!outcome.stale);
        assert!(!outcome.rescan_scheduled);
        assert!(outcome.error.is_none());
        assert!(outcome.replay.is_empty());

        let batch = outcome
            .batch
            .iter()
            .map(|event| (event.paths[0].clone(), event.kind))
            .collect::<Vec<_>>();
        assert_eq!(
            batch,
            vec![
                (
                    PathBuf::from("/watch/a"),
                    EventKind::Remove(RemoveKind::Any)
                ),
                (
                    PathBuf::from("/watch/c"),
                    EventKind::Modify(ModifyKind::Data(DataChange::Any))
                ),
                (
                    PathBuf::from("/watch/d"),
                    EventKind::Create(CreateKind::Any)
                ),
            ]
        );

        // No rename events are ever synthesized.
        assert!(outcome.batch.iter().all(|event| event.paths.len() == 1));
        assert!(
            outcome
                .batch
                .iter()
                .all(|event| event.info() == Some("resync"))
        );
    }

    #[test]
    fn consecutive_overflows_collapse_into_a_single_follow_up_scan() {
        let mut coordinator = ResyncCoordinator::new(FingerprintScanner::new());
        coordinator.merge_baseline(snapshot(&[("/watch/a", 1)]));

        let first = coordinator.begin_overflow();
        assert_eq!(coordinator.begin_overflow(), first);
        assert_eq!(coordinator.begin_overflow(), first);
        assert!(coordinator.is_scanning());

        // The in-flight scan's result is outdated before it even arrives and
        // must not overwrite the confirmed snapshot.
        let outcome = coordinator.complete_scan(first, Ok(snapshot(&[("/watch/a", 2)])));
        assert!(outcome.rescan_scheduled);
        assert!(outcome.batch.is_empty());
        assert_eq!(
            coordinator.confirmed_snapshot(),
            &snapshot(&[("/watch/a", 1)])
        );
        assert!(coordinator.is_scanning());

        // The follow-up scan is applied.
        let outcome = coordinator.complete_scan(first + 1, Ok(snapshot(&[("/watch/a", 2)])));
        assert!(!outcome.stale);
        assert!(!outcome.rescan_scheduled);
        assert_eq!(outcome.batch.len(), 1);
        assert!(matches!(outcome.batch[0].kind, EventKind::Modify(_)));
        assert_eq!(
            coordinator.confirmed_snapshot(),
            &snapshot(&[("/watch/a", 2)])
        );
        assert!(!coordinator.is_scanning());
    }

    #[test]
    fn stale_scan_result_is_discarded() {
        let mut coordinator = ResyncCoordinator::new(FingerprintScanner::new());
        coordinator.merge_baseline(snapshot(&[("/watch/a", 1)]));

        let first = coordinator.begin_overflow();
        // A consecutive overflow marks the in-flight scan as outdated.
        coordinator.begin_overflow();
        let outcome = coordinator.complete_scan(first, Ok(snapshot(&[("/watch/a", 2)])));
        assert!(outcome.rescan_scheduled);

        // The follow-up scan completes and is applied.
        let outcome = coordinator.complete_scan(first + 1, Ok(snapshot(&[("/watch/a", 3)])));
        assert!(!outcome.stale);
        assert_eq!(
            coordinator.confirmed_snapshot(),
            &snapshot(&[("/watch/a", 3)])
        );

        // A late result for the old generation arrives and must be discarded.
        let outcome =
            coordinator.complete_scan(first, Ok(snapshot(&[("/watch/a", 1), ("/watch/b", 5)])));
        assert!(outcome.stale);
        assert!(outcome.batch.is_empty());
        assert!(outcome.replay.is_empty());
        assert!(outcome.error.is_none());
        assert_eq!(
            coordinator.confirmed_snapshot(),
            &snapshot(&[("/watch/a", 3)])
        );
        assert!(!coordinator.is_scanning());
    }

    #[test]
    fn scan_failure_keeps_confirmed_snapshot_and_replays_buffered_events() {
        let mut coordinator = ResyncCoordinator::new(FingerprintScanner::new());
        coordinator.merge_baseline(snapshot(&[("/watch/a", 1)]));

        let generation = coordinator.begin_overflow();
        coordinator
            .buffer_event(
                Event::new(EventKind::Create(CreateKind::File)).add_path("/watch/x".into()),
            )
            .unwrap();

        let outcome = coordinator.complete_scan(generation, Err(Error::generic("boom")));

        assert!(outcome.error.is_some());
        assert!(outcome.batch.is_empty());
        assert_eq!(outcome.replay.len(), 1);
        assert_eq!(
            coordinator.confirmed_snapshot(),
            &snapshot(&[("/watch/a", 1)])
        );
        assert!(!coordinator.is_scanning());
    }

    #[test]
    fn buffer_limit_exceeded_reports_error_and_forces_follow_up_scan() {
        let mut coordinator =
            ResyncCoordinator::new(FingerprintScanner::new()).with_max_buffered_events(1);
        coordinator.merge_baseline(snapshot(&[("/watch/a", 1)]));

        let first = coordinator.begin_overflow();
        coordinator
            .buffer_event(Event::new(EventKind::Any).add_path("/watch/one".into()))
            .unwrap();

        let error = coordinator
            .buffer_event(Event::new(EventKind::Any).add_path("/watch/two".into()))
            .unwrap_err();
        assert!(format!("{error:?}").contains("buffer limit"));

        // Further events are dropped silently.
        coordinator
            .buffer_event(Event::new(EventKind::Any).add_path("/watch/three".into()))
            .unwrap();
        assert_eq!(coordinator.buffered_event_count(), 1);

        // The in-flight scan's result is discarded and a follow-up scan is
        // forced, so the dropped events are recovered by reconciliation.
        let outcome = coordinator.complete_scan(first, Ok(snapshot(&[("/watch/a", 1)])));
        assert!(outcome.rescan_scheduled);

        let outcome = coordinator.complete_scan(first + 1, Ok(snapshot(&[("/watch/a", 1)])));
        assert!(!outcome.rescan_scheduled);
        assert_eq!(outcome.replay.len(), 1);
    }

    #[test]
    fn resync_recovers_lost_remove_and_create() {
        let dir = tempdir().unwrap();
        let removed = dir.path().join("removed.txt");
        fs::write(&removed, b"removed").unwrap();

        let (tx, rx) = mpsc::channel();
        let mut debouncer = new_debouncer_opt::<_, MockWatcher, NoCache>(
            Duration::from_millis(50),
            Some(Duration::from_millis(10)),
            tx,
            NoCache,
            notify::Config::default(),
        )
        .unwrap();
        debouncer.set_resync_coordinator(ResyncCoordinator::new(FingerprintScanner::new()));
        debouncer
            .watch(dir.path(), RecursiveMode::Recursive)
            .unwrap();

        // The baseline snapshot contains the file.
        {
            let data = debouncer.data.inner.lock().unwrap();
            let confirmed = data.resync.as_ref().unwrap().confirmed_snapshot();
            assert!(confirmed.contains_key(&removed));
        }

        // Simulate losing a remove and a create event.
        fs::remove_file(&removed).unwrap();
        let created = dir.path().join("created.txt");
        fs::write(&created, b"created").unwrap();

        debouncer.watcher.send(rescan_event());

        let events = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("did not receive resync batch")
            .expect("received errors");

        let find = |path: &PathBuf| {
            events
                .iter()
                .find(|event| event.paths.first() == Some(path))
        };

        let remove = find(&removed).expect("missing remove event");
        assert!(matches!(remove.kind, EventKind::Remove(_)));
        assert_eq!(remove.info(), Some("resync"));

        let create = find(&created).expect("missing create event");
        assert!(matches!(create.kind, EventKind::Create(_)));
        assert_eq!(create.info(), Some("resync"));

        // The confirmed snapshot was updated.
        let data = debouncer.data.inner.lock().unwrap();
        let confirmed = data.resync.as_ref().unwrap().confirmed_snapshot();
        assert!(!confirmed.contains_key(&removed));
        assert!(confirmed.contains_key(&created));
    }

    /// A scanner that blocks mid-scan while armed, so the test can create a
    /// file while the scan is in flight.
    struct GateScanner {
        armed: Arc<AtomicBool>,
        started: mpsc::Sender<()>,
        proceed: mpsc::Receiver<()>,
    }

    impl DirScanner for GateScanner {
        fn scan(&mut self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
            if self.armed.swap(false, Ordering::SeqCst) {
                let _ = self.started.send(());
                let _ = self.proceed.recv();
            }
            FingerprintScanner::new().scan(roots)
        }
    }

    #[test]
    fn resync_includes_file_created_mid_scan() {
        let dir = tempdir().unwrap();
        let existing = dir.path().join("existing.txt");
        fs::write(&existing, b"existing").unwrap();

        let (started_tx, started_rx) = mpsc::channel();
        let (proceed_tx, proceed_rx) = mpsc::channel();
        let armed = Arc::new(AtomicBool::new(false));

        let (tx, rx) = mpsc::channel();
        let mut debouncer = new_debouncer_opt::<_, MockWatcher, NoCache>(
            Duration::from_millis(50),
            Some(Duration::from_millis(10)),
            tx,
            NoCache,
            notify::Config::default(),
        )
        .unwrap();
        debouncer.set_resync_coordinator(ResyncCoordinator::new(GateScanner {
            armed: Arc::clone(&armed),
            started: started_tx,
            proceed: proceed_rx,
        }));
        debouncer
            .watch(dir.path(), RecursiveMode::Recursive)
            .unwrap();

        // Arm the gate so the overflow-triggered scan blocks mid-scan.
        armed.store(true, Ordering::SeqCst);
        debouncer.watcher.send(rescan_event());

        // Wait until the scan is in flight, then create a file mid-scan.
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("scan did not start");
        let mid_scan = dir.path().join("mid_scan.txt");
        fs::write(&mid_scan, b"mid-scan").unwrap();

        // The corresponding events arrive while the scan is in flight and
        // must be buffered, not published.
        debouncer
            .watcher
            .send(Event::new(EventKind::Create(CreateKind::File)).add_path(mid_scan.clone()));
        debouncer.watcher.send(
            Event::new(EventKind::Modify(ModifyKind::Data(DataChange::Content)))
                .add_path(mid_scan.clone()),
        );

        // Let the scan finish; it picks up the new file.
        proceed_tx.send(()).unwrap();

        // First batch: the resync diff, containing the mid-scan file.
        let events = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("did not receive resync batch")
            .expect("received errors");
        let creates = events
            .iter()
            .filter(|event| {
                event.paths.first() == Some(&mid_scan) && matches!(event.kind, EventKind::Create(_))
            })
            .count();
        assert_eq!(
            creates, 1,
            "expected exactly one create event for the mid-scan file"
        );

        // Second batch: the replayed modify event, delivered through the
        // normal debounce path.
        let events = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("did not receive replayed events")
            .expect("received errors");
        assert!(events.iter().any(|event| {
            event.paths.first() == Some(&mid_scan)
                && matches!(event.kind, EventKind::Modify(_))
                && event.info().is_none()
        }));

        // The buffered create event was deduplicated against the resync
        // batch, so nothing else is published.
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err());

        // The final confirmed snapshot contains the mid-scan file.
        let data = debouncer.data.inner.lock().unwrap();
        let confirmed = data.resync.as_ref().unwrap().confirmed_snapshot();
        assert!(confirmed.contains_key(&mid_scan));
    }

    #[test]
    fn scan_failure_is_reported_and_normal_debounce_continues() {
        let dir = tempdir().unwrap();

        let armed = Arc::new(AtomicBool::new(false));
        let scanner_armed = Arc::clone(&armed);
        let scanner = move |roots: &[(PathBuf, RecursiveMode)]| -> Result<Snapshot, Error> {
            if scanner_armed.load(Ordering::SeqCst) {
                Err(Error::generic("scan failed"))
            } else {
                FingerprintScanner::new().scan(roots)
            }
        };

        let (tx, rx) = mpsc::channel();
        let mut debouncer = new_debouncer_opt::<_, MockWatcher, NoCache>(
            Duration::from_millis(50),
            Some(Duration::from_millis(10)),
            tx,
            NoCache,
            notify::Config::default(),
        )
        .unwrap();
        debouncer.set_resync_coordinator(ResyncCoordinator::new(scanner));
        debouncer
            .watch(dir.path(), RecursiveMode::Recursive)
            .unwrap();

        armed.store(true, Ordering::SeqCst);
        debouncer.watcher.send(rescan_event());

        let errors = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("did not receive scan error")
            .expect_err("expected scan error");
        assert!(format!("{errors:?}").contains("scan failed"));

        // The normal debounce path still works afterwards.
        armed.store(false, Ordering::SeqCst);
        let file = dir.path().join("file.txt");
        fs::write(&file, b"content").unwrap();
        debouncer
            .watcher
            .send(Event::new(EventKind::Create(CreateKind::File)).add_path(file.clone()));

        let events = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("did not receive debounced events")
            .expect("received errors");
        assert!(events.iter().any(|event| {
            event.paths.first() == Some(&file)
                && matches!(event.kind, EventKind::Create(_))
                && event.info().is_none()
        }));
    }

    #[test]
    fn normal_debounce_path_is_unaffected_by_coordinator() {
        let dir = tempdir().unwrap();

        let (tx, rx) = mpsc::channel();
        let mut debouncer = new_debouncer_opt::<_, MockWatcher, NoCache>(
            Duration::from_millis(50),
            Some(Duration::from_millis(10)),
            tx,
            NoCache,
            notify::Config::default(),
        )
        .unwrap();
        debouncer.set_resync_coordinator(ResyncCoordinator::new(FingerprintScanner::new()));
        debouncer
            .watch(dir.path(), RecursiveMode::Recursive)
            .unwrap();

        let file = dir.path().join("file.txt");
        fs::write(&file, b"content").unwrap();
        debouncer
            .watcher
            .send(Event::new(EventKind::Create(CreateKind::File)).add_path(file.clone()));

        let events = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("did not receive debounced events")
            .expect("received errors");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, EventKind::Create(_)));
        assert_eq!(events[0].paths, vec![file.clone()]);
        assert!(events[0].info().is_none());
    }
}
