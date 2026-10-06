//! Optional overflow recovery coordination for the debouncer.
//!
//! When the underlying watcher reports that events were lost
//! ([`Event::need_rescan`]), the recovery coordinator pauses normal batch
//! emission, takes a fresh directory snapshot through an injected
//! [`DirScanner`], diffs it against the last confirmed snapshot and emits a
//! resync batch. Events that arrive while a scan is in flight are buffered
//! per generation and replayed (and re-checked against disk) once the scan
//! completes, so a stale scan can never overwrite newer state.
//!
//! Reconciliation is strictly path-based: creations, removals and content
//! fingerprint changes. Renames are intentionally *not* guessed; a rename
//! shows up as a removal plus a creation.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    hash::Hasher,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use notify::{
    Error, ErrorKind, Event, EventKind, RecursiveMode,
    event::{CreateKind, DataChange, ModifyKind, RemoveKind},
};
use walkdir::WalkDir;

/// Content fingerprint of a single path.
pub type Fingerprint = u64;

/// Fingerprint used for directories (their "content" is not hashed).
const DIRECTORY_FINGERPRINT: Fingerprint = 0x9e37_79b9_7f4a_7c15;

/// Info tag attached to every event of a resync batch.
pub const RESYNC_INFO: &str = "resync";

/// A single entry of a [`Snapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotEntry {
    /// Content fingerprint of the path.
    pub fingerprint: Fingerprint,
    /// Whether the path is a directory.
    pub is_dir: bool,
}

/// A point-in-time snapshot of all watched paths.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Snapshot {
    entries: BTreeMap<PathBuf, SnapshotEntry>,
}

impl Snapshot {
    /// Create an empty snapshot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update an entry.
    pub fn insert(&mut self, path: impl Into<PathBuf>, entry: SnapshotEntry) {
        self.entries.insert(path.into(), entry);
    }

    /// Remove the entry for `path` and all entries below it.
    pub fn remove(&mut self, path: &Path) {
        let paths: Vec<PathBuf> = self
            .entries
            .range(path.to_path_buf()..)
            .take_while(|(p, _)| p.starts_with(path))
            .map(|(p, _)| p.clone())
            .collect();
        for p in paths {
            self.entries.remove(&p);
        }
    }

    /// Get the entry for `path`, if present.
    #[must_use]
    pub fn get(&self, path: &Path) -> Option<&SnapshotEntry> {
        self.entries.get(path)
    }

    /// Whether the snapshot contains `path`.
    #[must_use]
    pub fn contains_path(&self, path: &Path) -> bool {
        self.entries.contains_key(path)
    }

    /// Number of entries in the snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the snapshot is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries, sorted by path.
    pub fn iter(&self) -> impl Iterator<Item = (&PathBuf, &SnapshotEntry)> {
        self.entries.iter()
    }

    /// Diff `self` (the new snapshot) against `previous` (the last confirmed
    /// snapshot) and return the events that reconcile the two.
    ///
    /// The diff only reports per-path creations, removals and content
    /// fingerprint changes; it never guesses renames. Events are emitted in
    /// path-sorted order (removals first) and tagged with [`RESYNC_INFO`].
    #[must_use]
    pub fn diff(&self, previous: &Snapshot) -> Vec<Event> {
        let mut events = Vec::new();

        for (path, old) in &previous.entries {
            if !self.entries.contains_key(path) {
                let kind = if old.is_dir {
                    RemoveKind::Folder
                } else {
                    RemoveKind::File
                };
                events.push(
                    Event::new(EventKind::Remove(kind))
                        .add_path(path.clone())
                        .set_info(RESYNC_INFO),
                );
            }
        }

        for (path, new) in &self.entries {
            match previous.entries.get(path) {
                None => {
                    let kind = if new.is_dir {
                        CreateKind::Folder
                    } else {
                        CreateKind::File
                    };
                    events.push(
                        Event::new(EventKind::Create(kind))
                            .add_path(path.clone())
                            .set_info(RESYNC_INFO),
                    );
                }
                Some(old) if old.fingerprint != new.fingerprint || old.is_dir != new.is_dir => {
                    events.push(
                        Event::new(EventKind::Modify(ModifyKind::Data(DataChange::Content)))
                            .add_path(path.clone())
                            .set_info(RESYNC_INFO),
                    );
                }
                _ => {}
            }
        }

        events
    }
}

/// A directory scanner injected into the overflow recovery coordinator.
///
/// Implementations must be able to take a consistent snapshot of the given
/// watch roots and to fingerprint a single path for the post-scan re-check.
pub trait DirScanner: Debug + Send + Sync + 'static {
    /// Take a snapshot of all `roots`.
    fn scan(&self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error>;

    /// Fingerprint a single path. Returns `None` if the path does not exist.
    fn fingerprint(&self, path: &Path) -> Option<SnapshotEntry>;
}

/// Configuration for the overflow recovery coordinator.
#[derive(Debug)]
pub struct RecoveryConfig<S: DirScanner> {
    /// The injected directory scanner.
    pub scanner: S,
    /// Maximum number of events buffered while a scan is in flight.
    ///
    /// When the limit is exceeded an error is emitted and another scan is
    /// forced once the in-flight scan completes.
    pub max_buffered_events: usize,
}

impl<S: DirScanner> RecoveryConfig<S> {
    /// Create a new recovery configuration with a default buffer limit.
    #[must_use]
    pub fn new(scanner: S) -> Self {
        Self {
            scanner,
            max_buffered_events: 1024,
        }
    }

    /// Set the maximum number of events buffered while scanning.
    #[must_use]
    pub fn with_max_buffered_events(mut self, max_buffered_events: usize) -> Self {
        self.max_buffered_events = max_buffered_events;
        self
    }
}

/// Internal state of the overflow recovery coordinator.
#[derive(Debug)]
pub(crate) struct RecoveryState {
    pub scanner: Arc<dyn DirScanner>,
    /// The last snapshot that was fully reconciled with the emitted events.
    pub last_confirmed: Option<Snapshot>,
    /// Monotonic generation counter; bumped every time a new scan starts.
    pub generation: u64,
    /// Whether a scan for `generation` is currently in flight.
    pub scanning: bool,
    /// Another overflow was signalled while scanning; scan again afterwards.
    pub rescan_requested: bool,
    /// Events buffered (by generation) while a scan is in flight.
    pub buffered: Vec<Event>,
    pub max_buffered_events: usize,
}

impl RecoveryState {
    pub(crate) fn new(config: RecoveryConfig<impl DirScanner>) -> Self {
        Self {
            scanner: Arc::new(config.scanner),
            last_confirmed: None,
            generation: 0,
            scanning: false,
            rescan_requested: false,
            buffered: Vec::new(),
            max_buffered_events: config.max_buffered_events,
        }
    }
}

/// A pending recovery scan request: generation, watch roots and the scanner
/// to run (outside the debouncer lock).
pub(crate) type PendingScan = (u64, Vec<(PathBuf, RecursiveMode)>, Arc<dyn DirScanner>);

/// Fingerprint a single path on disk.
///
/// Files are fingerprinted by hashing their contents (falling back to
/// metadata if the contents cannot be read), directories get a constant
/// marker fingerprint.
pub(crate) fn fingerprint_path(path: &Path) -> Option<SnapshotEntry> {
    let metadata = std::fs::symlink_metadata(path).ok()?;

    if metadata.is_dir() {
        return Some(SnapshotEntry {
            fingerprint: DIRECTORY_FINGERPRINT,
            is_dir: true,
        });
    }

    let fingerprint = hash_file_contents(path).unwrap_or_else(|| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write_u64(metadata.len());
        if let Ok(modified) = metadata.modified() {
            if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                hasher.write_u128(duration.as_nanos());
            }
        }
        hasher.finish()
    });

    Some(SnapshotEntry {
        fingerprint,
        is_dir: false,
    })
}

fn hash_file_contents(path: &Path) -> Option<Fingerprint> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut buffer = [0u8; 8192];
    loop {
        match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => hasher.write(&buffer[..n]),
            Err(_) => return None,
        }
    }
    Some(hasher.finish())
}

/// The default [`DirScanner`] implementation, based on [`walkdir`].
///
/// Files are fingerprinted by their contents, so the reconciliation reports
/// creations, removals and content changes per path.
#[derive(Debug, Default, Clone, Copy)]
pub struct WalkDirScanner;

impl WalkDirScanner {
    /// Create a new scanner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DirScanner for WalkDirScanner {
    fn scan(&self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
        let mut snapshot = Snapshot::new();

        for (root, recursive_mode) in roots {
            let max_depth = match recursive_mode {
                RecursiveMode::Recursive => usize::MAX,
                RecursiveMode::NonRecursive => 1,
            };

            for entry in WalkDir::new(root).max_depth(max_depth).follow_links(false) {
                let entry = entry.map_err(|error| {
                    Error::new(ErrorKind::Generic(format!(
                        "overflow recovery scan of `{}` failed: {error}",
                        root.display()
                    )))
                })?;

                if let Some(snapshot_entry) = fingerprint_path(entry.path()) {
                    snapshot.insert(entry.path().to_path_buf(), snapshot_entry);
                }
            }
        }

        Ok(snapshot)
    }

    fn fingerprint(&self, path: &Path) -> Option<SnapshotEntry> {
        fingerprint_path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DebounceDataInner, NoCache, time::MockTime};
    use notify::event::{Flag, ModifyKind};
    use std::{
        collections::{HashMap, VecDeque},
        sync::Mutex,
        time::{Duration, Instant},
    };

    #[derive(Debug, Default)]
    struct MockScanner {
        fingerprints: Mutex<HashMap<PathBuf, SnapshotEntry>>,
    }

    impl MockScanner {
        fn with_fingerprint(self, path: &str, fingerprint: Fingerprint) -> Self {
            self.fingerprints.lock().unwrap().insert(
                PathBuf::from(path),
                SnapshotEntry {
                    fingerprint,
                    is_dir: false,
                },
            );
            self
        }
    }

    impl DirScanner for MockScanner {
        fn scan(&self, _roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
            panic!("unit tests drive complete_scan directly")
        }

        fn fingerprint(&self, path: &Path) -> Option<SnapshotEntry> {
            self.fingerprints.lock().unwrap().get(path).copied()
        }
    }

    fn snapshot(entries: &[(&str, u64)]) -> Snapshot {
        let mut snapshot = Snapshot::new();
        for (path, fingerprint) in entries {
            snapshot.insert(
                PathBuf::from(path),
                SnapshotEntry {
                    fingerprint: *fingerprint,
                    is_dir: false,
                },
            );
        }
        snapshot
    }

    fn rescan_event() -> Event {
        Event::new(EventKind::Other).set_flag(Flag::Rescan)
    }

    fn create_event(path: &str) -> Event {
        Event::new(EventKind::Create(CreateKind::File)).add_path(PathBuf::from(path))
    }

    fn state(max_buffered: usize, scanner: MockScanner) -> DebounceDataInner<NoCache> {
        let mut state = DebounceDataInner::new(NoCache, Duration::from_millis(50));
        state.recovery = Some(RecoveryState::new(
            RecoveryConfig::new(scanner).with_max_buffered_events(max_buffered),
        ));
        state.roots = VecDeque::from([(PathBuf::from("/test"), RecursiveMode::Recursive)]);
        state
    }

    fn set_baseline(state: &mut DebounceDataInner<NoCache>, entries: &[(&str, u64)]) {
        state.recovery.as_mut().unwrap().last_confirmed = Some(snapshot(entries));
    }

    fn confirmed(state: &DebounceDataInner<NoCache>) -> Snapshot {
        state
            .recovery
            .as_ref()
            .unwrap()
            .last_confirmed
            .clone()
            .unwrap()
    }

    #[test]
    fn diff_reports_create_remove_and_modify_without_rename_guessing() {
        let previous = snapshot(&[("/test/a", 1), ("/test/b", 2), ("/test/c", 3)]);
        let current = snapshot(&[("/test/b", 2), ("/test/c", 4), ("/test/d", 5)]);

        let events = current.diff(&previous);

        assert_eq!(events.len(), 3);

        assert!(matches!(
            events[0].kind,
            EventKind::Remove(RemoveKind::File)
        ));
        assert_eq!(events[0].paths, vec![PathBuf::from("/test/a")]);

        assert!(matches!(
            events[1].kind,
            EventKind::Modify(ModifyKind::Data(DataChange::Content))
        ));
        assert_eq!(events[1].paths, vec![PathBuf::from("/test/c")]);

        assert!(matches!(
            events[2].kind,
            EventKind::Create(CreateKind::File)
        ));
        assert_eq!(events[2].paths, vec![PathBuf::from("/test/d")]);

        for event in &events {
            assert_eq!(event.info(), Some(RESYNC_INFO));
            // no rename guessing: every event refers to exactly one path
            assert_eq!(event.paths.len(), 1);
        }

        // a rename is reported as remove + create, never as a rename
        let previous = snapshot(&[("/test/old", 1)]);
        let current = snapshot(&[("/test/new", 1)]);
        let events = current.diff(&previous);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0].kind, EventKind::Remove(_)));
        assert!(matches!(events[1].kind, EventKind::Create(_)));
    }

    #[test]
    fn resync_recovers_lost_remove_and_create() {
        MockTime::set_time(Instant::now());

        let mut state = state(16, MockScanner::default());
        set_baseline(&mut state, &[("/test/a", 1), ("/test/b", 2)]);

        // signal event loss
        state.add_event(rescan_event());
        assert!(state.scan_pending);
        assert!(state.rescan_event.is_none());
        assert!(
            state.debounced_events().is_empty(),
            "normal batch publishing must be paused while scanning"
        );

        let (generation, roots, _scanner) = state.take_pending_scan().unwrap();
        assert_eq!(generation, 1);
        assert_eq!(
            roots,
            vec![(PathBuf::from("/test"), RecursiveMode::Recursive)]
        );

        // the scan observes: a deleted, b unchanged, c created
        state.complete_scan(generation, Ok(snapshot(&[("/test/b", 2), ("/test/c", 3)])));

        let events = state.debounced_events();
        assert_eq!(events.len(), 2, "resync batch: {events:?}");

        assert!(matches!(
            events[0].kind,
            EventKind::Remove(RemoveKind::File)
        ));
        assert_eq!(events[0].paths, vec![PathBuf::from("/test/a")]);
        assert_eq!(events[0].info(), Some(RESYNC_INFO));

        assert!(matches!(
            events[1].kind,
            EventKind::Create(CreateKind::File)
        ));
        assert_eq!(events[1].paths, vec![PathBuf::from("/test/c")]);
        assert_eq!(events[1].info(), Some(RESYNC_INFO));

        assert_eq!(
            confirmed(&state),
            snapshot(&[("/test/b", 2), ("/test/c", 3)])
        );
    }

    #[test]
    fn file_created_mid_scan_is_replayed_and_kept_in_snapshot() {
        MockTime::set_time(Instant::now());

        let scanner = MockScanner::default().with_fingerprint("/test/d", 9);
        let mut state = state(16, scanner);
        set_baseline(&mut state, &[("/test/a", 1)]);

        state.add_event(rescan_event());
        let (generation, _, _) = state.take_pending_scan().unwrap();

        // mid-scan: a new file is created and its event arrives while the
        // scan is still in flight, so it must be buffered by generation
        state.add_event(create_event("/test/d"));
        assert_eq!(state.recovery.as_ref().unwrap().buffered.len(), 1);
        assert!(state.debounced_events().is_empty());

        // the scan result predates the creation of d
        state.complete_scan(generation, Ok(snapshot(&[("/test/a", 1)])));

        // the replayed create event is emitted through the normal path
        MockTime::advance(Duration::from_millis(100));
        let events = state.debounced_events();
        assert_eq!(events.len(), 1, "replayed events: {events:?}");
        assert!(matches!(
            events[0].kind,
            EventKind::Create(CreateKind::File)
        ));
        assert_eq!(events[0].paths, vec![PathBuf::from("/test/d")]);

        // the stale scan must not have wiped the newer state: the final
        // confirmed snapshot still contains the mid-scan file
        let confirmed = confirmed(&state);
        assert!(confirmed.contains_path(Path::new("/test/d")));
        assert_eq!(confirmed.get(Path::new("/test/d")).unwrap().fingerprint, 9);
        assert!(confirmed.contains_path(Path::new("/test/a")));
    }

    #[test]
    fn stale_scan_result_is_discarded_and_consecutive_overflow_rescans() {
        MockTime::set_time(Instant::now());

        let mut state = state(16, MockScanner::default());
        set_baseline(&mut state, &[("/test/a", 1)]);

        state.add_event(rescan_event());
        let (generation, _, _) = state.take_pending_scan().unwrap();
        assert_eq!(generation, 1);

        // consecutive overflow while the first scan is in flight
        state.add_event(rescan_event());
        assert!(state.recovery.as_ref().unwrap().rescan_requested);

        state.complete_scan(generation, Ok(snapshot(&[("/test/a", 1), ("/test/b", 2)])));

        // the consecutive overflow triggered another scan generation
        assert!(state.scan_pending);
        assert!(state.recovery.as_ref().unwrap().scanning);
        let (generation2, _, _) = state.take_pending_scan().unwrap();
        assert_eq!(generation2, 2);

        // a late result for the old generation arrives and must be discarded
        state.complete_scan(generation, Ok(snapshot(&[("/test/z", 26)])));
        assert_eq!(
            confirmed(&state),
            snapshot(&[("/test/a", 1), ("/test/b", 2)])
        );
        // only the first generation's resync batch (create b) is pending,
        // the stale result did not add anything
        assert_eq!(state.pending_resync.len(), 1);
        assert_eq!(
            state.pending_resync[0].paths,
            vec![PathBuf::from("/test/b")]
        );
        assert!(state.recovery.as_ref().unwrap().scanning);

        // the current generation completes normally
        state.complete_scan(generation2, Ok(snapshot(&[("/test/a", 1), ("/test/c", 3)])));
        assert_eq!(
            confirmed(&state),
            snapshot(&[("/test/a", 1), ("/test/c", 3)])
        );

        let events = state.debounced_events();
        let kinds = events
            .iter()
            .map(|event| (event.kind, event.paths[0].clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                (
                    EventKind::Create(CreateKind::File),
                    PathBuf::from("/test/b")
                ),
                (
                    EventKind::Remove(RemoveKind::File),
                    PathBuf::from("/test/b")
                ),
                (
                    EventKind::Create(CreateKind::File),
                    PathBuf::from("/test/c")
                ),
            ]
        );
    }

    #[test]
    fn scan_failure_surfaces_error_and_replays_buffered_events() {
        MockTime::set_time(Instant::now());

        let mut state = state(16, MockScanner::default());
        set_baseline(&mut state, &[("/test/a", 1)]);

        state.add_event(rescan_event());
        let (generation, _, _) = state.take_pending_scan().unwrap();

        state.add_event(create_event("/test/b"));

        state.complete_scan(
            generation,
            Err(Error::new(ErrorKind::Generic("boom".to_string()))),
        );

        let errors = state.errors();
        assert_eq!(errors.len(), 1);
        assert!(format!("{errors:?}").contains("boom"));

        // the confirmed snapshot is untouched and nothing was lost
        assert_eq!(confirmed(&state), snapshot(&[("/test/a", 1)]));
        assert!(!state.recovery.as_ref().unwrap().scanning);

        MockTime::advance(Duration::from_millis(100));
        let events = state.debounced_events();
        assert_eq!(events.len(), 1, "buffered events must be replayed");
        assert_eq!(events[0].paths, vec![PathBuf::from("/test/b")]);
    }

    #[test]
    fn buffer_limit_exceeded_surfaces_error_and_forces_rescan() {
        MockTime::set_time(Instant::now());

        let mut state = state(1, MockScanner::default());
        set_baseline(&mut state, &[]);

        state.add_event(rescan_event());
        let (generation, _, _) = state.take_pending_scan().unwrap();

        state.add_event(create_event("/test/x"));
        state.add_event(create_event("/test/y")); // exceeds the limit of 1

        let errors = state.errors();
        assert_eq!(errors.len(), 1);
        assert!(format!("{errors:?}").contains("buffer limit exceeded"));
        assert!(state.recovery.as_ref().unwrap().rescan_requested);

        state.complete_scan(generation, Ok(snapshot(&[])));
        assert!(
            state.scan_pending,
            "exceeding the buffer limit must force another scan"
        );
    }

    #[test]
    fn normal_debounce_path_is_unaffected_by_recovery_coordinator() {
        MockTime::set_time(Instant::now());

        let mut state = state(16, MockScanner::default());
        set_baseline(&mut state, &[("/test/a", 1)]);

        state.add_event(create_event("/test/new"));
        state.add_event(
            Event::new(EventKind::Modify(ModifyKind::Data(DataChange::Any)))
                .add_path(PathBuf::from("/test/a")),
        );

        // not emitted before the timeout
        assert!(state.debounced_events().is_empty());

        MockTime::advance(Duration::from_millis(100));
        let events = state.debounced_events();
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| event.info().is_none()));
        assert!(!state.scan_pending);
        assert!(state.pending_resync.is_empty());
        assert!(!state.recovery.as_ref().unwrap().scanning);
    }
}
