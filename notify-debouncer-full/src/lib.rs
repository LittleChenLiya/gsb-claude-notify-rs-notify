//! A debouncer for [notify] that is optimized for ease of use.
//!
//! * Only emits a single `Rename` event if the rename `From` and `To` events can be matched
//! * Merges multiple `Rename` events
//! * Takes `Rename` events into account and updates paths for events that occurred before the rename event, but which haven't been emitted, yet
//! * Optionally keeps track of the file system IDs all files and stitches rename events together (macOS FS Events, Windows)
//! * Emits only one `Remove` event when deleting a directory (inotify)
//! * Doesn't emit duplicate create events
//! * Doesn't emit `Modify` events after a `Create` event
//! * Optionally recovers from lost events (overflow) by diffing a fresh directory
//!   snapshot against the last confirmed one and emitting a resync batch, see
//!   [`new_debouncer_opt_with_recovery`]
//!
//! # Installation
//!
//! ```toml
//! [dependencies]
//! notify-debouncer-full = "0.8.0-rc.2"
//! ```
//!
//! In case you want to select specific features of notify,
//! specify notify as dependency explicitly in your dependencies.
//! Otherwise you can just use the re-export of notify from debouncer-full.
//!
//! ```toml
//! notify-debouncer-full = "0.8.0-rc.2"
//! notify = { version = "..", features = [".."] }
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! # use std::path::Path;
//! # use std::time::Duration;
//! use notify_debouncer_full::{notify::*, new_debouncer, DebounceEventResult};
//!
//! // Select recommended watcher for debouncer.
//! // Using a callback here, could also be a channel.
//! let mut debouncer = new_debouncer(Duration::from_secs(2), None, |result: DebounceEventResult| {
//!     match result {
//!         Ok(events) => events.iter().for_each(|event| println!("{event:?}")),
//!         Err(errors) => errors.iter().for_each(|error| println!("{error:?}")),
//!     }
//! }).unwrap();
//!
//! // Add a path to be watched. All files and directories at that path and
//! // below will be monitored for changes.
//! debouncer.watch(".", RecursiveMode::Recursive).unwrap();
//! ```
//!
//! # Features
//!
//! The following crate features can be configured in your Cargo dependency:
//!
//! - `macos_fsevent` (default) enables notify's FSEvents backend on macOS
//! - `freebsd_inotify` enables notify's inotify backend on FreeBSD 14.5+
//!   - inotify is automatically enabled when built natively, this feature is only needed for cross-compilation
//! - `macos_kqueue` enables notify's kqueue backend on macOS
//! - `serde` enables serialization support in notify-types
//! - `web-time` uses `web_time::Instant` for debounced events
//! - `crossbeam-channel`, `flume`, `futures`, and `tokio` enable the corresponding
//!   channel senders as event handlers
//! - `serialization-compat-6` restores notify 6 serialization behavior
//!
//! # Caveats
//!
//! As all file events are sourced from notify, the [known problems](https://docs.rs/notify/latest/notify/#known-problems) section applies here too.

mod cache;
mod resync;
mod time;

#[cfg(test)]
mod testing;

#[cfg(not(target_family = "wasm"))]
mod file_id_map;

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use rustc_hash::FxHashMap as HashMap;
use time::now;

pub use cache::{FileIdCache, NoCache, RecommendedCache};
pub use resync::{
    DirScanner, Fingerprint, RecoveryConfig, Snapshot, SnapshotEntry, WalkDirScanner,
};

use resync::{PendingScan, RecoveryState};

#[cfg(not(target_family = "wasm"))]
pub use file_id_map::FileIdMap;

pub use file_id;
pub use notify;
pub use notify_types::debouncer_full::DebouncedEvent;

use file_id::FileId;
use notify::{
    Error, ErrorKind, Event, EventKind, PathOp, RecommendedWatcher, RecursiveMode,
    UpdatePathsError, Watcher, WatcherKind,
    event::{ModifyKind, RemoveKind, RenameMode},
};

/// The set of requirements for watcher debounce event handling functions.
///
/// # Example implementation
///
/// ```rust,no_run
/// # use notify::{Event, Result, EventHandler};
/// # use notify_debouncer_full::{DebounceEventHandler, DebounceEventResult};
///
/// /// Prints received events
/// struct EventPrinter;
///
/// impl DebounceEventHandler for EventPrinter {
///     fn handle_event(&mut self, result: DebounceEventResult) {
///         match result {
///             Ok(events) => events.iter().for_each(|event| println!("{event:?}")),
///             Err(errors) => errors.iter().for_each(|error| println!("{error:?}")),
///         }
///     }
/// }
/// ```
pub trait DebounceEventHandler: Send + 'static {
    /// Handles an event.
    fn handle_event(&mut self, event: DebounceEventResult);
}

impl<F> DebounceEventHandler for F
where
    F: FnMut(DebounceEventResult) + Send + 'static,
{
    fn handle_event(&mut self, event: DebounceEventResult) {
        (self)(event);
    }
}

#[cfg(feature = "crossbeam-channel")]
impl DebounceEventHandler for crossbeam_channel::Sender<DebounceEventResult> {
    fn handle_event(&mut self, event: DebounceEventResult) {
        let _ = self.send(event);
    }
}

#[cfg(feature = "flume")]
impl DebounceEventHandler for flume::Sender<DebounceEventResult> {
    fn handle_event(&mut self, event: DebounceEventResult) {
        let _ = self.send(event);
    }
}

#[cfg(feature = "futures")]
impl DebounceEventHandler for futures::channel::mpsc::UnboundedSender<DebounceEventResult> {
    fn handle_event(&mut self, event: DebounceEventResult) {
        let _ = self.unbounded_send(event);
    }
}

#[cfg(feature = "tokio")]
impl DebounceEventHandler for tokio::sync::mpsc::UnboundedSender<DebounceEventResult> {
    fn handle_event(&mut self, event: DebounceEventResult) {
        let _ = self.send(event);
    }
}

impl DebounceEventHandler for std::sync::mpsc::Sender<DebounceEventResult> {
    fn handle_event(&mut self, event: DebounceEventResult) {
        let _ = self.send(event);
    }
}

/// A result of debounced events.
/// Comes with either a vec of events or vec of errors.
pub type DebounceEventResult = Result<Vec<DebouncedEvent>, Vec<Error>>;

type DebounceData<T> = Arc<SharedDebounceData<T>>;

#[derive(Debug)]
struct SharedDebounceData<T> {
    inner: Mutex<DebounceDataInner<T>>,
    changed: Condvar,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Queue {
    /// Events must be stored in the following order:
    /// 1. `remove` or `move out` event
    /// 2. `rename` event
    /// 3. Other events
    events: VecDeque<DebouncedEvent>,
}

impl Queue {
    fn was_created(&self) -> bool {
        self.events.front().is_some_and(|event| {
            matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(ModifyKind::Name(RenameMode::To))
            )
        })
    }

    fn was_removed(&self) -> bool {
        self.events.front().is_some_and(|event| {
            matches!(
                event.kind,
                EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(RenameMode::From))
            )
        })
    }
}

#[derive(Debug)]
pub(crate) struct DebounceDataInner<T> {
    queues: HashMap<PathBuf, Queue>,
    /// Registered watch roots, kept **sorted by path** so that `add_root`
    /// can dedupe via binary search in O(log N) and doesn't suffer from injection
    roots: VecDeque<(PathBuf, RecursiveMode)>,
    cache: T,
    rename_event: Option<(DebouncedEvent, Option<FileId>)>,
    rescan_event: Option<DebouncedEvent>,
    errors: Vec<Error>,
    timeout: Duration,
    /// Optional overflow recovery coordinator state.
    recovery: Option<RecoveryState>,
    /// A recovery scan has been requested and waits to be picked up by the
    /// debouncer thread.
    scan_pending: bool,
    /// Resync batch produced by a finished recovery scan, published with the
    /// next debounced events.
    pending_resync: Vec<DebouncedEvent>,
}

impl<T: FileIdCache> DebounceDataInner<T> {
    pub(crate) fn new(cache: T, timeout: Duration) -> Self {
        Self {
            queues: HashMap::default(),
            roots: VecDeque::new(),
            cache,
            rename_event: None,
            rescan_event: None,
            errors: Vec::new(),
            timeout,
            recovery: None,
            scan_pending: false,
            pending_resync: Vec::new(),
        }
    }

    /// Retrieve a vec of debounced events, removing them if not continuous
    pub fn debounced_events(&mut self) -> Vec<DebouncedEvent> {
        // Normal batch publishing is paused while a recovery scan is pending
        // or in flight.
        if self.scan_pending
            || self
                .recovery
                .as_ref()
                .is_some_and(|recovery| recovery.scanning)
        {
            return Vec::new();
        }

        let now = now();
        // A finished resync batch is published immediately.
        let mut events_expired = std::mem::take(&mut self.pending_resync);

        if let Some(event) = self.rescan_event.take() {
            if now.saturating_duration_since(event.time) >= self.timeout {
                log::trace!("debounced event: {event:?}");
                events_expired.push(event);
            } else {
                self.rescan_event = Some(event);
            }
        }

        // Visit each queue in place and remove only the ones that become empty.
        self.queues
            .extract_if(|_, queue| {
                let mut kind_index: HashMap<EventKind, usize> = HashMap::default();
                let mut queue_expired = Vec::new();

                while let Some(event) = queue.events.pop_front() {
                    // remove previous event of the same kind
                    if now.saturating_duration_since(event.time) >= self.timeout {
                        if let Some(idx) = kind_index.insert(event.kind, queue_expired.len()) {
                            queue_expired[idx] = None;
                        }

                        queue_expired.push(Some(event));
                    } else {
                        if let Some(&idx) = kind_index.get(&event.kind) {
                            queue_expired[idx] = None;
                        }
                        queue.events.push_front(event);
                        break;
                    }
                }

                events_expired.extend(queue_expired.into_iter().flatten());

                queue.events.is_empty()
            })
            .for_each(drop);

        sort_events(events_expired)
    }

    /// Returns all currently stored errors
    pub fn errors(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.errors)
    }

    /// Add an error entry to re-send later on
    pub fn add_error(&mut self, error: Error) {
        log::trace!("raw error: {error:?}");
        self.errors.push(error);
    }

    /// Add new event to debouncer cache
    pub fn add_event(&mut self, event: Event) {
        log::trace!("raw event: {event:?}");

        // Overflow recovery in progress: buffer events by generation instead
        // of publishing them, so the in-flight scan cannot race newer state.
        if self
            .recovery
            .as_ref()
            .is_some_and(|recovery| recovery.scanning)
        {
            if event.need_rescan() {
                // Consecutive overflow: the in-flight scan may be based on a
                // state that already lost events, scan again once it finishes.
                self.recovery.as_mut().unwrap().rescan_requested = true;
            } else {
                let recovery = self.recovery.as_mut().unwrap();
                if recovery.buffered.len() >= recovery.max_buffered_events {
                    log::warn!("overflow recovery buffer limit exceeded");
                    self.errors.push(Error::new(ErrorKind::Generic(
                        "overflow recovery buffer limit exceeded".to_string(),
                    )));
                    self.recovery.as_mut().unwrap().rescan_requested = true;
                } else {
                    recovery.buffered.push(event);
                }
            }
            return;
        }

        if event.need_rescan() {
            let roots = self.roots.make_contiguous();
            self.cache.rescan(roots);

            if self.recovery.is_some() {
                // Pause normal batch publishing and recover by diffing a fresh
                // directory snapshot against the last confirmed one.
                self.start_recovery();
            } else {
                self.rescan_event = Some(DebouncedEvent { event, time: now() });
            }
            return;
        }

        let path = match event.paths.first() {
            Some(path) => path,
            None => {
                log::info!("skipping event with no paths: {event:?}");
                return;
            }
        };

        match &event.kind {
            EventKind::Create(_) => {
                let recursive_mode = self.recursive_mode(path);

                self.cache.add_path(path, recursive_mode);

                self.push_event(event, now());
            }
            EventKind::Modify(ModifyKind::Name(rename_mode)) => {
                match rename_mode {
                    RenameMode::Any => {
                        if event.paths[0].exists() {
                            self.handle_rename_to(event);
                        } else {
                            self.handle_rename_from(event);
                        }
                    }
                    RenameMode::To => {
                        self.handle_rename_to(event);
                    }
                    RenameMode::From => {
                        self.handle_rename_from(event);
                    }
                    RenameMode::Both => {
                        // ignore and handle `To` and `From` events instead
                    }
                    RenameMode::Other => {
                        // unused
                    }
                }
            }
            EventKind::Remove(_) => {
                self.push_remove_event(event, now());
            }
            EventKind::Other => {
                // ignore meta events
            }
            _ => {
                if self.cache.cached_file_id(path).is_none() {
                    let recursive_mode = self.recursive_mode(path);

                    self.cache.add_path(path, recursive_mode);
                }

                self.push_event(event, now());
            }
        }
    }

    fn recursive_mode(&self, path: &Path) -> RecursiveMode {
        for ancestor in path.ancestors() {
            if let Ok(index) = self
                .roots
                .binary_search_by(|(root, _)| root.as_path().cmp(ancestor))
            {
                if self.roots[index].1 == RecursiveMode::Recursive {
                    return RecursiveMode::Recursive;
                }
            }
        }

        RecursiveMode::NonRecursive
    }

    /// Start a new recovery generation: pause normal publishing and request a
    /// fresh snapshot scan from the debouncer thread.
    fn start_recovery(&mut self) {
        if let Some(recovery) = &mut self.recovery {
            recovery.generation += 1;
            recovery.scanning = true;
            recovery.rescan_requested = false;
            recovery.buffered.clear();
            self.scan_pending = true;
        }
    }

    /// Take the pending scan request, if any, so the debouncer thread can run
    /// the scanner outside the lock.
    pub(crate) fn take_pending_scan(&mut self) -> Option<PendingScan> {
        if !self.scan_pending {
            return None;
        }
        self.scan_pending = false;

        let recovery = self.recovery.as_ref()?;
        let roots = self.roots.iter().cloned().collect();
        Some((recovery.generation, roots, recovery.scanner.clone()))
    }

    /// Complete the recovery scan for `generation`.
    ///
    /// A late result from an older generation is discarded and never
    /// overwrites newer state. Otherwise the fresh snapshot is diffed against
    /// the last confirmed one, the resulting resync batch is queued for
    /// publishing, and the events buffered during the scan are replayed and
    /// re-checked against disk.
    pub(crate) fn complete_scan(&mut self, generation: u64, result: Result<Snapshot, Error>) {
        {
            let Some(recovery) = &self.recovery else {
                return;
            };
            if !recovery.scanning || recovery.generation != generation {
                log::debug!("discarding stale recovery scan for generation {generation}");
                return;
            }
        }
        self.recovery.as_mut().unwrap().scanning = false;

        match result {
            Ok(snapshot) => {
                let previous = self
                    .recovery
                    .as_mut()
                    .unwrap()
                    .last_confirmed
                    .take()
                    .unwrap_or_default();

                let time = now();
                self.pending_resync.extend(
                    snapshot
                        .diff(&previous)
                        .into_iter()
                        .map(|event| DebouncedEvent::new(event, time)),
                );

                self.recovery.as_mut().unwrap().last_confirmed = Some(snapshot);
            }
            Err(error) => {
                log::warn!("overflow recovery scan failed: {error:?}");
                self.errors.push(error);
            }
        }

        // Replay the events buffered (by generation) during the scan and
        // re-check them against disk, so the confirmed snapshot reflects the
        // newest state and the just-finished scan cannot clobber it.
        self.replay_buffered_events();

        // A consecutive overflow was signalled while scanning: scan again.
        if self
            .recovery
            .as_ref()
            .is_some_and(|recovery| recovery.rescan_requested)
        {
            self.start_recovery();
        }
    }

    fn replay_buffered_events(&mut self) {
        let Some(recovery) = &mut self.recovery else {
            return;
        };
        let buffered = std::mem::take(&mut recovery.buffered);

        for event in buffered {
            self.recheck_confirmed_snapshot(&event);
            self.add_event(event);
        }
    }

    /// Re-check a replayed event against disk and reconcile the confirmed
    /// snapshot, so it always reflects the newest known state.
    fn recheck_confirmed_snapshot(&mut self, event: &Event) {
        let Some(path) = event.paths.first() else {
            return;
        };
        let Some(recovery) = &self.recovery else {
            return;
        };
        let scanner = recovery.scanner.clone();
        let Some(confirmed) = self
            .recovery
            .as_mut()
            .and_then(|recovery| recovery.last_confirmed.as_mut())
        else {
            return;
        };

        match &event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                if let Some(entry) = scanner.fingerprint(path) {
                    confirmed.insert(path.clone(), entry);
                } else {
                    confirmed.remove(path);
                }
            }
            EventKind::Remove(_) => {
                confirmed.remove(path);
            }
            _ => {}
        }
    }

    /// Take the initial baseline snapshot once the first root is watched.
    fn refresh_baseline_snapshot(&mut self) {
        let Some(recovery) = &self.recovery else {
            return;
        };
        if recovery.last_confirmed.is_some() || recovery.scanning {
            return;
        }

        let scanner = recovery.scanner.clone();
        let roots: Vec<_> = self.roots.iter().cloned().collect();
        match scanner.scan(&roots) {
            Ok(snapshot) => {
                self.recovery.as_mut().unwrap().last_confirmed = Some(snapshot);
            }
            Err(error) => {
                self.errors.push(error);
            }
        }
    }

    fn handle_rename_from(&mut self, event: Event) {
        let time = now();
        let path = &event.paths[0];

        // store event
        let file_id = self.cache.cached_file_id(path).map(|id| *id.as_ref());
        self.rename_event = Some((DebouncedEvent::new(event.clone(), time), file_id));

        self.cache.remove_path(path);

        self.push_event(event, time);
    }

    fn handle_rename_to(&mut self, event: Event) {
        let recursive_mode = self.recursive_mode(&event.paths[0]);

        self.cache.add_path(&event.paths[0], recursive_mode);

        let trackers_match = self
            .rename_event
            .as_ref()
            .and_then(|(e, _)| e.tracker())
            .and_then(|from_tracker| {
                event
                    .attrs
                    .tracker()
                    .map(|to_tracker| from_tracker == to_tracker)
            })
            .unwrap_or_default();

        let file_ids_match = self
            .rename_event
            .as_ref()
            .and_then(|(_, id)| id.as_ref())
            .and_then(|from_file_id| {
                self.cache
                    .cached_file_id(&event.paths[0])
                    .map(|to_file_id| from_file_id == to_file_id.as_ref())
            })
            .unwrap_or_default();

        if trackers_match || file_ids_match {
            // connect rename
            let (mut rename_event, _) = self.rename_event.take().unwrap(); // unwrap is safe because `rename_event` must be set at this point
            let path = rename_event.paths.remove(0);
            let time = rename_event.time;
            self.push_rename_event(path, event, time);
        } else {
            // move in
            self.push_event(event, now());
        }

        self.rename_event = None;
    }

    fn push_rename_event(&mut self, path: PathBuf, event: Event, time: Instant) {
        self.cache.remove_path(&path);

        let mut source_queue = self.queues.remove(&path).unwrap_or_default();

        // remove rename `from` event
        source_queue.events.pop_back();

        // remove existing rename event
        let (remove_index, original_path, original_time) = source_queue
            .events
            .iter()
            .enumerate()
            .find_map(|(index, e)| {
                if matches!(
                    e.kind,
                    EventKind::Modify(ModifyKind::Name(RenameMode::Both))
                ) {
                    Some((Some(index), e.paths[0].clone(), e.time))
                } else {
                    None
                }
            })
            .unwrap_or((None, path, time));

        if let Some(remove_index) = remove_index {
            source_queue.events.remove(remove_index);
        }

        // split off remove or move out event and add it back to the events map
        if source_queue.was_removed() {
            let event = source_queue.events.pop_front().unwrap();

            self.queues.insert(
                event.paths[0].clone(),
                Queue {
                    events: [event].into(),
                },
            );
        }

        // update paths
        for e in &mut source_queue.events {
            e.paths = vec![event.paths[0].clone()];
        }

        // insert rename event at the front, unless the file was just created
        if !source_queue.was_created() {
            source_queue.events.push_front(DebouncedEvent {
                event: Event {
                    kind: EventKind::Modify(ModifyKind::Name(RenameMode::Both)),
                    paths: vec![original_path, event.paths[0].clone()],
                    attrs: event.attrs,
                },
                time: original_time,
            });
        }

        if let Some(target_queue) = self.queues.get_mut(&event.paths[0]) {
            if !target_queue.was_created() {
                let mut remove_event = DebouncedEvent {
                    event: Event {
                        kind: EventKind::Remove(RemoveKind::Any),
                        paths: vec![event.paths[0].clone()],
                        attrs: Default::default(),
                    },
                    time: original_time,
                };
                if !target_queue.was_removed() {
                    remove_event.event = remove_event.event.set_info("override");
                }
                source_queue.events.push_front(remove_event);
            }
            *target_queue = source_queue;
        } else {
            self.queues.insert(event.paths[0].clone(), source_queue);
        }
    }

    fn push_remove_event(&mut self, event: Event, time: Instant) {
        let path = &event.paths[0];

        // remove child queues
        self.queues.retain(|p, _| !p.starts_with(path) || p == path);

        // remove cached file ids
        self.cache.remove_path(path);

        match self.queues.get_mut(path) {
            Some(queue) => {
                queue.events = [DebouncedEvent::new(event, time)].into();
            }
            None => {
                self.push_event(event, time);
            }
        }
    }

    fn push_event(&mut self, event: Event, time: Instant) {
        let path = &event.paths[0];

        if let Some(queue) = self.queues.get_mut(path) {
            // Skip duplicate create events and modifications right after creation.
            // This code relies on backends never emitting a `Modify` event with kind other than `Name` for a rename event.
            if match event.kind {
                EventKind::Modify(
                    ModifyKind::Any
                    | ModifyKind::Data(_)
                    | ModifyKind::Metadata(_)
                    | ModifyKind::Other,
                )
                | EventKind::Create(_) => !queue.was_created(),
                _ => true,
            } {
                queue.events.push_back(DebouncedEvent::new(event, time));
            }
        } else {
            self.queues.insert(
                path.to_path_buf(),
                Queue {
                    events: [DebouncedEvent::new(event, time)].into(),
                },
            );
        }
    }
}

/// Debouncer guard, stops the debouncer on drop.
#[derive(Debug)]
pub struct Debouncer<T: Watcher, C: FileIdCache> {
    watcher: T,
    debouncer_thread: Option<std::thread::JoinHandle<()>>,
    data: DebounceData<C>,
    stop: Arc<AtomicBool>,
}

impl<T: Watcher, C: FileIdCache> Debouncer<T, C> {
    /// Stop the debouncer, waits for the event thread to finish.
    /// May block for the duration of one `tick_rate`.
    pub fn stop(mut self) {
        self.set_stop();
        if let Some(t) = self.debouncer_thread.take() {
            let _ = t.join();
        }
    }

    /// Stop the debouncer, does not wait for the event thread to finish.
    pub fn stop_nonblocking(self) {
        self.set_stop();
    }

    fn set_stop(&self) {
        let _lock = self.data.inner.lock().unwrap();
        self.stop.store(true, Ordering::Relaxed);
        self.data.changed.notify_all();
    }

    #[deprecated = "`Debouncer` provides all methods from `Watcher` itself now. Remove `.watcher()` and use those methods directly."]
    pub fn watcher(&mut self) {}

    #[deprecated = "`Debouncer` now manages root paths automatically. Remove all calls to `add_root` and `remove_root`."]
    pub fn cache(&mut self) {}

    fn add_root(&mut self, path: impl Into<PathBuf>, recursive_mode: RecursiveMode) {
        let path = path.into();

        let mut data = self.data.inner.lock().unwrap();

        match data
            .roots
            .binary_search_by(|(p, _)| p.as_path().cmp(path.as_path()))
        {
            Ok(_) => return, // already registered
            Err(pos) => {
                // `VecDeque::insert` is O(min(pos, len - pos))
                data.roots.insert(pos, (path.clone(), recursive_mode));
            }
        }

        data.cache.add_path(&path, recursive_mode);

        data.refresh_baseline_snapshot();
    }

    fn remove_root(&mut self, path: impl AsRef<Path>) {
        let mut data = self.data.inner.lock().unwrap();

        data.roots.retain(|(root, _)| !root.starts_with(&path));

        data.cache.remove_path(path.as_ref());

        if let Some(confirmed) = data
            .recovery
            .as_mut()
            .and_then(|recovery| recovery.last_confirmed.as_mut())
        {
            confirmed.remove(path.as_ref());
        }
    }

    pub fn watch(
        &mut self,
        path: impl AsRef<Path>,
        recursive_mode: RecursiveMode,
    ) -> notify::Result<()> {
        self.watcher.watch(path.as_ref(), recursive_mode)?;
        self.add_root(path.as_ref(), recursive_mode);
        Ok(())
    }

    pub fn unwatch(&mut self, path: impl AsRef<Path>) -> notify::Result<()> {
        self.watcher.unwatch(path.as_ref())?;
        self.remove_root(path);
        Ok(())
    }

    pub fn watched_paths(&self) -> notify::Result<Vec<(PathBuf, RecursiveMode)>> {
        self.watcher.watched_paths()
    }

    /// Add/remove paths to watch in batch.
    ///
    /// For some [`Watcher`] implementations this method provides better performance than multiple
    /// calls to [`Watcher::watch`] and [`Watcher::unwatch`] if you want to add/remove many paths at once.
    ///
    /// # Errors
    ///
    /// Returns [`UpdatePathsError`] if any operation fails. Operations are applied sequentially.
    /// When an error occurs, processing stops: operations before `origin` have been applied,
    /// `origin` is the operation that failed (if known), and `remaining` are the operations that
    /// were not attempted. `remaining` does not include `origin`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use notify::{Watcher, RecursiveMode, PathOp};
    /// # use notify_debouncer_full::{RecommendedCache, new_debouncer_opt};
    /// # use std::path::{Path, PathBuf};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut debouncer = new_debouncer_opt::<_, notify::NullWatcher, _>(
    /// #     std::time::Duration::from_secs(1),
    /// #     None,
    /// #     |e| {},
    /// #     RecommendedCache::new(),
    /// #     Default::default()
    /// # )?;
    /// debouncer.update_paths([
    ///     PathOp::watch_recursive("path/to/file"),
    ///     PathOp::unwatch("path/to/file2"),
    /// ])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_paths<Op: Into<PathOp>>(
        &mut self,
        ops: impl IntoIterator<Item = Op>,
    ) -> std::result::Result<(), UpdatePathsError> {
        let mut paths = Vec::new();
        let ops: Vec<_> = ops
            .into_iter()
            .map(Into::into)
            .inspect(|op| {
                paths.push((
                    op.as_path().to_path_buf(),
                    match op {
                        PathOp::Watch(_, config) => Some(config.recursive_mode()),
                        PathOp::Unwatch(_) => None,
                    },
                ));
            })
            .collect();

        let res = self.watcher.update_paths(ops);
        let updated_len = match res.as_ref() {
            Ok(()) => paths.len(),
            Err(e) => {
                let failed = usize::from(e.origin.is_some());
                paths.len().saturating_sub(e.remaining.len() + failed)
            }
        };
        let updated_paths = &paths[..updated_len];
        for (path, watch_mode) in updated_paths {
            match watch_mode {
                Some(recursive_mode) => self.add_root(path, *recursive_mode),
                None => self.remove_root(path),
            }
        }
        res
    }

    pub fn configure(&mut self, option: notify::Config) -> notify::Result<bool> {
        self.watcher.configure(option)
    }

    #[must_use]
    pub fn kind() -> WatcherKind
    where
        Self: Sized,
    {
        T::kind()
    }
}

impl<T: Watcher, C: FileIdCache> Drop for Debouncer<T, C> {
    fn drop(&mut self) {
        self.set_stop();
    }
}

/// Creates a new debounced watcher with custom configuration.
///
/// Timeout is the amount of time after which a debounced event is emitted.
///
/// If `tick_rate` is `None`, notify will select a tick rate that is 1/4 of the provided timeout.
pub fn new_debouncer_opt<F: DebounceEventHandler, T: Watcher, C: FileIdCache + Send + 'static>(
    timeout: Duration,
    tick_rate: Option<Duration>,
    event_handler: F,
    file_id_cache: C,
    config: notify::Config,
) -> Result<Debouncer<T, C>, Error> {
    new_debouncer_impl(
        timeout,
        tick_rate,
        event_handler,
        file_id_cache,
        config,
        None,
    )
}

/// Creates a new debounced watcher with custom configuration and an overflow
/// recovery coordinator.
///
/// When the watcher reports lost events, the coordinator pauses normal batch
/// emission, scans the watched roots through the injected [`DirScanner`],
/// diffs the fresh snapshot against the last confirmed one and emits a resync
/// batch with per-path creations, removals and content fingerprint changes.
/// Events arriving while a scan is in flight are buffered per generation and
/// replayed (and re-checked) once the scan completes.
///
/// Timeout is the amount of time after which a debounced event is emitted.
///
/// If `tick_rate` is `None`, notify will select a tick rate that is 1/4 of the provided timeout.
pub fn new_debouncer_opt_with_recovery<
    F: DebounceEventHandler,
    T: Watcher,
    C: FileIdCache + Send + 'static,
    S: DirScanner,
>(
    timeout: Duration,
    tick_rate: Option<Duration>,
    event_handler: F,
    file_id_cache: C,
    config: notify::Config,
    recovery: RecoveryConfig<S>,
) -> Result<Debouncer<T, C>, Error> {
    new_debouncer_impl(
        timeout,
        tick_rate,
        event_handler,
        file_id_cache,
        config,
        Some(RecoveryState::new(recovery)),
    )
}

fn new_debouncer_impl<F: DebounceEventHandler, T: Watcher, C: FileIdCache + Send + 'static>(
    timeout: Duration,
    tick_rate: Option<Duration>,
    mut event_handler: F,
    file_id_cache: C,
    config: notify::Config,
    recovery: Option<RecoveryState>,
) -> Result<Debouncer<T, C>, Error> {
    let mut inner = DebounceDataInner::new(file_id_cache, timeout);
    inner.recovery = recovery;

    let data = Arc::new(SharedDebounceData {
        inner: Mutex::new(inner),
        changed: Condvar::new(),
    });
    let stop = Arc::new(AtomicBool::new(false));

    let tick_div = 4;
    let tick = match tick_rate {
        Some(v) => {
            if v > timeout {
                return Err(Error::new(ErrorKind::Generic(format!(
                    "Invalid tick_rate, tick rate {v:?} > {timeout:?} timeout!"
                ))));
            }
            v
        }
        None => timeout.checked_div(tick_div).ok_or_else(|| {
            Error::new(ErrorKind::Generic(format!(
                "Failed to calculate tick as {timeout:?}/{tick_div}!"
            )))
        })?,
    };

    let data_c = data.clone();
    let stop_c = stop.clone();
    let thread = std::thread::Builder::new()
        .name("notify-rs debouncer loop".to_string())
        .spawn(move || {
            loop {
                let mut lock = data_c.inner.lock().unwrap();
                while lock.queues.is_empty()
                    && lock.errors.is_empty()
                    && lock.rescan_event.is_none()
                    && !lock.scan_pending
                    && lock.pending_resync.is_empty()
                    && !stop_c.load(Ordering::Acquire)
                {
                    lock = data_c.changed.wait(lock).unwrap();
                }
                if stop_c.load(Ordering::Acquire) {
                    break;
                }
                // Run a pending recovery scan outside the lock, so the watcher
                // callback can keep buffering events while scanning.
                if let Some((generation, roots, scanner)) = lock.take_pending_scan() {
                    drop(lock);
                    let result = scanner.scan(&roots);
                    let mut lock = data_c.inner.lock().unwrap();
                    lock.complete_scan(generation, result);
                    drop(lock);
                    continue;
                }
                drop(lock);
                std::thread::sleep(tick);
                if stop_c.load(Ordering::Acquire) {
                    break;
                }
                lock = data_c.inner.lock().unwrap();
                let send_data = lock.debounced_events();
                let errors = lock.errors();
                drop(lock);
                if !send_data.is_empty() {
                    event_handler.handle_event(Ok(send_data));
                }
                if !errors.is_empty() {
                    event_handler.handle_event(Err(errors));
                }
            }
        })?;

    let data_c = data.clone();
    let watcher = T::new(
        move |e: Result<Event, Error>| {
            let mut lock = data_c.inner.lock().unwrap();
            match e {
                Ok(e) => lock.add_event(e),
                // can't have multiple TX, so we need to pipe that through our debouncer
                Err(e) => lock.add_error(e),
            }
            data_c.changed.notify_all();
        },
        config,
    )?;

    let guard = Debouncer {
        watcher,
        debouncer_thread: Some(thread),
        data,
        stop,
    };

    Ok(guard)
}

/// Short function to create a new debounced watcher with the recommended debouncer and the built-in file ID cache.
///
/// Timeout is the amount of time after which a debounced event is emitted.
///
/// If `tick_rate` is `None`, notify will select a tick rate that is 1/4 of the provided timeout.
pub fn new_debouncer<F: DebounceEventHandler>(
    timeout: Duration,
    tick_rate: Option<Duration>,
    event_handler: F,
) -> Result<Debouncer<RecommendedWatcher, RecommendedCache>, Error> {
    new_debouncer_opt::<F, RecommendedWatcher, RecommendedCache>(
        timeout,
        tick_rate,
        event_handler,
        RecommendedCache::new(),
        notify::Config::default(),
    )
}

fn sort_events(events: Vec<DebouncedEvent>) -> Vec<DebouncedEvent> {
    let mut sorted = Vec::with_capacity(events.len());

    // group events by path
    let mut groups = Vec::<(PathBuf, VecDeque<DebouncedEvent>)>::new();
    let mut group_indexes: HashMap<PathBuf, usize> = HashMap::default();
    group_indexes.reserve(events.len());
    groups.reserve(events.len());

    for event in events {
        let path = event.paths.last().cloned().unwrap_or_default();

        if let Some(&index) = group_indexes.get(&path) {
            groups[index].1.push_back(event);
        } else {
            group_indexes.insert(path.clone(), groups.len());
            groups.push((path, [event].into()));
        }
    }

    // Keep path order as the tie-breaker for identical timestamps.
    groups.sort_unstable_by(|(left_path, _), (right_path, _)| left_path.cmp(right_path));

    // push events for different paths in chronological order and keep the order of events with the same path

    let mut min_time_heap = groups
        .iter()
        .enumerate()
        .map(|(index, (_, events))| Reverse((events[0].time, index)))
        .collect::<BinaryHeap<_>>();

    while let Some(Reverse((min_time, index))) = min_time_heap.pop() {
        let events = &mut groups[index].1;

        let mut push_next = false;

        while events.front().is_some_and(|event| event.time <= min_time) {
            // unwrap is safe because `pop_front` mus return some in order to enter the loop
            let event = events.pop_front().unwrap();
            sorted.push(event);
            push_next = true;
        }

        if push_next {
            if let Some(event) = events.front() {
                min_time_heap.push(Reverse((event.time, index)));
            }
        }
    }

    sorted
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::Duration,
    };

    use super::*;

    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use tempfile::tempdir;
    use testing::TestCase;
    use time::MockTime;

    #[derive(Debug)]
    struct FailingWatcher {
        fail_path: PathBuf,
    }

    impl Watcher for FailingWatcher {
        fn new<F: notify::EventHandler>(
            _event_handler: F,
            _config: notify::Config,
        ) -> notify::Result<Self> {
            Ok(Self {
                fail_path: PathBuf::from("bad"),
            })
        }

        fn watch(&mut self, path: &Path, _recursive_mode: RecursiveMode) -> notify::Result<()> {
            if path == self.fail_path {
                Err(Error::path_not_found())
            } else {
                Ok(())
            }
        }

        fn unwatch(&mut self, path: &Path) -> notify::Result<()> {
            if path == self.fail_path {
                Err(Error::path_not_found())
            } else {
                Ok(())
            }
        }

        fn kind() -> WatcherKind {
            WatcherKind::NullWatcher
        }
    }

    #[derive(Debug, Default)]
    struct TrackingWatcher {
        watched: Vec<(PathBuf, RecursiveMode)>,
    }

    impl Watcher for TrackingWatcher {
        fn new<F: notify::EventHandler>(
            _event_handler: F,
            _config: notify::Config,
        ) -> notify::Result<Self> {
            Ok(Self::default())
        }

        fn watch(&mut self, path: &Path, recursive_mode: RecursiveMode) -> notify::Result<()> {
            self.watched.push((path.to_path_buf(), recursive_mode));
            Ok(())
        }

        fn unwatch(&mut self, path: &Path) -> notify::Result<()> {
            let original_len = self.watched.len();
            self.watched
                .retain(|(watched_path, _)| watched_path != path);

            if self.watched.len() == original_len {
                Err(Error::watch_not_found())
            } else {
                Ok(())
            }
        }

        fn kind() -> WatcherKind {
            WatcherKind::NullWatcher
        }

        fn watched_paths(&self) -> notify::Result<Vec<(PathBuf, RecursiveMode)>> {
            Ok(self.watched.clone())
        }
    }

    #[rstest]
    fn state(
        #[values(
            "add_create_and_remove_event",
            "add_create_event",
            "add_create_event_after_remove_event",
            "add_create_dir_event_twice",
            "add_event_with_no_paths_is_ok",
            "add_modify_any_event_after_create_event",
            "add_modify_content_event_after_create_event",
            "add_rename_from_event",
            "add_rename_from_event_after_create_event",
            "add_rename_from_event_after_modify_event",
            "add_rename_from_event_after_create_and_modify_event",
            "add_rename_from_event_after_rename_from_event",
            "add_rename_to_event",
            "add_rename_to_dir_event",
            "add_rename_from_and_to_event",
            "add_rename_from_and_to_event_after_create",
            "add_rename_from_and_to_event_after_rename",
            "add_rename_from_and_to_event_after_modify_content",
            "add_rename_from_and_to_event_override_created",
            "add_rename_from_and_to_event_override_modified",
            "add_rename_from_and_to_event_override_removed",
            "add_rename_from_and_to_event_with_file_ids",
            "add_rename_from_and_to_event_with_different_file_ids",
            "add_rename_from_and_to_event_with_different_tracker",
            "add_rename_both_event",
            "add_remove_event",
            "add_remove_event_after_create_event",
            "add_remove_event_after_modify_event",
            "add_remove_event_after_create_and_modify_event",
            "add_remove_parent_event_after_remove_child_event",
            "add_errors",
            "debounce_modify_events",
            "emit_continuous_modify_content_events",
            "emit_events_in_chronological_order",
            "emit_events_with_a_prepended_rename_event",
            "emit_close_events_only_once",
            "emit_modify_event_after_close_event",
            "emit_needs_rescan_event",
            "read_file_id_without_create_event",
            "sort_events_chronologically",
            "sort_events_with_reordering"
        )]
        file_name: &str,
    ) {
        let file_content =
            fs::read_to_string(Path::new(&format!("./test_cases/{file_name}.hjson"))).unwrap();
        let mut test_case = deser_hjson::from_str::<TestCase>(&file_content).unwrap();

        let time = now();
        MockTime::set_time(time);

        let mut state = test_case.state.into_debounce_data_inner(time);
        state.roots = VecDeque::from([(PathBuf::from("/"), RecursiveMode::Recursive)]);

        let mut prev_event_time = Duration::default();

        for event in test_case.events {
            let event_time = Duration::from_millis(event.time);
            let event = event.into_debounced_event(time, None);
            MockTime::advance(event_time - prev_event_time);
            prev_event_time = event_time;
            state.add_event(event.event);
        }

        for error in test_case.errors {
            let error = error.into_notify_error();
            state.add_error(error);
        }

        let expected_errors = std::mem::take(&mut test_case.expected.errors);
        let expected_events = std::mem::take(&mut test_case.expected.events);
        let expected_state = test_case.expected.into_debounce_data_inner(time);
        assert_eq!(
            state.queues, expected_state.queues,
            "queues not as expected"
        );
        assert_eq!(
            state.rename_event, expected_state.rename_event,
            "rename event not as expected"
        );
        assert_eq!(
            state.rescan_event, expected_state.rescan_event,
            "rescan event not as expected"
        );
        assert_eq!(
            state.cache.paths, expected_state.cache.paths,
            "cache not as expected"
        );

        assert_eq!(
            state
                .errors
                .iter()
                .map(|e| format!("{e:?}"))
                .collect::<Vec<_>>(),
            expected_errors
                .iter()
                .map(|e| format!("{:?}", e.clone().into_notify_error()))
                .collect::<Vec<_>>(),
            "errors not as expected"
        );

        let backup_time = now();
        let backup_queues = state.queues.clone();

        for (delay, events) in expected_events {
            MockTime::set_time(backup_time);
            state.queues = backup_queues.clone();

            match delay.as_str() {
                "none" => {}
                "short" => MockTime::advance(Duration::from_millis(10)),
                "long" => MockTime::advance(Duration::from_millis(100)),
                _ => {
                    if let Ok(ts) = delay.parse::<u64>() {
                        MockTime::set_time(time + Duration::from_millis(ts));
                    }
                }
            }

            let events = events
                .into_iter()
                .map(|event| event.into_debounced_event(time, None))
                .collect::<Vec<_>>();

            assert_eq!(
                state.debounced_events(),
                events,
                "debounced events after a `{delay}` delay"
            );
        }
    }

    #[test]
    fn recursive_mode_uses_recursive_root_for_overlapping_watches() {
        let state = DebounceDataInner {
            queues: HashMap::default(),
            roots: VecDeque::from([
                (PathBuf::from("root"), RecursiveMode::NonRecursive),
                (PathBuf::from("root/nested"), RecursiveMode::Recursive),
            ]),
            cache: NoCache,
            rename_event: None,
            rescan_event: None,
            errors: Vec::new(),
            timeout: Duration::from_millis(50),
            recovery: None,
            scan_pending: false,
            pending_resync: Vec::new(),
        };

        assert_eq!(
            state.recursive_mode(Path::new("root/nested/child")),
            RecursiveMode::Recursive
        );
        assert_eq!(
            state.recursive_mode(Path::new("root/other")),
            RecursiveMode::NonRecursive
        );
    }

    #[test]
    fn sort_events_ties_by_path() {
        let time = now();
        let events = vec![
            DebouncedEvent::new(
                Event::new(EventKind::Any).add_path(PathBuf::from("/watch/b")),
                time,
            ),
            DebouncedEvent::new(
                Event::new(EventKind::Any).add_path(PathBuf::from("/watch/a")),
                time,
            ),
        ];

        let sorted = sort_events(events);
        let paths = sorted
            .into_iter()
            .map(|event| event.paths[0].clone())
            .collect::<Vec<_>>();

        assert_eq!(
            paths,
            vec![PathBuf::from("/watch/a"), PathBuf::from("/watch/b")]
        );
    }

    #[test]
    fn integration() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;

        // set up the watcher
        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = new_debouncer(Duration::from_millis(10), None, tx)?;
        debouncer.watch(dir.path(), RecursiveMode::Recursive)?;

        // create a new file
        let file_path = dir.path().join("file.txt");
        fs::write(&file_path, b"Lorem ipsum")?;

        println!("waiting for event at {}", file_path.display());

        // wait for up to 10 seconds for the create event, ignore all other events
        let deadline = Instant::now() + Duration::from_secs(10);
        while deadline > Instant::now() {
            let events = rx
                .recv_timeout(deadline - Instant::now())
                .expect("did not receive expected event")
                .expect("received an error");

            for event in events {
                if event.event.paths == vec![file_path.clone()]
                    || event.event.paths == vec![file_path.canonicalize()?]
                {
                    return Ok(());
                }

                println!("unexpected event: {event:?}");
            }
        }

        panic!("did not receive expected event");
    }

    /// End-to-end test of the overflow recovery coordinator on the real
    /// debouncer thread, with a scripted scanner:
    ///
    /// 1. a delete and a create are lost → the resync batch must recover both
    /// 2. a file created mid-scan must end up in the final confirmed snapshot
    #[test]
    fn integration_overflow_recovery() -> Result<(), Box<dyn std::error::Error>> {
        use notify::event::{CreateKind, Flag};
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        /// Scanner that snapshots the real file system, but holds its second
        /// scan open so the test can create a file mid-scan.
        #[derive(Debug)]
        struct HookScanner {
            scan_count: AtomicUsize,
            started_tx: std::sync::mpsc::Sender<usize>,
            continue_rx: Mutex<std::sync::mpsc::Receiver<()>>,
        }

        impl DirScanner for HookScanner {
            fn scan(&self, roots: &[(PathBuf, RecursiveMode)]) -> Result<Snapshot, Error> {
                // snapshot first, so a file created while we block below is
                // deliberately missing from this scan's result
                let snapshot = WalkDirScanner::new().scan(roots)?;
                let n = self.scan_count.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                self.started_tx.send(n).unwrap();
                if n == 2 {
                    self.continue_rx.lock().unwrap().recv().unwrap();
                }
                Ok(snapshot)
            }

            fn fingerprint(&self, path: &Path) -> Option<SnapshotEntry> {
                resync::fingerprint_path(path)
            }
        }

        let dir = tempdir()?;
        let root = dir.path().to_path_buf();
        fs::write(root.join("a.txt"), b"a")?;
        fs::write(root.join("b.txt"), b"b")?;

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        let scanner = HookScanner {
            scan_count: AtomicUsize::new(0),
            started_tx,
            continue_rx: Mutex::new(continue_rx),
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = new_debouncer_opt_with_recovery::<_, notify::NullWatcher, NoCache, _>(
            Duration::from_millis(50),
            Some(Duration::from_millis(10)),
            tx,
            NoCache::new(),
            notify::Config::default(),
            RecoveryConfig::new(scanner).with_max_buffered_events(16),
        )?;
        debouncer.watch(&root, RecursiveMode::Recursive)?;

        // the baseline snapshot is taken synchronously during `watch`
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(5))?, 1);
        {
            let data = debouncer.data.inner.lock().unwrap();
            let confirmed = data
                .recovery
                .as_ref()
                .and_then(|recovery| recovery.last_confirmed.as_ref())
                .expect("baseline snapshot");
            assert!(confirmed.contains_path(&root.join("a.txt")));
            assert!(confirmed.contains_path(&root.join("b.txt")));
        }

        // silently lose a delete and a create (NullWatcher reports nothing)
        fs::remove_file(root.join("a.txt"))?;
        fs::write(root.join("c.txt"), b"c")?;

        // signal the event loss
        {
            let mut data = debouncer.data.inner.lock().unwrap();
            data.add_event(Event::new(EventKind::Other).set_flag(Flag::Rescan));
        }
        debouncer.data.changed.notify_all();

        // wait for the recovery scan to start, then create another file
        // mid-scan and inject its event (it must be buffered and replayed)
        assert_eq!(started_rx.recv_timeout(Duration::from_secs(5))?, 2);
        fs::write(root.join("d.txt"), b"d")?;
        {
            let mut data = debouncer.data.inner.lock().unwrap();
            data.add_event(
                Event::new(EventKind::Create(CreateKind::File)).add_path(root.join("d.txt")),
            );
        }
        debouncer.data.changed.notify_all();
        continue_tx.send(())?;

        // collect events: the resync batch must recover the lost delete and
        // create, and the mid-scan file must be replayed afterwards
        let deadline = Instant::now() + Duration::from_secs(10);
        let (mut remove_a, mut create_c, mut create_d) = (false, false, false);
        while !(remove_a && create_c && create_d) && Instant::now() < deadline {
            let events = rx
                .recv_timeout(deadline - Instant::now())
                .expect("did not receive expected events")
                .expect("received an error");

            for event in events {
                println!("event: {event:?}");
                let resync = event.info() == Some(resync::RESYNC_INFO);
                if resync
                    && matches!(event.kind, EventKind::Remove(_))
                    && event.paths.contains(&root.join("a.txt"))
                {
                    remove_a = true;
                }
                if resync
                    && matches!(event.kind, EventKind::Create(_))
                    && event.paths.contains(&root.join("c.txt"))
                {
                    create_c = true;
                }
                if matches!(event.kind, EventKind::Create(_))
                    && event.paths.contains(&root.join("d.txt"))
                {
                    create_d = true;
                }
            }
        }
        assert!(remove_a, "resync batch must recover the lost delete");
        assert!(create_c, "resync batch must recover the lost create");
        assert!(create_d, "mid-scan create must be replayed");

        // the final confirmed snapshot must contain the mid-scan file and
        // must not contain the deleted file
        let data = debouncer.data.inner.lock().unwrap();
        let confirmed = data
            .recovery
            .as_ref()
            .and_then(|recovery| recovery.last_confirmed.as_ref())
            .expect("confirmed snapshot");
        assert!(!confirmed.contains_path(&root.join("a.txt")));
        assert!(confirmed.contains_path(&root.join("b.txt")));
        assert!(confirmed.contains_path(&root.join("c.txt")));
        assert!(confirmed.contains_path(&root.join("d.txt")));

        Ok(())
    }

    /// Regression test: on macOS, FSEvents reports file deletions as a burst
    /// of `Create(File)` + `Modify(Data)` + `Remove(File)`.  The debouncer
    /// must not suppress the `Remove` event, even though a prior `Create`
    /// for the same path exists in the queue.
    ///
    /// Without the fix, `push_remove_event` would see `was_created() == true`
    /// and cancel the entire queue, swallowing the removal.
    #[test]
    #[cfg(all(target_os = "macos", feature = "macos_fsevent"))]
    fn remove_event_not_swallowed_after_create() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let dir_path = dir.path().canonicalize()?;

        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = new_debouncer(Duration::from_millis(10), None, tx)?;
        debouncer.watch(&dir_path, RecursiveMode::NonRecursive)?;

        // Create a file and wait for the debouncer to deliver the Create event.
        let file_path = dir_path.join("ephemeral.txt");
        fs::write(&file_path, b"will be deleted")?;

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut got_create = false;
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(events)) => {
                    if events.iter().any(|e| {
                        matches!(e.event.kind, EventKind::Create(_))
                            && e.event.paths.contains(&file_path)
                    }) {
                        got_create = true;
                        break;
                    }
                }
                Ok(Err(_)) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            }
        }
        assert!(got_create, "expected Create event for ephemeral.txt");

        // Drain any remaining events from the creation.
        std::thread::sleep(Duration::from_millis(200));
        while rx.try_recv().is_ok() {}

        // Delete the file.
        fs::remove_file(&file_path)?;

        // The debouncer MUST deliver an event for this path (Remove, or at
        // minimum any event whose path matches so the consumer can stat it).
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut got_removal = false;
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(events)) => {
                    if events.iter().any(|e| e.event.paths.contains(&file_path)) {
                        got_removal = true;
                        break;
                    }
                }
                Ok(Err(_)) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            }
        }
        assert!(
            got_removal,
            "expected Remove (or any) event for deleted ephemeral.txt, got none within 5s"
        );

        Ok(())
    }

    /// Unit-level reproducer for the same bug: feed a Create + Remove
    /// sequence into `DebounceDataInner` directly, with the queue already
    /// flushed between them (simulating the debounce tick).  The Remove
    /// must not be swallowed.
    #[test]
    #[cfg(target_os = "macos")]
    fn push_remove_after_flushed_create() {
        use crate::NoCache;
        use notify::event::{CreateKind, RemoveKind};
        use std::path::PathBuf;

        let mut state = DebounceDataInner::new(NoCache, Duration::from_millis(50));

        let time = std::time::Instant::now();
        MockTime::set_time(time);

        let path = PathBuf::from("/tmp/test_file.txt");

        // Simulate: file created → Create event added
        state.add_event(Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![path.clone()],
            ..Default::default()
        });

        // Simulate: debounce tick flushes the Create
        MockTime::advance(Duration::from_millis(100));
        let flushed = state.debounced_events();
        assert_eq!(flushed.len(), 1);
        assert!(matches!(flushed[0].event.kind, EventKind::Create(_)));

        // Simulate: FSEvents sends Create + Remove for the deletion
        // (this is what macOS does)
        state.add_event(Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![path.clone()],
            ..Default::default()
        });
        state.add_event(Event {
            kind: EventKind::Remove(RemoveKind::File),
            paths: vec![path.clone()],
            ..Default::default()
        });

        // Flush again — we MUST get the Remove event
        MockTime::advance(Duration::from_millis(100));
        let flushed = state.debounced_events();
        assert!(
            flushed
                .iter()
                .any(|e| matches!(e.event.kind, EventKind::Remove(_))),
            "expected Remove event after flushed Create, got: {flushed:?}"
        );
    }

    #[cfg(feature = "futures")]
    #[tokio::test]
    async fn futures_unbounded_sender_as_handler() {
        use futures::StreamExt;

        let dir = tempdir().unwrap();

        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        let mut debouncer = new_debouncer(Duration::from_millis(10), None, tx).unwrap();
        debouncer
            .watch(dir.path(), RecursiveMode::Recursive)
            .unwrap();

        let file_path = dir.path().join("file.txt");
        fs::write(&file_path, b"Lorem ipsum").unwrap();

        tokio::time::timeout(Duration::from_secs(10), rx.next())
            .await
            .expect("timeout")
            .expect("No event")
            .expect("error");
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn tokio_unbounded_sender_as_handler() {
        let dir = tempdir().unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut debouncer = new_debouncer(Duration::from_millis(10), None, tx).unwrap();
        debouncer
            .watch(dir.path(), RecursiveMode::Recursive)
            .unwrap();

        let file_path = dir.path().join("file.txt");
        fs::write(&file_path, b"Lorem ipsum").unwrap();

        tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("timeout")
            .expect("No event")
            .expect("error");
    }

    #[test]
    fn update_paths() -> Result<(), Box<dyn std::error::Error>> {
        let dir1 = tempdir()?;
        let dir2 = tempdir()?;

        // set up the watcher
        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = new_debouncer(Duration::from_millis(10), None, tx)?;
        debouncer.update_paths([
            PathOp::watch_recursive(dir1.path()),
            PathOp::watch_recursive(dir2.path()),
        ])?;

        // create a new file
        let file_path1 = dir1.path().join("file.txt");
        let file_path2 = dir2.path().join("file.txt");
        fs::write(&file_path1, b"Lorem ipsum1")?;
        fs::write(&file_path2, b"Lorem ipsum1")?;

        println!("waiting for events at {file_path1:?} and {file_path2:?}");

        // wait for up to 10 seconds for the create event, ignore all other events
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut received = (false, false);
        while deadline > Instant::now() {
            let events = rx
                .recv_timeout(deadline - Instant::now())
                .expect("did not receive expected event")
                .expect("received an error");

            for event in events {
                println!("event {event:?}");
                if event.event.paths == vec![file_path1.clone()]
                    || event.event.paths == vec![file_path1.canonicalize()?]
                {
                    received.0 = true;
                }

                if event.event.paths == vec![file_path2.clone()]
                    || event.event.paths == vec![file_path2.canonicalize()?]
                {
                    received.1 = true;
                }

                if received == (true, true) {
                    return Ok(());
                }
            }
        }

        panic!("did not receive expected event");
    }

    #[test]
    fn update_paths_error_does_not_add_failed_root() -> Result<(), Box<dyn std::error::Error>> {
        let mut debouncer = new_debouncer_opt::<_, FailingWatcher, NoCache>(
            Duration::from_millis(20),
            Some(Duration::from_millis(5)),
            |_| {},
            NoCache::new(),
            notify::Config::default(),
        )?;

        let err = debouncer
            .update_paths([
                PathOp::watch_recursive("ok1"),
                PathOp::watch_recursive("bad"),
                PathOp::watch_recursive("ok2"),
            ])
            .unwrap_err();
        assert!(err.origin.is_some());
        assert_eq!(err.remaining.len(), 1);

        let roots = debouncer.data.inner.lock().unwrap().roots.clone();
        assert_eq!(
            roots,
            VecDeque::from([(PathBuf::from("ok1"), RecursiveMode::Recursive)])
        );

        Ok(())
    }

    #[test]
    fn watched_paths_with_watch_update_paths_and_unwatch() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut debouncer = new_debouncer_opt::<_, TrackingWatcher, NoCache>(
            Duration::from_millis(20),
            Some(Duration::from_millis(5)),
            |_| {},
            NoCache::new(),
            notify::Config::default(),
        )?;

        let path1 = PathBuf::from("one");
        let path2 = PathBuf::from("two");
        let path3 = PathBuf::from("three");

        assert!(debouncer.watched_paths()?.is_empty());

        debouncer.watch(&path1, RecursiveMode::Recursive)?;
        assert_eq!(
            debouncer.watched_paths()?,
            vec![(path1.clone(), RecursiveMode::Recursive)]
        );

        debouncer.update_paths([
            PathOp::unwatch(&path1),
            PathOp::watch_non_recursive(path2.clone()),
            PathOp::watch_recursive(path3.clone()),
        ])?;
        assert_eq!(
            debouncer.watched_paths()?,
            vec![
                (path2.clone(), RecursiveMode::NonRecursive),
                (path3.clone(), RecursiveMode::Recursive),
            ]
        );

        debouncer.unwatch(&path2)?;
        assert_eq!(
            debouncer.watched_paths()?,
            vec![(path3, RecursiveMode::Recursive)]
        );

        Ok(())
    }
}
