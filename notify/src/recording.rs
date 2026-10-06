//! Recording and replaying raw file events for offline debugging.
//!
//! When investigating a problem it is often useful to capture a short burst of
//! file events as they arrive, and later replay them against the same handling
//! code *without* starting a real watcher. This module provides exactly that:
//!
//! - [`Recorder`] writes events to any [`io::Write`] sink in a compact,
//!   self-contained binary format. Each record stores the raw [`EventKind`],
//!   the full path list, the tracker id (if any), and the time of the event
//!   relative to the previous record.
//! - [`RecordReader`] lazily decodes a recording back into [`RecordedEvent`]s.
//!   It reads exactly one record at a time, so stopping early never consumes
//!   the rest of the recording.
//! - [`Replayer`] drives a recording into your handler, honouring the recorded
//!   relative timing through a pluggable [`Clock`]. Use [`SystemClock`] for
//!   real-time replay or [`ManualClock`] (a fake clock) for deterministic
//!   tests.
//!
//! Paths are stored as their raw, platform-native encoded bytes
//! ([`OsStr::as_encoded_bytes`]), so non-UTF-8 paths survive a round trip
//! byte-for-byte and are never mangled into replacement characters.
//!
//! # Format
//!
//! All integers are little-endian.
//!
//! ```text
//! header  := "NFREC" u8(version = 1)
//! record  := u64(delta_micros) tracker kind u32(path_count) path*
//! tracker := u8(0) | u8(1) u64(value)
//! path    := u32(len) u8[len]   // raw OsStr encoded bytes
//! ```
//!
//! `delta_micros` is the number of microseconds between this event and the
//! previous one (or the start of the recording for the first event), so two
//! events with the same timestamp are stored with a zero delta.
//!
//! # Example
//!
//! ```
//! use notify::event::{CreateKind, EventKind};
//! use notify::recording::{ManualClock, Recorder, RecordReader, Replayer};
//! use notify::Event;
//! use std::time::Duration;
//!
//! // Record two events, 5ms apart, against a fake clock.
//! let mut clock = ManualClock::new();
//! let mut recorder = Recorder::with_clock(Vec::new(), clock.clone()).unwrap();
//! recorder.record(&Event::new(EventKind::Create(CreateKind::File)).add_path("a.txt".into())).unwrap();
//! clock.advance_by(Duration::from_millis(5));
//! recorder.record(&Event::new(EventKind::Any).add_path("a.txt".into())).unwrap();
//! let bytes = recorder.into_inner();
//!
//! // Replay them back with a fake clock: no watcher, no real sleeping.
//! let reader = RecordReader::new(bytes.as_slice()).unwrap();
//! let mut seen = Vec::new();
//! let mut replayer = Replayer::new(reader, ManualClock::new());
//! let stats = replayer
//!     .run(|recorded| {
//!         seen.push(recorded.event);
//!         notify::recording::ReplayControl::Continue
//!     })
//!     .unwrap();
//! assert_eq!(stats.delivered, 2);
//! assert_eq!(seen.len(), 2);
//! ```

use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use notify_types::event::{
    AccessKind, AccessMode, CreateKind, DataChange, MetadataKind, ModifyKind, RemoveKind,
    RenameMode,
};

use crate::{Event, EventHandler, EventKind};

/// Magic bytes at the start of every recording, including the format version.
const MAGIC: &[u8; 6] = b"NFREC\x01";

/// Sentinel limit guarding against corrupt path-count prefixes.
const MAX_PATHS_PER_EVENT: u32 = 1 << 16;
/// Maximum byte length accepted for a single encoded path.
const MAX_PATH_BYTES: u32 = 1 << 20;

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// A source of time for recording and replaying events.
///
/// The clock only ever deals in relative [`Duration`]s measured from an
/// arbitrary epoch chosen by the implementation, so recordings stay
/// comparable regardless of wall-clock time.
pub trait Clock {
    /// Returns the current time relative to the clock's epoch.
    fn now(&mut self) -> Duration;

    /// Blocks (or pretends to block) until `deadline` is reached.
    ///
    /// Implementations must tolerate deadlines in the past by returning
    /// immediately.
    fn wait_until(&mut self, deadline: Duration);
}

/// A [`Clock`] backed by [`Instant`], for recording live events and replaying
/// them in real time.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock {
    start: Option<Instant>,
}

impl SystemClock {
    /// Creates a new system clock.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Clock for SystemClock {
    fn now(&mut self) -> Duration {
        self.start.get_or_insert_with(Instant::now).elapsed()
    }

    fn wait_until(&mut self, deadline: Duration) {
        let now = self.now();
        if let Some(remaining) = deadline.checked_sub(now) {
            std::thread::sleep(remaining);
        }
    }
}

/// A fake [`Clock`] driven entirely by the caller, for deterministic replays
/// and tests. Waiting simply jumps the clock forward; nothing ever sleeps.
#[derive(Debug, Default, Clone)]
pub struct ManualClock {
    now: Duration,
    waits: Vec<Duration>,
}

impl ManualClock {
    /// Creates a new manual clock set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current fake time.
    #[must_use]
    pub fn time(&self) -> Duration {
        self.now
    }

    /// Returns every deadline this clock was asked to wait for, in order.
    ///
    /// Useful for asserting that a replay waited for exactly the recorded
    /// timestamps.
    #[must_use]
    pub fn waits(&self) -> &[Duration] {
        &self.waits
    }

    /// Advances the fake time by `delta`.
    pub fn advance_by(&mut self, delta: Duration) {
        self.now += delta;
    }

    /// Advances the fake time to `time` (never backwards).
    pub fn advance_to(&mut self, time: Duration) {
        self.now = self.now.max(time);
    }
}

impl Clock for ManualClock {
    fn now(&mut self) -> Duration {
        self.now
    }

    fn wait_until(&mut self, deadline: Duration) {
        self.waits.push(deadline);
        self.advance_to(deadline);
    }
}

/// Writes events to a sink in the recording format.
///
/// The header is written on construction; every call to [`Recorder::record`]
/// (or [`Recorder::record_at`]) appends one record. Timestamps are taken from
/// the recorder's [`Clock`]; only the delta to the previous event is stored.
///
/// `Recorder` implements [`EventHandler`], so it can be attached directly to
/// a watcher (or composed with a fan-out handler) to capture live traffic.
/// Error events are ignored by that implementation.
pub struct Recorder<W, C = SystemClock> {
    writer: W,
    clock: C,
    last: Duration,
    count: u64,
}

impl<W: Write> Recorder<W, SystemClock> {
    /// Creates a recorder that timestamps events with a [`SystemClock`].
    pub fn new(writer: W) -> io::Result<Self> {
        Self::with_clock(writer, SystemClock::new())
    }
}

impl<W: Write, C: Clock> Recorder<W, C> {
    /// Creates a recorder using the given clock, writing the header
    /// immediately.
    pub fn with_clock(mut writer: W, clock: C) -> io::Result<Self> {
        writer.write_all(MAGIC)?;
        Ok(Self {
            writer,
            clock,
            last: Duration::ZERO,
            count: 0,
        })
    }

    /// Records an event, timestamped with the recorder's clock.
    pub fn record(&mut self, event: &Event) -> io::Result<()> {
        let at = self.clock.now();
        self.record_at(at, event)
    }

    /// Records an event with an explicit timestamp.
    ///
    /// Timestamps are expected to be monotonic; a timestamp earlier than the
    /// previous one is clamped to a zero delta.
    pub fn record_at(&mut self, at: Duration, event: &Event) -> io::Result<()> {
        let delta = at.saturating_sub(self.last);
        self.last = self.last.max(at);

        let delta_micros = u64::try_from(delta.as_micros()).unwrap_or(u64::MAX);
        self.writer.write_all(&delta_micros.to_le_bytes())?;

        match event.tracker() {
            None => self.writer.write_all(&[0])?,
            Some(tracker) => {
                self.writer.write_all(&[1])?;
                let tracker = u64::try_from(tracker)
                    .map_err(|_| invalid_data("tracker id does not fit into u64"))?;
                self.writer.write_all(&tracker.to_le_bytes())?;
            }
        }

        write_kind(&mut self.writer, &event.kind)?;

        let path_count = u32::try_from(event.paths.len())
            .map_err(|_| invalid_data("too many paths on event"))?;
        self.writer.write_all(&path_count.to_le_bytes())?;
        for path in &event.paths {
            let bytes = path.as_os_str().as_encoded_bytes();
            let len =
                u32::try_from(bytes.len()).map_err(|_| invalid_data("path too long to record"))?;
            self.writer.write_all(&len.to_le_bytes())?;
            self.writer.write_all(bytes)?;
        }

        self.count += 1;
        Ok(())
    }

    /// Returns how many events have been recorded so far.
    #[must_use]
    pub fn recorded_count(&self) -> u64 {
        self.count
    }

    /// Flushes the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Consumes the recorder and returns the underlying writer.
    #[must_use]
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W, C> EventHandler for Recorder<W, C>
where
    W: Write + Send + 'static,
    C: Clock + Send + 'static,
{
    fn handle_event(&mut self, event: crate::Result<Event>) {
        if let Ok(event) = event {
            // Recording must never panic a watcher's event loop; a broken
            // sink simply stops capturing.
            let _ = self.record(&event);
        }
    }
}

/// A single event decoded from a recording, with its timestamp relative to
/// the start of the recording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedEvent {
    /// Time of the event relative to the start of the recording.
    pub at: Duration,
    /// The recorded event.
    pub event: Event,
}

/// Lazily decodes a recording, one record at a time.
///
/// The reader is fully streaming: it only ever consumes the bytes of the
/// record currently being decoded. Dropping it (or stopping iteration early)
/// therefore never reads the remainder of the recording, and a truncated
/// final record is reported as an [`io::ErrorKind::UnexpectedEof`] error only
/// once that record is reached.
pub struct RecordReader<R> {
    reader: R,
    time: Duration,
    failed: bool,
}

impl<R: Read> RecordReader<R> {
    /// Creates a reader, consuming and validating the recording header.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                invalid_data("recording is too short to contain a header")
            } else {
                e
            }
        })?;
        if magic != *MAGIC {
            return Err(invalid_data(
                "not a notify event recording (bad magic or version)",
            ));
        }
        Ok(Self {
            reader,
            time: Duration::ZERO,
            failed: false,
        })
    }

    /// Returns a reference to the underlying byte reader.
    #[must_use]
    pub fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Decodes the next record, or `None` at a clean end of the recording.
    pub fn next_event(&mut self) -> io::Result<Option<RecordedEvent>> {
        if self.failed {
            return Ok(None);
        }
        match self.read_record() {
            Ok(event) => Ok(event),
            Err(e) => {
                // Stick to the error: a desynchronised stream cannot be
                // resumed reliably.
                self.failed = true;
                Err(e)
            }
        }
    }

    fn read_record(&mut self) -> io::Result<Option<RecordedEvent>> {
        let Some(delta_micros) = read_u64_or_eof(&mut self.reader)? else {
            return Ok(None);
        };
        self.time += Duration::from_micros(delta_micros);

        let tracker = match read_u8(&mut self.reader)? {
            0 => None,
            1 => {
                let raw = read_u64(&mut self.reader)?;
                Some(
                    usize::try_from(raw)
                        .map_err(|_| invalid_data("tracker id does not fit into usize"))?,
                )
            }
            _ => return Err(invalid_data("invalid tracker tag in recording")),
        };

        let kind = read_kind(&mut self.reader)?;

        let path_count = read_u32(&mut self.reader)?;
        if path_count > MAX_PATHS_PER_EVENT {
            return Err(invalid_data("event has an implausible number of paths"));
        }
        let mut paths = Vec::with_capacity(path_count.min(64) as usize);
        for _ in 0..path_count {
            paths.push(read_path(&mut self.reader)?);
        }

        let mut event = Event::new(kind);
        event.paths = paths;
        if let Some(tracker) = tracker {
            event.attrs.set_tracker(tracker);
        }
        Ok(Some(RecordedEvent {
            at: self.time,
            event,
        }))
    }
}

impl<R: Read> Iterator for RecordReader<R> {
    type Item = io::Result<RecordedEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_event().transpose()
    }
}

/// Whether the replay should continue after an event was delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayControl {
    /// Deliver the next recorded event.
    Continue,
    /// Stop the replay immediately, without reading further records.
    Stop,
}

/// Summary of a finished (or cancelled) replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayStats {
    /// How many events were delivered to the handler.
    pub delivered: u64,
    /// Whether the handler stopped the replay before the recording ended.
    pub stopped_early: bool,
}

/// Replays a recording into a handler, honouring the recorded timing.
///
/// For every record the replayer asks its [`Clock`] to wait until the
/// record's relative timestamp, then hands the event to the handler. With a
/// [`ManualClock`] no real time passes, making replays deterministic and
/// instant.
pub struct Replayer<R, C> {
    reader: RecordReader<R>,
    clock: C,
}

impl<R: Read, C: Clock> Replayer<R, C> {
    /// Creates a replayer from a record reader and a clock.
    #[must_use]
    pub fn new(reader: RecordReader<R>, clock: C) -> Self {
        Self { reader, clock }
    }

    /// Returns the clock used by this replayer.
    #[must_use]
    pub fn clock(&self) -> &C {
        &self.clock
    }

    /// Returns the underlying record reader.
    #[must_use]
    pub fn reader(&self) -> &RecordReader<R> {
        &self.reader
    }

    /// Runs the replay to completion (or until the handler returns
    /// [`ReplayControl::Stop`]).
    ///
    /// The handler receives each [`RecordedEvent`] in recorded order. Because
    /// the underlying reader is lazy, stopping early does not read the rest
    /// of the recording. A truncated or corrupt record surfaces as an error
    /// after all complete records before it were delivered.
    pub fn run<H>(&mut self, mut handler: H) -> io::Result<ReplayStats>
    where
        H: FnMut(RecordedEvent) -> ReplayControl,
    {
        let mut delivered = 0u64;
        while let Some(recorded) = self.reader.next_event()? {
            self.clock.wait_until(recorded.at);
            delivered += 1;
            if handler(recorded) == ReplayControl::Stop {
                return Ok(ReplayStats {
                    delivered,
                    stopped_early: true,
                });
            }
        }
        Ok(ReplayStats {
            delivered,
            stopped_early: false,
        })
    }
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Reads a u64, mapping a clean end-of-stream (not a single byte read) to
/// `None`. A partial value is an `UnexpectedEof` error.
fn read_u64_or_eof(reader: &mut impl Read) -> io::Result<Option<u64>> {
    let mut buf = [0u8; 8];
    let mut read = 0;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => {
                if read == 0 {
                    return Ok(None);
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "recording ends in the middle of a record",
                ));
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(Some(u64::from_le_bytes(buf)))
}

fn read_path(reader: &mut impl Read) -> io::Result<PathBuf> {
    let len = read_u32(reader)?;
    if len > MAX_PATH_BYTES {
        return Err(invalid_data("path length in recording is implausible"));
    }
    let mut bytes = vec![0u8; len as usize];
    reader.read_exact(&mut bytes)?;
    // SAFETY: recordings produced by `Recorder` store the exact byte sequence
    // returned by `OsStr::as_encoded_bytes`, which is the only input this
    // format accepts. Bytes from any other source are outside the format's
    // contract.
    let os_str = unsafe { OsStr::from_encoded_bytes_unchecked(&bytes) };
    Ok(PathBuf::from(os_str))
}

fn write_access_mode(writer: &mut impl Write, mode: &AccessMode) -> io::Result<()> {
    match mode {
        AccessMode::Any => writer.write_all(&[0]),
        AccessMode::Execute => writer.write_all(&[1]),
        AccessMode::Read => writer.write_all(&[2]),
        AccessMode::Write => writer.write_all(&[3]),
        AccessMode::Other => writer.write_all(&[4]),
    }
}

fn write_kind(writer: &mut impl Write, kind: &EventKind) -> io::Result<()> {
    match kind {
        EventKind::Any => writer.write_all(&[0]),
        EventKind::Access(access) => {
            writer.write_all(&[1])?;
            match access {
                AccessKind::Any => writer.write_all(&[0]),
                AccessKind::Read => writer.write_all(&[1]),
                AccessKind::Open(mode) => {
                    writer.write_all(&[2])?;
                    write_access_mode(writer, mode)
                }
                AccessKind::Close(mode) => {
                    writer.write_all(&[3])?;
                    write_access_mode(writer, mode)
                }
                AccessKind::Other => writer.write_all(&[4]),
            }
        }
        EventKind::Create(create) => {
            writer.write_all(&[2])?;
            match create {
                CreateKind::Any => writer.write_all(&[0]),
                CreateKind::File => writer.write_all(&[1]),
                CreateKind::Folder => writer.write_all(&[2]),
                CreateKind::Other => writer.write_all(&[3]),
            }
        }
        EventKind::Modify(modify) => {
            writer.write_all(&[3])?;
            match modify {
                ModifyKind::Any => writer.write_all(&[0]),
                ModifyKind::Data(change) => {
                    writer.write_all(&[1])?;
                    match change {
                        DataChange::Any => writer.write_all(&[0]),
                        DataChange::Size => writer.write_all(&[1]),
                        DataChange::Content => writer.write_all(&[2]),
                        DataChange::Other => writer.write_all(&[3]),
                    }
                }
                ModifyKind::Metadata(meta) => {
                    writer.write_all(&[2])?;
                    match meta {
                        MetadataKind::Any => writer.write_all(&[0]),
                        MetadataKind::AccessTime => writer.write_all(&[1]),
                        MetadataKind::WriteTime => writer.write_all(&[2]),
                        MetadataKind::Permissions => writer.write_all(&[3]),
                        MetadataKind::Ownership => writer.write_all(&[4]),
                        MetadataKind::Extended => writer.write_all(&[5]),
                        MetadataKind::Other => writer.write_all(&[6]),
                    }
                }
                ModifyKind::Name(mode) => {
                    writer.write_all(&[3])?;
                    match mode {
                        RenameMode::Any => writer.write_all(&[0]),
                        RenameMode::To => writer.write_all(&[1]),
                        RenameMode::From => writer.write_all(&[2]),
                        RenameMode::Both => writer.write_all(&[3]),
                        RenameMode::Other => writer.write_all(&[4]),
                    }
                }
                ModifyKind::Other => writer.write_all(&[4]),
            }
        }
        EventKind::Remove(remove) => {
            writer.write_all(&[4])?;
            match remove {
                RemoveKind::Any => writer.write_all(&[0]),
                RemoveKind::File => writer.write_all(&[1]),
                RemoveKind::Folder => writer.write_all(&[2]),
                RemoveKind::Other => writer.write_all(&[3]),
            }
        }
        EventKind::Other => writer.write_all(&[5]),
    }
}

fn read_access_mode(reader: &mut impl Read) -> io::Result<AccessMode> {
    Ok(match read_u8(reader)? {
        0 => AccessMode::Any,
        1 => AccessMode::Execute,
        2 => AccessMode::Read,
        3 => AccessMode::Write,
        4 => AccessMode::Other,
        _ => return Err(invalid_data("invalid access mode tag in recording")),
    })
}

fn read_kind(reader: &mut impl Read) -> io::Result<EventKind> {
    const BAD: &str = "invalid event kind tag in recording";
    Ok(match read_u8(reader)? {
        0 => EventKind::Any,
        1 => EventKind::Access(match read_u8(reader)? {
            0 => AccessKind::Any,
            1 => AccessKind::Read,
            2 => AccessKind::Open(read_access_mode(reader)?),
            3 => AccessKind::Close(read_access_mode(reader)?),
            4 => AccessKind::Other,
            _ => return Err(invalid_data(BAD)),
        }),
        2 => EventKind::Create(match read_u8(reader)? {
            0 => CreateKind::Any,
            1 => CreateKind::File,
            2 => CreateKind::Folder,
            3 => CreateKind::Other,
            _ => return Err(invalid_data(BAD)),
        }),
        3 => EventKind::Modify(match read_u8(reader)? {
            0 => ModifyKind::Any,
            1 => ModifyKind::Data(match read_u8(reader)? {
                0 => DataChange::Any,
                1 => DataChange::Size,
                2 => DataChange::Content,
                3 => DataChange::Other,
                _ => return Err(invalid_data(BAD)),
            }),
            2 => ModifyKind::Metadata(match read_u8(reader)? {
                0 => MetadataKind::Any,
                1 => MetadataKind::AccessTime,
                2 => MetadataKind::WriteTime,
                3 => MetadataKind::Permissions,
                4 => MetadataKind::Ownership,
                5 => MetadataKind::Extended,
                6 => MetadataKind::Other,
                _ => return Err(invalid_data(BAD)),
            }),
            3 => ModifyKind::Name(match read_u8(reader)? {
                0 => RenameMode::Any,
                1 => RenameMode::To,
                2 => RenameMode::From,
                3 => RenameMode::Both,
                4 => RenameMode::Other,
                _ => return Err(invalid_data(BAD)),
            }),
            4 => ModifyKind::Other,
            _ => return Err(invalid_data(BAD)),
        }),
        4 => EventKind::Remove(match read_u8(reader)? {
            0 => RemoveKind::Any,
            1 => RemoveKind::File,
            2 => RemoveKind::Folder,
            3 => RemoveKind::Other,
            _ => return Err(invalid_data(BAD)),
        }),
        5 => EventKind::Other,
        _ => return Err(invalid_data(BAD)),
    })
}
