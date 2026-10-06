//! Recording and replaying of file system events, for debugging and testing.
//!
//! This module provides a small, self-contained binary recording format plus a
//! [`Recorder`] and a [`Replayer`]. While investigating a problem, a `Recorder`
//! can capture a short stream of [`Event`]s — the raw event kind, the list of
//! paths, the tracker attribute, and the time of each event relative to the
//! previous one — without requiring a real watcher afterwards: the [`Replayer`]
//! reads the recording back and feeds the events to the same handling code,
//! sleeping between events according to a pluggable [`Clock`].
//!
//! Paths are stored as their platform encoded bytes ([`OsStr::as_encoded_bytes`]),
//! which is a reversible encoding: non-UTF-8 paths are preserved byte-for-byte
//! and are never mangled with replacement characters.
//!
//! # Format
//!
//! All integers are little-endian.
//!
//! ```text
//! header:  "NEVR" u16(version=1) u16(reserved=0)
//! record:  u32(payload_len) payload
//! payload: u64(delta_micros) [u8; 3](kind tags)
//!          u8(tracker_present) [u64(tracker)]?
//!          u32(path_count) (u32(len) [u8; len](encoded path bytes))*
//! ```
//!
//! A record whose length prefix or payload is cut short (e.g. because the
//! process died while writing) is treated as a truncated tail: replay stops
//! cleanly at the last complete record instead of failing.
//!
//! # Example
//!
//! ```
//! use notify::recording::{ManualClock, Recorder, Replayer};
//! use notify::{event::CreateKind, Event, EventKind};
//! use std::time::Duration;
//!
//! // Record.
//! let clock = ManualClock::new();
//! let mut recorder = Recorder::create(Vec::new(), &clock).unwrap();
//! recorder.record(&Event::new(EventKind::Create(CreateKind::File)).add_path("a.txt".into())).unwrap();
//! clock.advance(Duration::from_millis(5));
//! recorder.record(&Event::new(EventKind::Any)).unwrap();
//! let bytes = recorder.into_inner();
//!
//! // Replay against a fake clock: no real time passes.
//! let clock = ManualClock::new();
//! let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
//! let mut events = Vec::new();
//! replayer.replay_all(|event| { events.push(event); true }).unwrap();
//! assert_eq!(events.len(), 2);
//! ```

use std::cell::{Cell, RefCell};
use std::ffi::OsStr;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use notify_types::event::{
    AccessKind, AccessMode, CreateKind, DataChange, MetadataKind, ModifyKind, RemoveKind,
    RenameMode,
};

use crate::{Event, EventKind};

/// Magic bytes at the start of every recording.
const MAGIC: [u8; 4] = *b"NEVR";

/// Version of the recording format written by [`Recorder`] and read by [`Replayer`].
pub const FORMAT_VERSION: u16 = 1;

const HEADER_LEN: usize = 8;

/// Upper bound for a single record payload, to reject corrupt length prefixes
/// before allocating.
const MAX_RECORD_LEN: u32 = 64 * 1024 * 1024;

/// Tag used for "no value at this position" in the kind tag triple.
const NONE: u8 = 0xFF;

/// A source of monotonic time used while recording and replaying.
///
/// The recorder stamps each event with `now()` relative to the previous event;
/// the replayer calls [`Clock::sleep_until`] with the absolute replay time of
/// the next event before handing it out.
pub trait Clock {
    /// The current monotonic time.
    fn now(&self) -> Duration;

    /// Block (or, for fake clocks, pretend to block) until `at`.
    fn sleep_until(&self, at: Duration);
}

/// A [`Clock`] backed by [`Instant`], for recording live sessions and replaying
/// them in real time.
#[derive(Debug, Clone, Copy)]
pub struct SystemClock {
    start: Instant,
}

impl SystemClock {
    /// Creates a clock whose zero point is now.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.start.elapsed()
    }

    fn sleep_until(&self, at: Duration) {
        let now = self.now();
        if at > now {
            std::thread::sleep(at - now);
        }
    }
}

/// A manually driven [`Clock`] for tests and offline replay: time only passes
/// when it is advanced, and `sleep_until` jumps straight to the target while
/// recording the requested instants in a log.
#[derive(Debug, Default)]
pub struct ManualClock {
    now: Cell<Duration>,
    sleeps: RefCell<Vec<Duration>>,
}

impl ManualClock {
    /// Creates a clock stopped at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances the clock by `delta`.
    pub fn advance(&self, delta: Duration) {
        self.set(self.now.get() + delta);
    }

    /// Sets the clock to an absolute time.
    pub fn set(&self, at: Duration) {
        self.now.set(at);
    }

    /// The instants at which the clock was asked to sleep until, in order.
    pub fn sleep_log(&self) -> Vec<Duration> {
        self.sleeps.borrow().clone()
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        self.now.get()
    }

    fn sleep_until(&self, at: Duration) {
        self.sleeps.borrow_mut().push(at);
        if at > self.now.get() {
            self.now.set(at);
        }
    }
}

impl Clock for &SystemClock {
    fn now(&self) -> Duration {
        (*self).now()
    }

    fn sleep_until(&self, at: Duration) {
        (*self).sleep_until(at);
    }
}

impl Clock for &ManualClock {
    fn now(&self) -> Duration {
        (*self).now()
    }

    fn sleep_until(&self, at: Duration) {
        (*self).sleep_until(at);
    }
}

/// Writes events to a recording.
///
/// The header is written on [`Recorder::create`]; each call to
/// [`Recorder::record`] appends one self-delimiting record. If the writer (or
/// the process) dies mid-record, the trailing partial record is ignored by the
/// replayer.
#[derive(Debug)]
pub struct Recorder<W, C> {
    writer: W,
    clock: C,
    last: Duration,
    count: u64,
}

impl<W: Write, C: Clock> Recorder<W, C> {
    /// Creates a recorder, writing the format header to `writer`.
    pub fn create(mut writer: W, clock: C) -> io::Result<Self> {
        let mut header = [0u8; HEADER_LEN];
        header[..4].copy_from_slice(&MAGIC);
        header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        // header[6..8] reserved, zero.
        writer.write_all(&header)?;
        let last = clock.now();
        Ok(Self {
            writer,
            clock,
            last,
            count: 0,
        })
    }

    /// Appends one event to the recording.
    ///
    /// The recorded timestamp is the clock reading now, relative to the
    /// previous recorded event (or to the creation of the recorder).
    pub fn record(&mut self, event: &Event) -> io::Result<()> {
        let now = self.clock.now();
        let delta = now.saturating_sub(self.last);
        self.last = now;

        let mut payload = Vec::new();
        payload.extend_from_slice(
            &u64::try_from(delta.as_micros())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        payload.extend_from_slice(&encode_kind(&event.kind));
        match event.tracker() {
            Some(tracker) => {
                payload.push(1);
                payload.extend_from_slice(&(tracker as u64).to_le_bytes());
            }
            None => payload.push(0),
        }
        payload.extend_from_slice(&(event.paths.len() as u32).to_le_bytes());
        for path in &event.paths {
            let bytes = path.as_os_str().as_encoded_bytes();
            payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(bytes);
        }

        self.writer
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.count += 1;
        Ok(())
    }

    /// Number of events recorded so far.
    pub fn events_recorded(&self) -> u64 {
        self.count
    }

    /// Flushes the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Returns a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.writer
    }

    /// Consumes the recorder and returns the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

/// Reads a recording back and hands the events to handling code, in order,
/// paced by a [`Clock`].
///
/// The replayer is lazy: it reads exactly one record from the source per
/// [`Replayer::next_event`] call and never buffers ahead, so cancelling a
/// replay stops reading immediately and leaves the rest of the recording
/// untouched.
#[derive(Debug)]
pub struct Replayer<R, C> {
    reader: R,
    clock: C,
    start: Duration,
    elapsed: Duration,
    count: u64,
    done: bool,
    cancelled: bool,
    truncated: bool,
}

impl<R: Read, C: Clock> Replayer<R, C> {
    /// Opens a recording, reading and validating the header.
    pub fn open(mut reader: R, clock: C) -> io::Result<Self> {
        let mut header = [0u8; HEADER_LEN];
        reader.read_exact(&mut header).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                io::Error::new(io::ErrorKind::InvalidData, "recording too short for header")
            } else {
                e
            }
        })?;
        if header[..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a notify event recording (bad magic)",
            ));
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported recording format version {version}"),
            ));
        }
        let start = clock.now();
        Ok(Self {
            reader,
            clock,
            start,
            elapsed: Duration::ZERO,
            count: 0,
            done: false,
            cancelled: false,
            truncated: false,
        })
    }

    /// Reads, waits for, and returns the next event, or `None` at the end of
    /// the recording (including a truncated tail) or after cancellation.
    pub fn next_event(&mut self) -> io::Result<Option<Event>> {
        if self.done || self.cancelled {
            return Ok(None);
        }

        let mut len_buf = [0u8; 4];
        match read_up_to(&mut self.reader, &mut len_buf)? {
            0 => {
                self.done = true;
                return Ok(None);
            }
            4 => {}
            _ => {
                // Partial length prefix: truncated tail.
                self.done = true;
                self.truncated = true;
                return Ok(None);
            }
        }
        let len = u32::from_le_bytes(len_buf);
        if len > MAX_RECORD_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("record length {len} exceeds limit"),
            ));
        }
        let mut payload = vec![0u8; len as usize];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                // Partial payload: truncated tail.
                self.done = true;
                self.truncated = true;
                return Ok(None);
            }
            return Err(e);
        }

        let (delta, event) = decode_record(&payload)?;
        self.elapsed += delta;
        self.clock.sleep_until(self.start + self.elapsed);
        self.count += 1;
        Ok(Some(event))
    }

    /// Replays every remaining event to `handler`.
    ///
    /// The handler returns `true` to continue and `false` to stop early, which
    /// cancels the replay. Returns the number of events delivered.
    pub fn replay_all(&mut self, mut handler: impl FnMut(Event) -> bool) -> io::Result<u64> {
        let mut delivered = 0;
        while let Some(event) = self.next_event()? {
            delivered += 1;
            if !handler(event) {
                self.cancel();
                break;
            }
        }
        Ok(delivered)
    }

    /// Stops the replay. Subsequent calls to [`Replayer::next_event`] return
    /// `None` without reading any more of the recording.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Whether the replay was cancelled before the recording ended.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Whether the recording ended with a truncated (partially written) record.
    pub fn was_truncated(&self) -> bool {
        self.truncated
    }

    /// Number of events replayed so far.
    pub fn events_replayed(&self) -> u64 {
        self.count
    }
}

/// Reads until `buf` is full or EOF, returning the number of bytes read.
fn read_up_to(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Decodes one record payload into the relative time and the event.
fn decode_record(payload: &[u8]) -> io::Result<(Duration, Event)> {
    let mut cur = Cursor::new(payload);
    let delta_micros = cur.u64()?;
    let tags = cur.bytes(3)?;
    let kind = decode_kind([tags[0], tags[1], tags[2]])?;
    let tracker = match cur.u8()? {
        0 => None,
        1 => Some(cur.u64()? as usize),
        _ => return Err(invalid_data("invalid tracker presence flag")),
    };
    let path_count = cur.u32()?;
    let mut paths = Vec::with_capacity(path_count.min(1024) as usize);
    for _ in 0..path_count {
        let len = cur.u32()?;
        paths.push(decode_path(cur.bytes(len as usize)?)?);
    }
    if !cur.is_empty() {
        return Err(invalid_data("trailing bytes in record"));
    }

    let mut event = Event::new(kind);
    event.paths = paths;
    if let Some(tracker) = tracker {
        event = event.set_tracker(tracker);
    }
    Ok((Duration::from_micros(delta_micros), event))
}

/// A bounds-checked reader over a record payload.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos == self.buf.len()
    }

    fn bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|end| *end <= self.buf.len())
            .ok_or(invalid_data("record payload too short"))?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
}

/// Reconstructs a path from its recorded encoded bytes.
fn decode_path(bytes: &[u8]) -> io::Result<PathBuf> {
    if !valid_encoded_bytes(bytes) {
        return Err(invalid_data("path bytes are not valid for this platform"));
    }
    // SAFETY: the bytes were checked above to be well-formed for the platform
    // `OsStr` encoding, as produced by `OsStr::as_encoded_bytes` when the
    // recording was written.
    Ok(PathBuf::from(unsafe {
        OsStr::from_encoded_bytes_unchecked(bytes)
    }))
}

/// On Unix, any byte sequence is a valid `OsStr`.
#[cfg(unix)]
fn valid_encoded_bytes(_bytes: &[u8]) -> bool {
    true
}

/// On other platforms (notably Windows), the encoded form is WTF-8: UTF-8
/// extended to allow unpaired surrogates. Validate so that corrupt recordings
/// are rejected instead of producing ill-formed `OsStr`s.
#[cfg(not(unix))]
fn valid_encoded_bytes(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let (len, min2, max2) = match b {
            0x00..=0x7F => {
                i += 1;
                continue;
            }
            0xC2..=0xDF => (2, 0x80, 0xBF),
            0xE0 => (3, 0xA0, 0xBF),
            // WTF-8: unlike UTF-8, 0xED may be followed by 0xA0..=0xBF
            // (surrogate halves).
            0xE1..=0xEF => (3, 0x80, 0xBF),
            0xF0 => (4, 0x90, 0xBF),
            0xF1..=0xF3 => (4, 0x80, 0xBF),
            0xF4 => (4, 0x80, 0x8F),
            _ => return false,
        };
        if i + len > bytes.len() {
            return false;
        }
        let b2 = bytes[i + 1];
        if !(min2..=max2).contains(&b2) {
            return false;
        }
        if !bytes[i + 2..i + len]
            .iter()
            .all(|c| (0x80..=0xBF).contains(c))
        {
            return false;
        }
        i += len;
    }
    true
}

fn encode_kind(kind: &EventKind) -> [u8; 3] {
    match kind {
        EventKind::Any => [0, NONE, NONE],
        EventKind::Other => [5, NONE, NONE],
        EventKind::Access(access) => match access {
            AccessKind::Any => [1, 0, NONE],
            AccessKind::Read => [1, 1, NONE],
            AccessKind::Open(mode) => [1, 2, encode_access_mode(mode)],
            AccessKind::Close(mode) => [1, 3, encode_access_mode(mode)],
            AccessKind::Other => [1, 4, NONE],
        },
        EventKind::Create(create) => match create {
            CreateKind::Any => [2, 0, NONE],
            CreateKind::File => [2, 1, NONE],
            CreateKind::Folder => [2, 2, NONE],
            CreateKind::Other => [2, 3, NONE],
        },
        EventKind::Modify(modify) => match modify {
            ModifyKind::Any => [3, 0, NONE],
            ModifyKind::Data(change) => [3, 1, encode_data_change(change)],
            ModifyKind::Metadata(meta) => [3, 2, encode_metadata_kind(meta)],
            ModifyKind::Name(mode) => [3, 3, encode_rename_mode(mode)],
            ModifyKind::Other => [3, 4, NONE],
        },
        EventKind::Remove(remove) => match remove {
            RemoveKind::Any => [4, 0, NONE],
            RemoveKind::File => [4, 1, NONE],
            RemoveKind::Folder => [4, 2, NONE],
            RemoveKind::Other => [4, 3, NONE],
        },
    }
}

fn encode_access_mode(mode: &AccessMode) -> u8 {
    match mode {
        AccessMode::Any => 0,
        AccessMode::Execute => 1,
        AccessMode::Read => 2,
        AccessMode::Write => 3,
        AccessMode::Other => 4,
    }
}

fn encode_data_change(change: &DataChange) -> u8 {
    match change {
        DataChange::Any => 0,
        DataChange::Size => 1,
        DataChange::Content => 2,
        DataChange::Other => 3,
    }
}

fn encode_metadata_kind(kind: &MetadataKind) -> u8 {
    match kind {
        MetadataKind::Any => 0,
        MetadataKind::AccessTime => 1,
        MetadataKind::WriteTime => 2,
        MetadataKind::Permissions => 3,
        MetadataKind::Ownership => 4,
        MetadataKind::Extended => 5,
        MetadataKind::Other => 6,
    }
}

fn encode_rename_mode(mode: &RenameMode) -> u8 {
    match mode {
        RenameMode::Any => 0,
        RenameMode::To => 1,
        RenameMode::From => 2,
        RenameMode::Both => 3,
        RenameMode::Other => 4,
    }
}

fn decode_kind(tags: [u8; 3]) -> io::Result<EventKind> {
    let kind = match tags {
        [0, NONE, NONE] => EventKind::Any,
        [5, NONE, NONE] => EventKind::Other,
        [1, sub, third] => EventKind::Access(match (sub, third) {
            (0, NONE) => AccessKind::Any,
            (1, NONE) => AccessKind::Read,
            (2, t) => AccessKind::Open(decode_access_mode(t)?),
            (3, t) => AccessKind::Close(decode_access_mode(t)?),
            (4, NONE) => AccessKind::Other,
            _ => return Err(invalid_data("unknown access kind tags")),
        }),
        [2, sub, NONE] => EventKind::Create(match sub {
            0 => CreateKind::Any,
            1 => CreateKind::File,
            2 => CreateKind::Folder,
            3 => CreateKind::Other,
            _ => return Err(invalid_data("unknown create kind tag")),
        }),
        [3, sub, third] => EventKind::Modify(match (sub, third) {
            (0, NONE) => ModifyKind::Any,
            (1, t) => ModifyKind::Data(decode_data_change(t)?),
            (2, t) => ModifyKind::Metadata(decode_metadata_kind(t)?),
            (3, t) => ModifyKind::Name(decode_rename_mode(t)?),
            (4, NONE) => ModifyKind::Other,
            _ => return Err(invalid_data("unknown modify kind tags")),
        }),
        [4, sub, NONE] => EventKind::Remove(match sub {
            0 => RemoveKind::Any,
            1 => RemoveKind::File,
            2 => RemoveKind::Folder,
            3 => RemoveKind::Other,
            _ => return Err(invalid_data("unknown remove kind tag")),
        }),
        _ => return Err(invalid_data("unknown event kind tags")),
    };
    Ok(kind)
}

fn decode_access_mode(tag: u8) -> io::Result<AccessMode> {
    Ok(match tag {
        0 => AccessMode::Any,
        1 => AccessMode::Execute,
        2 => AccessMode::Read,
        3 => AccessMode::Write,
        4 => AccessMode::Other,
        _ => return Err(invalid_data("unknown access mode tag")),
    })
}

fn decode_data_change(tag: u8) -> io::Result<DataChange> {
    Ok(match tag {
        0 => DataChange::Any,
        1 => DataChange::Size,
        2 => DataChange::Content,
        3 => DataChange::Other,
        _ => return Err(invalid_data("unknown data change tag")),
    })
}

fn decode_metadata_kind(tag: u8) -> io::Result<MetadataKind> {
    Ok(match tag {
        0 => MetadataKind::Any,
        1 => MetadataKind::AccessTime,
        2 => MetadataKind::WriteTime,
        3 => MetadataKind::Permissions,
        4 => MetadataKind::Ownership,
        5 => MetadataKind::Extended,
        6 => MetadataKind::Other,
        _ => return Err(invalid_data("unknown metadata kind tag")),
    })
}

fn decode_rename_mode(tag: u8) -> io::Result<RenameMode> {
    Ok(match tag {
        0 => RenameMode::Any,
        1 => RenameMode::To,
        2 => RenameMode::From,
        3 => RenameMode::Both,
        4 => RenameMode::Other,
        _ => return Err(invalid_data("unknown rename mode tag")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A reader that hands out at most `chunk` bytes per `read` call and
    /// tracks how many bytes were consumed, to prove laziness.
    struct CountingReader<'a> {
        data: &'a [u8],
        chunk: usize,
        pos: usize,
        consumed: &'a Cell<usize>,
    }

    impl Read for CountingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.chunk).min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            self.consumed.set(self.consumed.get() + n);
            Ok(n)
        }
    }

    fn create_event() -> Event {
        Event::new(EventKind::Create(CreateKind::File)).add_path("watched/a.txt".into())
    }

    fn rename_event() -> Event {
        Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path("watched/a.txt".into())
            .add_path("watched/b.txt".into())
            .set_tracker(42)
    }

    fn remove_event() -> Event {
        Event::new(EventKind::Remove(RemoveKind::File)).add_path("watched/b.txt".into())
    }

    fn record_events(events: &[Event], gaps: &[Duration]) -> Vec<u8> {
        let clock = ManualClock::new();
        let mut recorder = Recorder::create(Vec::new(), &clock).unwrap();
        for (i, event) in events.iter().enumerate() {
            if let Some(gap) = i.checked_sub(1).and_then(|prev| gaps.get(prev)) {
                clock.advance(*gap);
            }
            recorder.record(event).unwrap();
        }
        assert_eq!(recorder.events_recorded(), events.len() as u64);
        recorder.into_inner()
    }

    #[test]
    fn record_and_replay_create_rename_remove_in_order() {
        let events = vec![create_event(), rename_event(), remove_event()];
        let gaps = vec![Duration::from_millis(5), Duration::from_millis(12)];
        let bytes = record_events(&events, &gaps);

        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
        let mut replayed = Vec::new();
        let delivered = replayer
            .replay_all(|event| {
                replayed.push(event);
                true
            })
            .unwrap();

        assert_eq!(delivered, 3);
        assert_eq!(replayed, events, "same events, same order");
        assert_eq!(replayed[1].tracker(), Some(42));
        // The fake clock was asked to sleep until the exact relative times.
        assert_eq!(
            clock.sleep_log(),
            vec![
                Duration::ZERO,
                Duration::from_millis(5),
                Duration::from_millis(17),
            ]
        );
        assert!(!replayer.was_truncated());
        assert!(!replayer.is_cancelled());
        assert_eq!(replayer.events_replayed(), 3);
    }

    #[test]
    fn same_timestamp_events_keep_order() {
        // No clock advance between records: all deltas are zero.
        let events = vec![create_event(), rename_event(), remove_event()];
        let bytes = record_events(&events, &[]);

        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
        let mut replayed = Vec::new();
        replayer
            .replay_all(|event| {
                replayed.push(event);
                true
            })
            .unwrap();

        assert_eq!(replayed, events);
        assert_eq!(
            clock.sleep_log(),
            vec![Duration::ZERO, Duration::ZERO, Duration::ZERO]
        );
    }

    #[test]
    fn multi_path_event_roundtrip() {
        let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path("one".into())
            .add_path("two".into())
            .add_path("three".into())
            .add_path("four".into());
        let bytes = record_events(&[event.clone()], &[]);

        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
        let replayed = replayer.next_event().unwrap().unwrap();
        assert_eq!(replayed, event);
        assert_eq!(replayed.paths.len(), 4);
        assert!(replayer.next_event().unwrap().is_none());
    }

    /// A path that is not valid UTF-8, built from controlled bytes.
    #[cfg(unix)]
    fn non_utf8_path() -> PathBuf {
        use std::os::unix::ffi::OsStrExt;
        PathBuf::from(OsStr::from_bytes(b"/tmp/ev-\xff\xfe-\x80name"))
    }

    /// A path that is not valid UTF-8, built from controlled bytes (an
    /// unpaired surrogate, encoded as WTF-8 internally).
    #[cfg(windows)]
    fn non_utf8_path() -> PathBuf {
        use std::os::windows::ffi::OsStringExt;
        PathBuf::from(std::ffi::OsString::from_wide(&[
            0x0061, 0x002F, 0xD800, 0x002F, 0x0062,
        ]))
    }

    #[test]
    fn non_utf8_path_roundtrip_byte_exact() {
        let path = non_utf8_path();
        assert!(
            path.as_os_str().to_str().is_none(),
            "fixture must not be valid UTF-8"
        );
        let raw = path.as_os_str().as_encoded_bytes();

        let event = Event::new(EventKind::Create(CreateKind::File)).add_path(path.clone());
        let bytes = record_events(&[event.clone()], &[]);

        // The raw path bytes appear verbatim in the recording: no replacement
        // characters, no lossy conversion.
        assert!(
            bytes.windows(raw.len()).any(|window| window == raw),
            "recording must contain the exact encoded path bytes"
        );
        assert!(!bytes.windows(3).any(|w| w == b"\xEF\xBF\xBD"));

        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
        let replayed = replayer.next_event().unwrap().unwrap();
        assert_eq!(replayed, event);
        assert_eq!(
            replayed.paths[0].as_os_str().as_encoded_bytes(),
            raw,
            "non-UTF-8 path must round-trip byte-for-byte"
        );
    }

    #[test]
    fn cancel_stops_early_without_reading_the_whole_recording() {
        let events: Vec<Event> = (0..200usize)
            .map(|i| {
                Event::new(EventKind::Create(CreateKind::File))
                    .add_path(format!("dir/file-{i:03}.txt").into())
                    .set_tracker(i)
            })
            .collect();
        let bytes = record_events(&events, &[]);
        let total = bytes.len();

        let consumed = Cell::new(0usize);
        let clock = ManualClock::new();
        let mut replayer = Replayer::open(
            CountingReader {
                data: &bytes,
                chunk: 7, // deliberately tiny reads
                pos: 0,
                consumed: &consumed,
            },
            &clock,
        )
        .unwrap();

        let mut replayed = Vec::new();
        let delivered = replayer
            .replay_all(|event| {
                replayed.push(event);
                replayed.len() < 2 // stop after the second event
            })
            .unwrap();

        assert_eq!(delivered, 2);
        assert_eq!(replayed, events[..2]);
        assert!(replayer.is_cancelled());
        assert!(replayer.next_event().unwrap().is_none());
        // Early stop must not read the whole recording: only the header plus
        // the first two records were consumed.
        assert!(
            consumed.get() < total / 4,
            "consumed {} of {} bytes",
            consumed.get(),
            total
        );
    }

    #[test]
    fn truncated_tail_stops_cleanly() {
        let events = vec![create_event(), rename_event(), remove_event()];
        let bytes = record_events(&events, &[]);

        // Cut the recording in the middle of the last record's payload.
        let cut = bytes.len() - 3;
        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..cut], &clock).unwrap();
        let mut replayed = Vec::new();
        let delivered = replayer
            .replay_all(|event| {
                replayed.push(event);
                true
            })
            .unwrap();

        assert_eq!(delivered, 2);
        assert_eq!(replayed, events[..2]);
        assert!(replayer.was_truncated());
        assert!(replayer.next_event().unwrap().is_none());
    }

    #[test]
    fn truncated_length_prefix_stops_cleanly() {
        let events = vec![create_event(), rename_event()];
        let bytes = record_events(&events, &[]);

        // Keep the first record plus two bytes of the next length prefix.
        let first_len =
            u32::from_le_bytes(bytes[HEADER_LEN..HEADER_LEN + 4].try_into().unwrap()) as usize;
        let cut = HEADER_LEN + 4 + first_len + 2;
        assert!(cut < bytes.len());

        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..cut], &clock).unwrap();
        let first = replayer.next_event().unwrap().unwrap();
        assert_eq!(first, events[0]);
        assert!(replayer.next_event().unwrap().is_none());
        assert!(replayer.was_truncated());
    }

    #[test]
    fn header_only_recording_yields_no_events() {
        let bytes = record_events(&[], &[]);
        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
        assert!(replayer.next_event().unwrap().is_none());
        assert!(!replayer.was_truncated());
    }

    #[test]
    fn bad_magic_and_short_header_are_rejected() {
        let clock = ManualClock::new();
        assert!(Replayer::open(&b"NOPE0000"[..], &clock).is_err());
        assert!(Replayer::open(&b"NEV"[..], &clock).is_err());
        // Wrong version.
        assert!(Replayer::open(&b"NEVR\x02\x00\x00\x00"[..], &clock).is_err());
    }

    #[test]
    fn system_clock_smoke() {
        let mut recorder = Recorder::create(Vec::new(), SystemClock::new()).unwrap();
        recorder.record(&create_event()).unwrap();
        recorder.record(&remove_event()).unwrap();
        let bytes = recorder.into_inner();

        let mut replayer = Replayer::open(&bytes[..], SystemClock::new()).unwrap();
        let mut replayed = Vec::new();
        replayer
            .replay_all(|event| {
                replayed.push(event);
                true
            })
            .unwrap();
        assert_eq!(replayed, vec![create_event(), remove_event()]);
    }

    #[test]
    fn all_event_kinds_roundtrip() {
        let kinds = vec![
            EventKind::Any,
            EventKind::Other,
            EventKind::Access(AccessKind::Any),
            EventKind::Access(AccessKind::Read),
            EventKind::Access(AccessKind::Open(AccessMode::Execute)),
            EventKind::Access(AccessKind::Close(AccessMode::Write)),
            EventKind::Access(AccessKind::Other),
            EventKind::Create(CreateKind::Any),
            EventKind::Create(CreateKind::Folder),
            EventKind::Create(CreateKind::Other),
            EventKind::Modify(ModifyKind::Any),
            EventKind::Modify(ModifyKind::Data(DataChange::Size)),
            EventKind::Modify(ModifyKind::Metadata(MetadataKind::Extended)),
            EventKind::Modify(ModifyKind::Name(RenameMode::From)),
            EventKind::Modify(ModifyKind::Other),
            EventKind::Remove(RemoveKind::Any),
            EventKind::Remove(RemoveKind::Folder),
            EventKind::Remove(RemoveKind::Other),
        ];
        let events: Vec<Event> = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Event::new(kind).add_path(format!("p{i}").into()))
            .collect();
        let bytes = record_events(&events, &[]);

        let clock = ManualClock::new();
        let mut replayer = Replayer::open(&bytes[..], &clock).unwrap();
        let mut replayed = Vec::new();
        replayer
            .replay_all(|event| {
                replayed.push(event);
                true
            })
            .unwrap();
        assert_eq!(replayed, events);
    }
}
