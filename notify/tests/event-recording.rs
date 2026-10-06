//! Acceptance tests for the event recording format and replay component.

use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Duration;

use notify::Event;
use notify::event::{CreateKind, EventKind, ModifyKind, RemoveKind, RenameMode};
use notify::recording::{
    ManualClock, RecordReader, RecordedEvent, Recorder, ReplayControl, Replayer,
};

/// Records the given events at the given (relative) timestamps into an
/// in-memory recording.
fn record_events(stamps: &[Duration], events: &[Event]) -> Vec<u8> {
    assert_eq!(stamps.len(), events.len());
    let mut recorder = Recorder::with_clock(Vec::new(), ManualClock::new()).unwrap();
    for (at, event) in stamps.iter().zip(events) {
        recorder.record_at(*at, event).unwrap();
    }
    assert_eq!(recorder.recorded_count(), events.len() as u64);
    recorder.into_inner()
}

/// Replays a recording with a fake clock, returning the delivered events.
fn replay_all(bytes: &[u8]) -> Vec<RecordedEvent> {
    let reader = RecordReader::new(bytes).unwrap();
    let mut replayer = Replayer::new(reader, ManualClock::new());
    let mut seen = Vec::new();
    let stats = replayer
        .run(|recorded| {
            seen.push(recorded);
            ReplayControl::Continue
        })
        .unwrap();
    assert!(!stats.stopped_early);
    seen
}

#[test]
fn create_rename_delete_roundtrip_in_order() {
    let events = vec![
        Event::new(EventKind::Create(CreateKind::File)).add_path(PathBuf::from("watched/a.txt")),
        Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
            .add_path(PathBuf::from("watched/a.txt"))
            .add_path(PathBuf::from("watched/b.txt"))
            .set_tracker(42),
        Event::new(EventKind::Remove(RemoveKind::File)).add_path(PathBuf::from("watched/b.txt")),
    ];
    let stamps = [
        Duration::from_millis(0),
        Duration::from_millis(120),
        Duration::from_millis(235),
    ];
    let bytes = record_events(&stamps, &events);

    // Replay with a fake clock: no watcher involved, no real time passes.
    let reader = RecordReader::new(bytes.as_slice()).unwrap();
    let mut replayer = Replayer::new(reader, ManualClock::new());
    let mut delivered = Vec::new();
    let stats = replayer
        .run(|recorded| {
            delivered.push(recorded);
            ReplayControl::Continue
        })
        .unwrap();

    assert_eq!(stats.delivered, 3);
    assert!(!stats.stopped_early);

    // Same content, same order.
    let replayed: Vec<Event> = delivered.iter().map(|r| r.event.clone()).collect();
    assert_eq!(replayed, events);

    // Same relative timing, honoured through the fake clock.
    let ats: Vec<Duration> = delivered.iter().map(|r| r.at).collect();
    assert_eq!(ats, stamps);
    assert_eq!(replayer.clock().waits(), stamps);
    assert_eq!(replayer.clock().time(), Duration::from_millis(235));

    // Kind, paths and tracker survived exactly.
    assert_eq!(delivered[0].event.kind, EventKind::Create(CreateKind::File));
    assert_eq!(delivered[0].event.paths, [PathBuf::from("watched/a.txt")]);
    assert_eq!(delivered[0].event.tracker(), None);
    assert_eq!(
        delivered[1].event.kind,
        EventKind::Modify(ModifyKind::Name(RenameMode::Both))
    );
    assert_eq!(
        delivered[1].event.paths,
        [
            PathBuf::from("watched/a.txt"),
            PathBuf::from("watched/b.txt")
        ]
    );
    assert_eq!(delivered[1].event.tracker(), Some(42));
    assert_eq!(delivered[2].event.kind, EventKind::Remove(RemoveKind::File));
}

#[test]
fn same_timestamp_events_keep_order_and_zero_delta() {
    let events = vec![
        Event::new(EventKind::Create(CreateKind::Folder)).add_path(PathBuf::from("d")),
        Event::new(EventKind::Create(CreateKind::File)).add_path(PathBuf::from("d/x")),
        Event::new(EventKind::Remove(RemoveKind::Any)).add_path(PathBuf::from("d/x")),
    ];
    let t = Duration::from_millis(50);
    let stamps = [t, t, t];
    let bytes = record_events(&stamps, &events);

    // The two trailing records must have a zero delta on the wire.
    // header (6) + record0 (8 delta + 1 tracker + 2 kind + 4 count + 4 len + 1 path)
    let record0_len = 8 + 1 + 2 + 4 + 4 + 1;
    let delta1 = u64::from_le_bytes(
        bytes[6 + record0_len..6 + record0_len + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        delta1, 0,
        "second event at the same timestamp must have zero delta"
    );

    let delivered = replay_all(&bytes);
    let replayed: Vec<Event> = delivered.iter().map(|r| r.event.clone()).collect();
    assert_eq!(replayed, events);
    assert!(delivered.iter().all(|r| r.at == t));
}

#[test]
fn multi_path_event_roundtrip() {
    let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
        .add_path(PathBuf::from("one"))
        .add_path(PathBuf::from("two"))
        .add_path(PathBuf::from("three"))
        .set_tracker(7);
    let bytes = record_events(&[Duration::ZERO], &[event.clone()]);

    let delivered = replay_all(&bytes);
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].event, event);
    assert_eq!(delivered[0].event.paths.len(), 3);
    assert_eq!(delivered[0].event.tracker(), Some(7));
}

/// A `Read` wrapper that counts how many bytes were actually consumed.
struct CountingReader<R> {
    inner: R,
    bytes_read: usize,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read += n;
        Ok(n)
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_paths_roundtrip_byte_for_byte() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    // Controlled byte fixtures: valid-on-unix paths that are not UTF-8.
    let raw_a: &[u8] = b"dir/bad-\xff\xfe-name";
    let raw_b: &[u8] = b"dir/\x80other";
    let event = Event::new(EventKind::Modify(ModifyKind::Name(RenameMode::Both)))
        .add_path(PathBuf::from(OsStr::from_bytes(raw_a)))
        .add_path(PathBuf::from(OsStr::from_bytes(raw_b)))
        .set_tracker(9);
    let bytes = record_events(&[Duration::from_millis(3)], &[event.clone()]);

    let delivered = replay_all(&bytes);
    assert_eq!(delivered.len(), 1);
    let paths = &delivered[0].event.paths;
    assert_eq!(paths.len(), 2);
    // Byte-for-byte identical, never replaced by U+FFFD.
    assert_eq!(paths[0].as_os_str().as_bytes(), raw_a);
    assert_eq!(paths[1].as_os_str().as_bytes(), raw_b);
    assert_eq!(delivered[0].event, event);
}

#[cfg(unix)]
#[test]
fn non_utf8_path_decoded_from_raw_fixture_bytes() {
    use std::os::unix::ffi::OsStrExt;

    // Hand-built recording fixture: header + one record whose single path is
    // the raw (non-UTF-8) byte sequence b"f\x80g". This exercises the decoder
    // directly, independent of the encoder.
    let mut fixture = Vec::new();
    fixture.extend_from_slice(b"NFREC\x01"); // header
    fixture.extend_from_slice(&0u64.to_le_bytes()); // delta: 0 microseconds
    fixture.push(0); // no tracker
    fixture.extend_from_slice(&[2, 1]); // kind: Create(File)
    fixture.extend_from_slice(&1u32.to_le_bytes()); // one path
    fixture.extend_from_slice(&3u32.to_le_bytes()); // path length 3
    fixture.extend_from_slice(b"f\x80g"); // raw, non-UTF-8 path bytes

    let mut reader = RecordReader::new(fixture.as_slice()).unwrap();
    let recorded = reader.next_event().unwrap().expect("one record");
    assert_eq!(recorded.event.paths.len(), 1);
    assert_eq!(recorded.event.paths[0].as_os_str().as_bytes(), b"f\x80g");
    // The path must not have been lossy-decoded into replacement characters.
    assert!(
        !recorded.event.paths[0]
            .as_os_str()
            .as_bytes()
            .contains(&0xEF)
    );
    assert!(reader.next_event().unwrap().is_none());
}

#[test]
fn cancellation_stops_early_without_reading_the_whole_recording() {
    let events: Vec<Event> = (0..5)
        .map(|i| {
            Event::new(EventKind::Create(CreateKind::File))
                .add_path(PathBuf::from(format!("file-{i}.txt")))
        })
        .collect();
    let stamps = [Duration::ZERO; 5];
    let bytes = record_events(&stamps, &events);
    let total = bytes.len();

    let counting = CountingReader {
        inner: bytes.as_slice(),
        bytes_read: 0,
    };
    let reader = RecordReader::new(counting).unwrap();
    let mut replayer = Replayer::new(reader, ManualClock::new());
    let mut delivered_events = Vec::new();
    let stats = replayer
        .run(|recorded| {
            delivered_events.push(recorded.event);
            ReplayControl::Stop // cancel after the very first event
        })
        .unwrap();

    assert!(stats.stopped_early);
    assert_eq!(stats.delivered, 1);
    assert_eq!(delivered_events, events[..1]);

    // The replay must not have consumed the rest of the recording.
    let consumed = replayer.reader().get_ref().bytes_read;
    assert!(
        consumed < total,
        "early stop read the whole recording: {consumed} of {total} bytes"
    );
}

#[test]
fn truncated_tail_errors_after_complete_records() {
    let events = vec![
        Event::new(EventKind::Create(CreateKind::File)).add_path(PathBuf::from("a")),
        Event::new(EventKind::Remove(RemoveKind::File)).add_path(PathBuf::from("a")),
        Event::new(EventKind::Create(CreateKind::File)).add_path(PathBuf::from("b")),
    ];
    let stamps = [
        Duration::ZERO,
        Duration::from_millis(1),
        Duration::from_millis(2),
    ];
    let bytes = record_events(&stamps, &events);

    // Cut the recording in the middle of the last record.
    let truncated = &bytes[..bytes.len() - 3];
    let reader = RecordReader::new(truncated).unwrap();
    let mut replayer = Replayer::new(reader, ManualClock::new());
    let mut delivered_events = Vec::new();
    let err = replayer
        .run(|recorded| {
            delivered_events.push(recorded.event);
            ReplayControl::Continue
        })
        .expect_err("truncated recording must error");

    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    // The two complete events were delivered before the error surfaced.
    assert_eq!(delivered_events, events[..2]);

    // A recording cut exactly at a record boundary ends cleanly instead.
    let mut reader = RecordReader::new(truncated).unwrap();
    assert!(reader.next_event().unwrap().is_some());
    assert!(reader.next_event().unwrap().is_some());
    // (the truncated tail still errors on a plain reader)
    assert!(reader.next_event().is_err());

    // Truncated header is rejected up front.
    let err = RecordReader::new(&bytes[..4]).map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);

    // Garbage is rejected up front.
    let err = RecordReader::new(&b"not-a-recording"[..])
        .map(|_| ())
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}
