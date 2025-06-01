use std::fs::File;
use std::io::BufReader;
use rodio::{Decoder, OutputStream, source::Source};

use rodio_scheduler::Scheduler;

fn main() {
    // Get an output stream handle to the default physical sound device.
    // Note that no sound will be played if _stream is dropped
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();

    // Load a sound from a file, using a path relative to Cargo.toml
    let metronome = BufReader::new(File::open("assets/metronome.wav").unwrap());

    // Decode that sound file into a source
    let metronome_decoder_source = Decoder::new(metronome).unwrap();

    // Load a sound from a file, using a path relative to Cargo.toml
    let note_hit = BufReader::new(File::open("assets/note_hit.wav").unwrap());

    // Decode that sound file into a source
    let note_hit_decoder_source = Decoder::new(note_hit).unwrap().buffered();

    let mut scheduler = Scheduler::new(metronome_decoder_source, 48000, 2);
    scheduler.add_source(note_hit_decoder_source);
    
    // Play the sound directly on the device
    stream_handle.play_raw(scheduler);

    // The sound plays in a separate audio thread,
    // so we need to keep the main thread alive while it's playing.
    std::thread::sleep(std::time::Duration::from_secs(50));
}

