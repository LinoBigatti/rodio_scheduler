use std::fs::File;
use std::io::BufReader;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64};

use rodio::{Decoder, OutputStream};

use rodio_scheduler::{Scheduler, PlaybackEvent};

#[cfg(feature = "profiler")]
use time_graph;

fn main() {
    #[cfg(feature = "profiler")]
    time_graph::enable_data_collection(true);

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
    let note_hit_decoder_source = Decoder::new(note_hit).unwrap();

    println!("Scheduling...");
    
    //let mut scheduler = Scheduler::new(metronome_decoder_source, 48000, 2);
    let sample_counter = Arc::new(AtomicU64::new(0));
    let mut scheduler = Scheduler::with_capacity(metronome_decoder_source, sample_counter.clone(), 48000, 2, 10);
    let note_hit_id = scheduler.schedule_source(note_hit_decoder_source);

    for i in 0..8000 {
        let event = PlaybackEvent { 
            source_id: note_hit_id,
            timestamp: i as u64 * 48000 / 2,
            repeat: None,
        };

        scheduler.get_scheduler(note_hit_id).unwrap().schedule_event(event);
    }

    println!("Scheduled");
    
    // Play the sound directly on the device
    let _ = stream_handle.play_raw(scheduler);

    // The sound plays in a separate audio thread,
    // so we need to keep the main thread alive while it's playing.
    std::thread::sleep(std::time::Duration::from_secs(5));
    //let mut last = 0;
    //while true {
        //let val = sample_counter.load(Ordering::SeqCst);

        //if val != last {
            //last = val;

            //println!("{}", val);
        //}
    //}


    #[cfg(feature = "profiler")]
    println!("{}", time_graph::get_full_graph().as_dot());
}

