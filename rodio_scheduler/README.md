# Rodio Scheduler

[![Crates.io](https://img.shields.io/crates/v/rodio_scheduler.svg)](https://crates.io/crates/rodio_scheduler)
[![Docs.rs](https://docs.rs/rodio_scheduler/badge.svg)](https://docs.rs/rodio_scheduler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A library for sample-perfect audio scheduling with rodio. 

## Important

This crate requires the nightly Rust compiler because it uses the `portable-simd` feature, which has not yet been stabilized.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
rodio_scheduler = "0.1.3"
```

Here is an example of how to use the library to schedule a sound and keep track of the sample counter. For more information, see the [Docs](https://docs.rs/rodio_scheduler).

```rust
use std::fs::File;

use rodio::{Source, Decoder, OutputStreamBuilder};
use rodio_scheduler::{Scheduler, PlaybackEvent};

fn main() {
   // Get an output stream handle to the default physical sound device.
   let stream = OutputStreamBuilder::open_default_stream().unwrap();

   // Load a sound from a file.
   let metronome = File::open("assets/metronome.wav").unwrap();
   let metronome_decoder_source = Decoder::new(metronome).unwrap();

   // Create a scheduler.
   let mut scheduler = Scheduler::new(metronome_decoder_source, 48000, 2);

   // Load another sound to be scheduled.
   let note_hit = File::open("assets/note_hit.wav").unwrap();
   let note_hit_decoder_source = Decoder::new(note_hit).unwrap();

   // Add the sound to the scheduler, with a list of playback events to schedule.
   let note_hit_id = scheduler.add_source(note_hit_decoder_source);

   // Schedule the sound to be played at a specific timestamp.
   let event = PlaybackEvent {
       source_id: note_hit_id,
       timestamp: scheduler.sample_rate() as u64 * 2, // 2 seconds in
       repeat: None,
   };
   scheduler.get_scheduler(note_hit_id).unwrap().schedule_event(event);

   // Get the sample counter before moving the scheduler to the audio thread
   let sample_counter = scheduler.get_sample_counter();

   // Play the scheduled sounds.
   let _ = stream.mixer().add(scheduler);

   // Get the current sample index while playing
   let _current_samples = sample_counter.get();
   //do_something(current_samples);

   // The sound plays in a separate audio thread,
   // so we need to keep the main thread alive while it's playing.
   std::thread::sleep(std::time::Duration::from_secs(5));
}
```

## Features

- `simd`: Enables SIMD optimizations for audio processing.
- `profiler`: Enables profiling with `time-graph`. Beware that this has a big impact on real-time performance.

## License
This project is licensed under the MIT license.
