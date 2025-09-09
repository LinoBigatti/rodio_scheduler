/*!
A library for scheduling audio playback with `rodio`.

`rodio_scheduler` provides a [rodio](https://crates.io/crates/rodio) `Source` that can schedule other sources to be played
at specific timestamps. This is useful for applications that need to accurately schedule a
source to be played along other sources, such as rhythm games, digital audio workstations
(DAWs), or music players. For synchronizing visuals or other external events with the audio
playback, see the [rodio_playback_position](https://crates.io/crates/rodio_playback_position) crate.

## Important

This crate requires the nightly Rust compiler because it uses the `portable-simd` feature,
which has not yet been stabilized.

## Features

- **Sample-perfect Scheduling**: Schedule audio playback with sample-level accuracy.
- **SIMD Acceleration**: Uses SIMD for mixing audio samples, providing a small
  performance boost. This can be enabled with the `simd` feature flag.
- **Optional Profiling**: Includes an optional `profiler` feature to instrument the code
  and analyze its performance using `time-graph`. Beware that this has a big performance
  penalty.

## Example

The following example shows how to schedule a sound to be played after 2 seconds.

```no_run
use std::fs::File;
use std::time::Duration;

use rodio::{Decoder, Source, OutputStreamBuilder};
use rodio_scheduler::{Scheduler, PlaybackEvent};

# fn main() {
    // Get an output stream handle to the default physical sound device.
    let stream = OutputStreamBuilder::open_default_stream().unwrap();

    // A source to play as the background audio.
    let background = rodio::source::SineWave::new(440.0);

    // Create a scheduler.
    let mut scheduler = Scheduler::new(background, 48000, 2);

    // Load a sound to be scheduled.
    let file = File::open("assets/note_hit.wav").unwrap();
    let note_hit = Decoder::new(file).unwrap();

    // Add the sound to the scheduler.
    let note_hit_id = scheduler.add_source(note_hit);

    // Schedule the sound to be played at 2 seconds.
    let event = PlaybackEvent {
        source_id: note_hit_id,
        timestamp: 48000 * 2, // 2 seconds in samples
        repeat: None,
    };
    scheduler.get_scheduler(note_hit_id).unwrap().schedule_event(event);

    // Play the scheduled sounds.
    let _ = stream.mixer().add(scheduler);

    // The sound plays in a separate audio thread, so we need to keep the main
    // thread alive while it's playing.
    std::thread::sleep(Duration::from_secs(5));
# }
```
*/

// rodio_scheduler requires nightly rust, because portable-simd is not stabilized yet.
#![feature(portable_simd)]

use rtsan_standalone::nonblocking;

#[cfg(feature = "profiler")]
use time_graph::instrument;

pub mod simd;
pub mod simd_utils;

use std::time::Duration;

use rodio::Sample;
use rodio::source::{SeekError, Source, UniformSourceIterator};

type SampleType = u64;

/// Represents a playback event to be scheduled.
pub struct PlaybackEvent {
    /// The identifier of the source to be played.
    pub source_id: usize,

    /// The timestamp at which the event should occur, measured in samples.
    /// The user is responsible for providing a timestamp that is compatible with the scheduler's sample rate.
    pub timestamp: SampleType,

    /// An optional repeat configuration.
    ///
    /// The tuple contains two values:
    /// 1. The duration of a single beat in samples.
    /// 2. The number of times the beat should be repeated.
    pub repeat: Option<(SampleType, SampleType)>,
}

/// A source that schedules playback for a single audio source at precise timestamps.
/// 
/// The source is fully loaded in memory when the scheduler is created, so scheduling long sources could
/// result in a large memory allocation.
pub struct SingleSourceScheduler {
    /// Backing buffer storing the sample to be scheduled.
    source: Vec<f32>,

    /// The target channel count.
    channels: u16,

    /// The target sample rate.
    sample_rate: u32,

    /// The playback position of each event scheduled for this source, in samples.
    playback_schedule: Vec<SampleType>,

    /// An internal window for currently playing events in this source.
    ///
    /// The first value of the tuple is the index to the oldest playback event 
    /// that is still playing, while the second value is the index to the newest 
    /// playback event that has not started playing yet. When they are the same, 
    /// no sounds are playing.
    playback_position: (usize, usize),

    /// Number of samples counted.
    /// We only keep track of samples counted, since the underlying source will be
    /// from an UniformSourceIterator.
    samples_counted: SampleType,
}

impl SingleSourceScheduler {
    /// Creates a new `SingleSourceScheduler`.
    ///
    /// # Arguments
    ///
    /// * `source`: The audio source to be scheduled.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    #[inline]
    pub fn new(source: impl Source, sample_rate: u32, channels: u16) -> SingleSourceScheduler {
        SingleSourceScheduler {
            source: UniformSourceIterator::new(source, channels, sample_rate).collect(),
            channels,
            sample_rate,
            playback_schedule: Vec::with_capacity(1000),
            playback_position: (0, 0),
            samples_counted: 0,
        }
    }

    /// Schedules a `PlaybackEvent` for this source.
    ///
    /// The event's timestamp is converted to a sample index and added to the playback schedule.
    /// The schedule is then sorted to ensure correct playback order.
    #[inline]
    pub fn schedule_event(&mut self, event: PlaybackEvent) {
        self.playback_schedule
            .push(event.timestamp * self.channels as SampleType);
        self.playback_schedule.sort();
    }
}

impl Iterator for SingleSourceScheduler {
    type Item = Sample;

    #[inline]
    #[nonblocking]
    #[cfg_attr(feature = "profiler", instrument(name = "SingleSourceScheduler::next"))]
    fn next(&mut self) -> Option<Sample> {
        // Cache the sample index for this sample
        let s = self.samples_counted;

        // Set the sample index for the next sample
        self.samples_counted += 1;

        // Update the playback position
        if !self.playback_schedule.is_empty() {
            let source_size: SampleType = self.source.len() as SampleType - 1;
            let schedule_size: usize = self.playback_schedule.len() - 1;

            while self.playback_position.0 < schedule_size
                && (self.playback_schedule[self.playback_position.0] + source_size) < s
            {
                self.playback_position.0 += 1
            }

            while self.playback_position.1 <= schedule_size
                && self.playback_schedule[self.playback_position.1] <= s
            {
                self.playback_position.1 += 1
            }
        }

        simd::retrieve_and_mix_samples(
            &self.source,
            &self.playback_schedule,
            self.playback_position,
            s,
        )
    }

    #[inline]
    #[cfg_attr(
        feature = "profiler",
        instrument(name = "SingleSourceScheduler::size_hint")
    )]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let last_element: usize = self.playback_schedule[self.playback_schedule.len() - 1]
            .try_into()
            .unwrap_or(usize::MAX);
        let lower_bound = last_element + self.source.len();

        (lower_bound, None)
    }
}

impl Source for SingleSourceScheduler {
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    #[inline]
    fn channels(&self) -> u16 {
        self.channels
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        None
    }

    #[inline]
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        let samples_secs = pos.as_secs() * self.sample_rate() as SampleType;

        let nanos_per_sample = 1_000_000_000 / self.sample_rate();
        let samples_nanos = pos.subsec_nanos() / nanos_per_sample;

        self.samples_counted = (samples_secs + samples_nanos as SampleType) * self.channels as SampleType;

        Ok(())
    }
}

/// A source that schedules playback of other sources at precise timestamps.
///
/// The `Scheduler` takes an input source and allows you to schedule additional sources
/// to be played at specific sample timestamps. It mixes the output of the scheduled sources
/// with the input source.
///
/// # Type Parameters
///
/// * `I`: The type of the input audio source, which will play behind other scheduled sources.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
///
/// use rodio::{Decoder, Source, OutputStreamBuilder};
/// use rodio_scheduler::{Scheduler, PlaybackEvent};
///
/// # fn main() {
///    // Get an output stream handle to the default physical sound device.
///    let stream = OutputStreamBuilder::open_default_stream().unwrap();
///
///    // Load a sound from a file.
///    let metronome = File::open("assets/metronome.wav").unwrap();
///    let metronome_decoder_source = Decoder::new(metronome).unwrap();
///
///    // Create a scheduler.
///    let mut scheduler = Scheduler::new(metronome_decoder_source, 48000, 2);
///
///    // Load a sound to be scheduled.
///    let note_hit = File::open("assets/note_hit.wav").unwrap();
///    let note_hit_decoder_source = Decoder::new(note_hit).unwrap();
///
///    // Add the sound to the scheduler, with a list of playback events to schedule.
///    let note_hit_id = scheduler.add_source(note_hit_decoder_source);
///
///    // Schedule the sound to be played at a specific timestamp.
///    let event = PlaybackEvent {
///        source_id: note_hit_id,
///        timestamp: scheduler.sample_rate() as u64 * 2, // 2 seconds in
///        repeat: None,
///    };
///    scheduler.get_scheduler(note_hit_id).unwrap().schedule_event(event);
///
///    // Load another sound to be scheduled.
///    let sine_clip = rodio::source::SineWave::new(440.0).take_duration(std::time::Duration::from_millis(500));
///
///    // Schedule the new sound.
///    let sine_clip_id = scheduler.add_source(sine_clip);
///    let event = PlaybackEvent {
///        source_id: sine_clip_id,
///        timestamp: scheduler.sample_rate() as u64 * 4, // 4 seconds in
///        repeat: None,
///    };
///    scheduler.get_scheduler(sine_clip_id).unwrap().schedule_event(event);
///
///    // Play the scheduled sounds.
///    let _ = stream.mixer().add(scheduler);
///
///    // The sound plays in a separate audio thread,
///    // so we need to keep the main thread alive while it's playing.
///    std::thread::sleep(std::time::Duration::from_secs(5));
/// # }
/// ```
pub struct Scheduler<I>
where
    I: Source,
{
    /// The main input source that the scheduled sources will be mixed with.
    input: UniformSourceIterator<I>,
    /// A vector of `SingleSourceScheduler`s, each managing a single scheduled source.
    sources: Vec<SingleSourceScheduler>,
}

impl<I> Scheduler<I>
where
    I: Source,
{
    /// Creates a new `Scheduler`.
    ///
    /// # Arguments
    ///
    /// * `input`: The main audio source.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    #[inline]
    pub fn new(input: I, sample_rate: u32, channels: u16) -> Scheduler<I> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::new(),
        }
    }

    /// Creates a new `Scheduler` with a specified capacity for scheduled sources.
    ///
    /// # Arguments
    ///
    /// * `input`: The main audio source.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    /// * `capacity`: The initial capacity for the number of scheduled sources.
    #[inline]
    pub fn with_capacity(
        input: I,
        sample_rate: u32,
        channels: u16,
        capacity: usize,
    ) -> Scheduler<I> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::with_capacity(capacity),
        }
    }

    /// Adds a new source to the scheduler.
    ///
    /// Returns a `usize` identifier for the new source, which can be used to schedule playback events.
    #[inline]
    #[cfg_attr(feature = "profiler", instrument)]
    pub fn add_source(&mut self, source: impl Source) -> usize {
        let source_scheduler: SingleSourceScheduler =
            SingleSourceScheduler::new(source, self.sample_rate(), self.channels());

        self.sources.push(source_scheduler);

        self.sources.len() - 1
    }

    /// Retrieves a mutable reference to a `SingleSourceScheduler` by its ID.
    ///
    /// This allows you to schedule events for a specific source.
    #[inline]
    #[cfg_attr(feature = "profiler", instrument)]
    pub fn get_scheduler(&mut self, source_idx: usize) -> Option<&mut SingleSourceScheduler> {
        self.sources.get_mut(source_idx)
    }
}

impl<I> Iterator for Scheduler<I>
where
    I: Source,
{
    type Item = Sample;

    #[inline]
    #[nonblocking]
    #[cfg_attr(feature = "profiler", instrument(name = "Scheduler::next"))]
    fn next(&mut self) -> Option<Sample> {
        let input_sample = self.input.next();

        let playing_samples: Vec<Sample> = self
            .sources
            .iter_mut()
            .filter_map(|source| source.next())
            .collect();

        // Mix scheduled and input samples
        simd::mix_samples(playing_samples.as_slice(), input_sample)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

impl<I> Source for Scheduler<I>
where
    I: Source,
{
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        self.input.current_span_len()
    }

    #[inline]
    fn channels(&self) -> u16 {
        self.input.channels()
    }

    #[inline]
    fn sample_rate(&self) -> u32 {
        self.input.sample_rate()
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration()
    }

    #[inline]
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.input.try_seek(pos)
    }
}
