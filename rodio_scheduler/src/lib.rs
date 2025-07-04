//! A library for scheduling audio playback with `rodio`.
//!
//! `rodio_scheduler` provides a `rodio` `Source` that can schedule other sources to be played
//! at specific timestamps. This is useful for applications that require precise audio
//! scheduling, such as rhythm games, digital audio workstations (DAWs), or music players.
//!
//! ## Nightly Rust Requirement
//!
//! This crate requires the nightly Rust compiler because it uses the `portable-simd` feature,
//! which has not yet been stabilized.
//!
//! ## Features
//!
//! - **Sample-perfect Scheduling**: Schedule audio playback with sample-level accuracy.
//! - **Atomic Sample Counter**: Provides a thread-safe sample counter to synchronize
//!   external events with audio playback.
//! - **SIMD Acceleration**: Uses SIMD for mixing audio samples, providing a small
//!   performance boost. This can be enabled with the `simd` feature flag.
//! - **Optional Profiling**: Includes an optional `profiler` feature to instrument the code
//!   and analyze its performance using `time-graph`. Beware that this has a big performance
//!   penalty.
//!
//! ## Example
//!
//! The following example shows how to schedule a sound to be played after 2 seconds.
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use std::time::Duration;
//!
//! use rodio::{Decoder, OutputStream, source::Source};
//! use rodio_scheduler::{Scheduler, PlaybackEvent};
//!
//! fn main() {
//!     // Get an output stream handle to the default physical sound device.
//!     let (_stream, stream_handle) = OutputStream::try_default().unwrap();
//!
//!     // A source to play as the background audio.
//!     let background = rodio::source::SineWave::new(440.0);
//!
//!     // Create a scheduler.
//!     let mut scheduler = Scheduler::<_, _, f32>::new(background, 48000, 2);
//!
//!     // Load a sound to be scheduled.
//!     let file = BufReader::new(File::open("assets/note_hit.wav").unwrap());
//!     let note_hit = Decoder::new(file).unwrap();
//!
//!     // Add the sound to the scheduler.
//!     let note_hit_id = scheduler.add_source(note_hit);
//!
//!     // Schedule the sound to be played at 2 seconds.
//!     let event = PlaybackEvent {
//!         source_id: note_hit_id,
//!         timestamp: 48000 * 2, // 2 seconds in samples
//!         repeat: None,
//!     };
//!     scheduler.get_scheduler(note_hit_id).unwrap().schedule_event(event);
//!
//!     // Play the scheduled sounds.
//!     stream_handle.play_raw(scheduler.convert_samples()).unwrap();
//!
//!     // The sound plays in a separate audio thread, so we need to keep the main
//!     // thread alive while it's playing.
//!     std::thread::sleep(Duration::from_secs(5));
//! }
//! ```

// rodio_scheduler requires nightly rust, because portable-simd is not stabilized yet.
#![feature(portable_simd)]

#[cfg(feature="profiler")]
use time_graph::instrument;

mod simd;
mod simd_utils;
use simd_utils::SimdOps;

use std::time::Duration;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rodio::source::{Source, UniformSourceIterator, SeekError};
use rodio::Sample;

use rodio::cpal::FromSample;

/// Represents a playback event to be scheduled.
pub struct PlaybackEvent {
    /// The identifier of the source to be played.
    pub source_id: usize,
    /// The timestamp at which the event should occur, measured in samples.
    /// The user is responsible for providing a timestamp that is compatible with the scheduler's sample rate.
    pub timestamp: u64,
    /// An optional repeat configuration.
    ///
    /// The tuple contains two values:
    /// 1. The duration of a single beat in samples.
    /// 2. The number of times the beat should be repeated.
    pub repeat: Option<(u64, u64)>,
}

/// A source that schedules playback of a single audio source at precise timestamps.
///
/// # Type Parameters
///
/// * `I`: The type of the input audio source.
/// * `D`: The sample type used for processing and output, after conversion from `I::Item`.
pub struct SingleSourceScheduler<I, D> 
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
{
    source: Vec<D>,
    channels: u16,
    sample_rate: u32,
    playback_schedule: Vec<u64>,
    playback_position: (usize, usize),
    // In this case we only keep track of samples counted, since the underlying source will be
    // from an UniformSourceIterator.
    samples_counted: u64,
    _original_source: std::marker::PhantomData<I>,
}

impl<I, D> SingleSourceScheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
{
    /// Creates a new `SingleSourceScheduler`.
    ///
    /// # Arguments
    ///
    /// * `source`: The audio source to be scheduled.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    #[inline]
    pub fn new(source: I, sample_rate: u32, channels: u16) -> SingleSourceScheduler<I, D> {
        SingleSourceScheduler {
            source: UniformSourceIterator::new(source, channels, sample_rate).collect(),
            channels: channels,
            sample_rate: sample_rate,
            playback_schedule: Vec::with_capacity(1000),
            playback_position: (0, 0),
            samples_counted: 0,
            _original_source: std::marker::PhantomData,
        }
    }

    /// Schedules a `PlaybackEvent` for this source.
    ///
    /// The event's timestamp is converted to a sample index and added to the playback schedule.
    /// The schedule is then sorted to ensure correct playback order.
    #[inline]
    pub fn schedule_event(&mut self, event: PlaybackEvent) {
        self.playback_schedule.push(event.timestamp * self.channels as u64);
        self.playback_schedule.sort();
    }
}

impl<I, D> Iterator for SingleSourceScheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
{
    type Item = D;

    #[inline]
    #[cfg_attr(feature = "profiler", instrument(name = "SingleSourceScheduler::next"))]
    fn next(&mut self) -> Option<D> {
        // Set the sample and channel index for the next sample
        self.samples_counted += 1;

        // Update the playback position
        if self.playback_schedule.len() != 0 {
            let source_size: u64 = self.source.len() as u64 - 1;
            let schedule_size: usize = self.playback_schedule.len() - 1;
            
            while self.playback_position.0 < schedule_size &&
                  (self.playback_schedule[self.playback_position.0] + source_size) < self.samples_counted {
                self.playback_position.0 += 1
            }

            while self.playback_position.1 <= schedule_size &&
                  self.playback_schedule[self.playback_position.1] <= self.samples_counted {
                self.playback_position.1 += 1
            }
        }

        simd::retrieve_and_mix_samples(&self.source, &self.playback_schedule, self.playback_position, self.samples_counted)
    }

    #[inline]
    #[cfg_attr(feature = "profiler", instrument(name = "SingleSourceScheduler::size_hint"))]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let last_element: usize = self.playback_schedule[self.playback_schedule.len() - 1].try_into().unwrap_or_else(|_| usize::MAX);
        let lower_bound = last_element + self.source.len();

        (lower_bound, None)
    }
}

impl<I, D> Source for SingleSourceScheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
{
    #[inline]
    fn current_frame_len(&self) -> Option<usize> {
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
        let samples_secs = pos.as_secs() * self.sample_rate() as u64;

        let nanos_per_sample = 1_000_000_000 / self.sample_rate();
        let samples_nanos = pos.subsec_nanos() / nanos_per_sample;

        self.samples_counted = (samples_secs + samples_nanos as u64) * self.channels as u64;
        
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
/// * `I1`: The type of the input audio source.
/// * `I2`: The type of the scheduled audio sources.
/// * `D`: The sample type used for processing and output.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use std::io::BufReader;
/// use std::sync::atomic::Ordering;
///
/// use rodio::{Source, Decoder, OutputStream};
/// use rodio_scheduler::{Scheduler, PlaybackEvent};
///
/// fn main() {
///    // Get an output stream handle to the default physical sound device.
///    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
///
///    // Load a sound from a file.
///    let metronome = BufReader::new(File::open("assets/metronome.wav").unwrap());
///    let metronome_decoder_source = Decoder::new(metronome).unwrap();
///
///    // Create a scheduler.
///    let mut scheduler = Scheduler::<_, _, f32>::new(metronome_decoder_source, 48000, 2);
///
///    // Load another sound to be scheduled.
///    let note_hit = BufReader::new(File::open("assets/note_hit.wav").unwrap());
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
///    // Get the sample counter before moving the scheduler to the audio thread
///    let sample_counter = scheduler.get_sample_counter();
///
///    // Play the scheduled sounds.
///    let _ = stream_handle.play_raw(scheduler.convert_samples());
///
///    // Get the current sample index while playing
///    let _current_samples = sample_counter.load(Ordering::SeqCst);
///    //do_something(current_samples);
///
///    // The sound plays in a separate audio thread,
///    // so we need to keep the main thread alive while it's playing.
///    std::thread::sleep(std::time::Duration::from_secs(5));
///}
/// ```
pub struct Scheduler<I1, I2, D> 
where
    I1: Source,
    I1::Item: Sample,
    I2: Source,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample + SimdOps,
{
    /// The main input source that the scheduled sources will be mixed with.
    input: UniformSourceIterator<I1, D>,
    /// A vector of `SingleSourceScheduler`s, each managing a single scheduled source.
    sources: Vec<SingleSourceScheduler<I2, D>>,
    /// An atomic counter to track the current sample number.
    /// This is managed by the user and should be shared with the audio thread.
    sample_counter: Arc<AtomicU64>,
    /// The number of samples that have been processed.
    samples_counted: u64,
    /// The number of channels that have been processed for the current sample.
    channels_counted: u16,
}

impl<I1, I2, D> Scheduler<I1, I2, D>
where
    I1: Source,
    I1::Item: Sample,
    I2: Source,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample + SimdOps,
{
    /// Creates a new `Scheduler`.
    ///
    /// # Arguments
    ///
    /// * `input`: The main audio source.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    #[inline]
    pub fn new(input: I1, sample_rate: u32, channels: u16) -> Scheduler<I1, I2, D> {
        let sample_counter = Arc::new(AtomicU64::new(0));

        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::new(),
            sample_counter: sample_counter,
            samples_counted: 0,
            channels_counted: 0,
        }
    }

    /// Creates a new `Scheduler`, with a given sample counter.
    ///
    /// # Arguments
    ///
    /// * `input`: The main audio source.
    /// * `sample_counter`: An `Arc<AtomicU64>` to keep track of the playback position.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    #[inline]
    pub fn with_sample_counter(input: I1, sample_counter: Arc<AtomicU64>, sample_rate: u32, channels: u16) -> Scheduler<I1, I2, D> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::new(),
            sample_counter: sample_counter,
            samples_counted: 0,
            channels_counted: 0,
        }
    }

    /// Creates a new `Scheduler` with a specified capacity for scheduled sources.
    ///
    /// # Arguments
    ///
    /// * `input`: The main audio source.
    /// * `sample_counter`: An `Arc<AtomicU64>` to keep track of the playback position.
    /// * `sample_rate`: The sample rate of the output audio.
    /// * `channels`: The number of channels in the output audio.
    /// * `capacity`: The initial capacity for the number of scheduled sources.
    #[inline]
    pub fn with_capacity(input: I1, sample_counter: Arc<AtomicU64>, sample_rate: u32, channels: u16, capacity: usize) -> Scheduler<I1, I2, D> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::with_capacity(capacity),
            sample_counter: sample_counter,
            samples_counted: 0,
            channels_counted: 0,
        }
    }

    /// Adds a new source to the scheduler.
    ///
    /// Returns a `usize` identifier for the new source, which can be used to schedule playback events.
    #[inline]
    #[cfg_attr(feature = "profiler", instrument)]
    pub fn add_source(&mut self, source: I2) -> usize 
    {
        let source_scheduler: SingleSourceScheduler<I2, D> = SingleSourceScheduler::new(source, self.sample_rate(), self.channels());

        self.sources.push(source_scheduler);

        self.sources.len() - 1
    }

    /// Retrieves a mutable reference to a `SingleSourceScheduler` by its ID.
    ///
    /// This allows you to schedule events for a specific source.
    #[inline]
    #[cfg_attr(feature = "profiler", instrument)]
    pub fn get_scheduler(&mut self, source_idx: usize) -> Option<&mut SingleSourceScheduler<I2, D>>
    {
        self.sources.get_mut(source_idx)
    }

    /// Retrieves a reference to the internal high-resolution sample counter.
    ///
    /// This allows you to synchronize external events with the audio playback.
    #[inline]
    pub fn get_sample_counter(&self) -> Arc<AtomicU64>
    {
        self.sample_counter.clone()
    }
}

impl<I1, I2, D> Iterator for Scheduler<I1, I2, D>
where
    I1: Source,
    I1::Item: Sample,
    I2: Source,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample + SimdOps,
{
    type Item = D;

    #[inline]
    #[cfg_attr(feature = "profiler", instrument(name = "Scheduler::next"))]
    fn next(&mut self) -> Option<D> {
        let input_sample = self.input.next();

        // Set the sample and channel index for the next sample
        if self.channels_counted == self.channels() {
            self.samples_counted += 1;
            self.channels_counted = 0;

            self.sample_counter.store(self.samples_counted, Ordering::SeqCst);
        } else {
            self.channels_counted += 1;
        }

        let playing_samples: Vec<D> = self.sources
                                            .iter_mut()
                                            .map(|source| source.next())
                                            .filter_map(|sample| sample)
                                            .collect();

        // Mix scheduled and input samples
        simd::mix_samples(playing_samples.as_slice(), input_sample)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

impl<I1, I2, D> Source for Scheduler<I1, I2, D>
where
    I1: Source,
    I1::Item: Sample,
    I2: Source,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample + SimdOps,
{
    #[inline]
    fn current_frame_len(&self) -> Option<usize> {
        self.input.current_frame_len()
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
