#![feature(portable_simd)]

#[cfg(feature="profiler")]
use time_graph::instrument;

mod simd;
mod simd_utils;
use simd_utils::SimdOps;
//mod simd_macros;

use std::time::Duration;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rodio::source::{Source, UniformSourceIterator, SeekError};
use rodio::Sample;

use rodio::cpal::FromSample;

pub struct PlaybackEvent {
    pub source_id: usize,
    pub timestamp: u64,
    pub repeat: Option<(u64, u64)>,
}

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
    /// Creates a new source inside of which sounds can be scheduled.
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

    /// Schedule this source for precise playback in the future.
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

            while self.playback_position.1 < schedule_size &&
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

pub struct Scheduler<I, D> 
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
{
    input: UniformSourceIterator<I, D>,
    sources: Vec<SingleSourceScheduler<I, D>>,
    sample_counter: Arc<AtomicU64>,
    samples_counted: u64,
    channels_counted: u16,
}

impl<I, D> Scheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
{
    /// Creates a new source inside of which sounds can be scheduled.
    #[inline]
    pub fn new(input: I, sample_counter: Arc<AtomicU64>, sample_rate: u32, channels: u16) -> Scheduler<I, D> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::new(),
            sample_counter: sample_counter,
            samples_counted: 0,
            channels_counted: 0,
        }
    }

    /// Creates a new source inside of which sounds can be scheduled, with a given capacity.
    #[inline]
    pub fn with_capacity(input: I, sample_counter: Arc<AtomicU64>, sample_rate: u32, channels: u16, capacity: usize) -> Scheduler<I, D> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::with_capacity(capacity),
            sample_counter: sample_counter,
            samples_counted: 0,
            channels_counted: 0,
        }
    }

    /// Adds a new Source.
    #[inline]
    #[cfg_attr(feature = "profiler", instrument)]
    pub fn schedule_source(&mut self, source: I) -> usize 
    {
        let source_scheduler: SingleSourceScheduler<I, D> = SingleSourceScheduler::new(source, self.sample_rate(), self.channels());

        self.sources.push(source_scheduler);

        self.sources.len() - 1
    }

    /// Retrieves a mutable reference to a specified Source Scheduler.
    #[inline]
    #[cfg_attr(feature = "profiler", instrument)]
    pub fn get_scheduler(&mut self, source_idx: usize) -> Option<&mut SingleSourceScheduler<I, D>>
    {
        self.sources.get_mut(source_idx)
    }
}

impl<I, D> Iterator for Scheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
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

impl<I, D> Source for Scheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample + SimdOps,
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
