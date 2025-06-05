use std::time::Duration;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rodio::source::{Source, UniformSourceIterator, SeekError};
use rodio::Sample;

use rodio::cpal::FromSample;

use time_graph::instrument;

use intmap::IntMap;

pub struct PlaybackEvent {
    pub source_id: usize,
    pub timestamp: u64,
    pub repeat: Option<(u64, u64)>,
}

pub struct Scheduler<I, D> 
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample,
{
    input: UniformSourceIterator<I, D>,
    sources: Vec<Vec<D>>,
    playback_schedule: IntMap<u64, D>,
    sample_counter: Arc<AtomicU64>,
    samples_counted: u64,
    channels_counted: u16,
}

impl<I, D> Scheduler<I, D>
where
    //I1: Source,
    //I1::Item: Sample,
    //I2: Source + Clone,
    //I2::Item: Sample,
    //D: FromSample<I1::Item> + FromSample<I2::Item> + Sample,
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample,
{
    /// Creates a new source inside of which sounds can be scheduled.
    #[inline]
    pub fn new(input: I, sample_counter: Arc<AtomicU64>, sample_rate: u32, channels: u16) -> Scheduler<I, D> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::with_capacity(10),
            playback_schedule: IntMap::new(), 
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
            sources: Vec::with_capacity(10),
            playback_schedule: IntMap::with_capacity(capacity as usize * sample_rate as usize), 
            sample_counter: sample_counter,
            samples_counted: 0,
            channels_counted: 0,
        }
    }

    /// Adds a new Source.
    #[inline]
    #[instrument]
    pub fn add_source<I2>(&mut self, source: I2) -> usize 
    where
        I2: Source,
        I2::Item: Sample,
        D: FromSample<I2::Item>,
    {
        let buffered_source: Vec<D> = UniformSourceIterator::new(source, 1, self.sample_rate()).collect();

        self.sources.push(buffered_source);

        self.sources.len() - 1
    }

    /// Schedule a Source to be played.
    #[inline]
    #[instrument]
    pub fn schedule_event(&mut self, event: PlaybackEvent) {
        let Some(original_source) = self.sources.get(event.source_id) else { return };

        let mut i: u64 = 0;
        for sample in original_source.iter() {
            let sample_i = event.timestamp + i;

            match event.repeat {
                Some((samples_per_cycle, cycles)) => for j in 0..cycles {
                    match self.playback_schedule.entry(sample_i + j * samples_per_cycle) {
                        intmap::Entry::Occupied(mut entry) => _ = entry.insert(entry.get().saturating_add(*sample)),
                        intmap::Entry::Vacant(entry) => _ = entry.insert(*sample),
                    };
                },
                None => match self.playback_schedule.entry(sample_i) {
                    intmap::Entry::Occupied(mut entry) => _ = entry.insert(entry.get().saturating_add(*sample)),
                    intmap::Entry::Vacant(entry) => _ = entry.insert(*sample),
                },
            };

            i += 1;
        } 
    }
}

impl<I, D> Iterator for Scheduler<I, D>
where
    I: Source,
    I::Item: Sample,
    D: FromSample<I::Item> + Sample,
{
    type Item = D;

    #[inline]
    #[instrument]
    fn next(&mut self) -> Option<D> {
        let current_samples = self.samples_counted;

        if self.channels_counted == self.channels() {
            self.samples_counted += 1;
            self.channels_counted = 0;

            self.sample_counter.store(self.samples_counted, Ordering::SeqCst);
        } else {
            self.channels_counted += 1;
        }

        let input_sample = self.input.next();

        let scheduled_sample = self.playback_schedule.get(current_samples);

        match (input_sample, scheduled_sample) {
            (Some(s1), Some(s2)) => Some(s1.saturating_add(*s2)),
            (Some(s1), None) => Some(s1),
            // If you want to make scheduled playback stop after the input Source ended, return None here
            (None, Some(s2)) => Some(*s2),
            (None, None) => None,
        }
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
    D: FromSample<I::Item> + Sample,
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
