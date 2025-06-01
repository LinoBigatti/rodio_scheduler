use std::time::Duration;

use rodio::source::{Source, TrackPosition, UniformSourceIterator, SeekError};
use rodio::Sample;

use rodio::cpal::FromSample;

pub struct Scheduler<I1, I2, D> 
where
    I1: Source,
    I1::Item: Sample,
    I2: Source + Clone,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample,
{
    input: TrackPosition<UniformSourceIterator<I1, D>>,
    sources: Vec<UniformSourceIterator<I2, D>>,
    playing_queue: Vec<UniformSourceIterator<I2, D>>,
    samples_counted: u64,
}

impl<I1, I2, D> Scheduler<I1, I2, D>
where
    I1: Source,
    I1::Item: Sample,
    I2: Source + Clone,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample,
{
    /// Creates a new source inside of which sounds can be scheduled.
    #[inline]
    pub fn new(input: I1, sample_rate: u32, channels: u16) -> Scheduler<I1, I2, D> {
        Scheduler {
            input: UniformSourceIterator::new(input, channels, sample_rate).track_position(),
            sources: Vec::with_capacity(10),
            playing_queue: Vec::with_capacity(10),
            samples_counted: 0,
        }
    }

    /// Adds a new Scheduled Source.
    #[inline]
    pub fn add_source(&mut self, source: I2) {
        self.sources.push(UniformSourceIterator::new(source, self.channels(), self.sample_rate()));
    }
}

impl<I1, I2, D> Iterator for Scheduler<I1, I2, D>
where
    I1: Source,
    I1::Item: Sample,
    I2: Source + Clone,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample,
{
    type Item = D;

    #[inline]
    fn next(&mut self) -> Option<D> {
        if self.samples_counted % (48000 * 2) == 0 {
            self.playing_queue.push(self.sources[0].clone());
        }

        self.samples_counted += 1;

        let input_sample = self.input.next();

        self.playing_queue
          .iter_mut()
          .map(|source| source.next())
          .fold(input_sample, |accumulator_sample, new_sample| {
            match (accumulator_sample, new_sample) {
                (Some(s1), Some(s2)) => Some(s1.saturating_add(s2)),
                (Some(s1), None) => Some(s1),
                (None, Some(s2)) => Some(s2),
                (None, None) => None,
            }
        })
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
    I2: Source + Clone,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample,
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
