use std::time::Duration;
use std::iter::Peekable;

use rodio::source::{Source, TrackPosition, UniformSourceIterator, SeekError};
use rodio::Sample;

use rodio::cpal::FromSample;

use time_graph::instrument;

use intmap::IntMap;

pub struct PlaybackEvent {
    pub source_id: usize,
    pub timestamp: u128,
    pub repeat: Option<u128>,
}

pub struct Scheduler<I1, I2, D> 
where
    I1: Source,
    I1::Item: Sample,
    I2: Source + Clone,
    I2::Item: Sample,
    D: FromSample<I1::Item> + FromSample<I2::Item> + Sample,
{
    input: UniformSourceIterator<I1, D>,
    sources: Vec<UniformSourceIterator<I2, D>>,
    playback_schedule: IntMap<u128, D>,
    samples_counted: u128,
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
            input: UniformSourceIterator::new(input, channels, sample_rate),
            sources: Vec::with_capacity(10),
            playback_schedule: IntMap::with_capacity(8000 * 48000 * 2), 
            samples_counted: 0,
        }
    }

    /// Adds a new Source.
    #[inline]
    #[instrument]
    pub fn add_source(&mut self, source: I2) -> usize {
        self.sources.push(UniformSourceIterator::new(source, self.channels(), self.sample_rate()));

        self.sources.len() - 1
    }

    /// Schedule a Source to be played.
    #[inline]
    #[instrument]
    pub fn schedule_event(&mut self, event: PlaybackEvent) {
        let Some(original_source) = self.sources.get(event.source_id) else { return };

        let mut source = original_source.clone();
        
        //let event_frames = source
                              //.enumerate()
                              //.map(|(i, sample)| (event.timestamp + i as u128, sample))
                              //.map(|(sample_i, sample)| {
                                //let new_sample = match self.playback_schedule.get(sample_i) {
                                    //Some(s) => s.saturating_add(sample),
                                    //None => sample,
                                //};

                                //(sample_i, new_sample)
                              //})
                              //.collect::<Vec<_>>()
                              //.into_iter();

        //self.playback_schedule.extend(event_frames); 
        let mut sample_iter: Option<D> = source.next();
        
        let mut i = 0;
        while let Some(sample) = sample_iter {
            let sample_i = event.timestamp + i as u128;

            match self.playback_schedule.entry(sample_i) {
                intmap::Entry::Occupied(mut entry) => entry.insert(entry.get().saturating_add(sample)),
                intmap::Entry::Vacant(mut entry) => *entry.insert(sample),
            };

            sample_iter = source.next();
            i += 1;
        } 
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
    #[instrument]
    fn next(&mut self) -> Option<D> {
        //if self.samples_counted % (48000 * 2) == 0 {
            //self.playing_queue.push(self.sources[0].clone().peekable());
        //}

        let real_samples = self.samples_counted; // / self.channels() as u128;
        self.samples_counted += 1;

        let input_sample = self.input.next();

        let scheduled_sample = self.playback_schedule.get(real_samples);

        match (input_sample, scheduled_sample) {
            (Some(s1), Some(s2)) => Some(s1.saturating_add(*s2)),
            (Some(s1), None) => Some(s1),
            // If you want to make scheduled playback stop after the input Source ended, return None here
            (None, Some(s2)) => Some(*s2),
            (None, None) => None,
        }

        //self.playing_queue
          //.iter_mut()
          //.map(|event| {
            //if real_samples < event.timestamp {
                //return None
            //}
            
            //event.source.next()
          //})
          //.fold(input_sample, |accumulator_sample, new_sample| {
        //})
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
