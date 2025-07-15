use rodio::source::Source;
use std::time::Duration;

#[derive(Clone)]
pub struct DummySource {
    sample_rate: u32,
    channels: u16,
    duration: u64,
    samples_counted: u64,
    channels_counted: u16,
    value: f32,
}

impl DummySource {
    pub fn new(sample_rate: u32, channels: u16, duration: u64, value: f32) -> Self {
        Self {
            sample_rate,
            channels,
            duration,
            channels_counted: 0,
            samples_counted: 0,
            value,
        }
    }
}

impl Iterator for DummySource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        // Get the current sample now before we potentially increase it for the next iteration
        let s = self.samples_counted;

        if self.channels_counted == (self.channels() - 1) {
            self.samples_counted += 1;

            self.channels_counted = 0;
        } else {
            self.channels_counted += 1;
        }

        if s == 0 {
            return Some(self.value);
        }

        if s < self.duration {
            return Some(0.0);
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.duration as usize, Some(self.duration as usize))
    }
}

impl Source for DummySource {
    fn current_span_len(&self) -> Option<usize> {
        Some(self.duration as usize * self.channels as usize)
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs(
            self.duration / self.sample_rate as u64,
        ))
    }
}
