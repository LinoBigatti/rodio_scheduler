mod common;

use rodio_scheduler::{PlaybackEvent, SingleSourceScheduler};

#[test]
fn test_single_source_scheduler_basic_playback() {
    let sample_rate = 48000_u32;
    let channels = 2;
    let duration_samples = 2 * sample_rate as u64;
    let value = 0.5f32;
    let scheduled_time: u64 = sample_rate as u64 / 2;

    let dummy_source = common::DummySource::new(sample_rate, channels, duration_samples, value);
    let mut scheduler = SingleSourceScheduler::new(dummy_source, sample_rate, channels);

    // Schedule an event to play at 0.5 seconds
    let event = PlaybackEvent {
        source_id: 0, // This is ignored for SingleSourceScheduler
        timestamp: scheduled_time,
        repeat: None,
    };
    scheduler.schedule_event(event);

    // Consume samples and check if the scheduled sound plays
    let mut samples_played = 0;

    let scheduled_end_time = scheduled_time + duration_samples;
    let sample_count = (scheduled_end_time * channels as u64) as usize;

    let mut mask = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        // Get the sample index
        let _i = i as u64;
        let s = _i / channels as u64;

        if let Some(sample) = scheduler.next() {
            samples_played += 1;

            // Check if the sample matches our dummy source's value at the scheduled time
            if s == scheduled_time {
                if sample == value {
                    mask.push(true);
                } else {
                    println!(
                        "The scheduled sample was not present at sample {s} (expected {value}, found {sample})."
                    );
                    mask.push(false);
                }

                continue;
            }

            // Check if the source is still present within the given duration.
            if s > scheduled_time || s < scheduled_end_time {
                if sample == 0.0 {
                    mask.push(true);
                } else {
                    println!(
                        "The source sample was incorrect at sample {} (expected {}, found {}).",
                        s, 0.0, sample
                    );
                    mask.push(false);
                }
            }
        } else if s < scheduled_time || s >= scheduled_end_time {
            mask.push(true);
        } else {
            println!(
                "The source sample was not present at sample {s} (expected Scheduler to return Some(_) within samples {scheduled_time} to {scheduled_end_time}, found None)."
            );
            mask.push(false);
        }
    }

    let expected_sample_count = duration_samples * channels as u64;
    assert!(
        mask.into_iter()
            .reduce(|acc: bool, b: bool| acc & b)
            .unwrap_or(false),
        "Scheduled sound was not detected"
    );
    assert!(
        samples_played == expected_sample_count,
        "An incorrect number of samples was played (Expected {expected_sample_count}, found {samples_played})."
    );
}
