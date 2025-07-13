// rodio_scheduler requires nightly rust, because portable-simd is not stabilized yet.
#![feature(portable_simd)]

use rodio_scheduler::SampleCounter;
use rodio_scheduler::simd;

#[test]
fn test_sample_counter_new() {
    let counter = SampleCounter::new();

    assert_eq!(counter.get(), 0);
}

#[test]
fn test_sample_counter_set() {
    let counter = SampleCounter::new();

    counter.set(100);

    assert_eq!(counter.get(), 100);
}

#[test]
fn test_sample_counter_increment() {
    let counter = SampleCounter::new();

    counter.increment();
    assert_eq!(counter.get(), 1);

    counter.set(99);
    counter.increment();
    assert_eq!(counter.get(), 100);
}

#[test]
fn test_mix_samples_some_input() {
    let samples = vec![0.1f32, 0.2, 0.3];
    let input_sample = Some(-0.05f32);

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(0.55f32));
}

#[test]
fn test_mix_samples_none_input() {
    let samples = vec![0.1f32, 0.2, 0.3];
    let input_sample = None;

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(0.6f32));
}

#[test]
fn test_mix_empty_samples() {
    let samples: Vec<f32> = vec![];
    let input_sample = Some(0.5f32);

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(0.5f32));
}

#[test]
fn test_mix_samples_clamping() {
    let samples = vec![1.0f32, 1.0, 23.0];
    let input_sample = Some(1.0f32);

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(1.0f32));
}

#[test]
fn test_retrieve_and_mix_samples_basic() {
    let source = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let playback_schedule = vec![0, 2, 4];
    let queue_index = (0, 3);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);

    // The samples should be [0.5, 0.3, 0.1]
    assert_eq!(result, Some(0.5f32 + 0.3 + 0.1));
}

#[test]
fn test_retrieve_and_mix_samples_none_playing() {
    let source = vec![1.0f32, 0.1, 0.2, -4.0, 0.0];
    let playback_schedule = vec![10, 12, 14];

    // NOTE: This test contains a situation that shouldn't happen
    // (the window provided by queue_index contains samples from the future)
    // Nevertheless, its useful to test for this as we could want to introduce
    // some sort of grouping in the future.
    let queue_index = (0, 3);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);
    assert_eq!(result, Some(0.0f32));
}

#[test]
fn test_retrieve_and_mix_samples_empty_schedule() {
    let source = vec![1.0f32, 0.0, -3.0, 0.2, 0.5];
    let playback_schedule = vec![];
    let queue_index = (0, 0);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);

    // retrieve_and_mix_samples should return None here because if the playback_schedule is empty,
    // then the scheduled source has ended (or never started).
    assert_eq!(result, None);
}

#[test]
fn test_retrieve_and_mix_samples_scalar_out_of_bounds() {
    let source = vec![0.0f32, 0.1, 0.2];
    let playback_schedule = vec![0, 5, 10];

    // NOTE: This test contains a situation that shouldn't happen
    // (the window provided by queue_index contains samples that have already ended)
    // Nevertheless, its useful to test for this as to prevent errors from the
    // SingleSourceScheduler window implementation from trickling down.
    let queue_index = (0, 3);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);
    
    assert_eq!(result, Some(0.0f32));
}

#[cfg(feature = "simd")]
mod simd_tests {
    use rodio_scheduler::simd_utils::{gather_select_or_checked_u64, SimdIter, SimdOps};
    use std::simd::{Mask, Simd};

    #[test]
    fn test_simd_iter_exact() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let or = Simd::splat(0.0f32);
        let mut iter = SimdIter::<'_, f32, 4>::from_slice_or(&data, or);

        let (vec1, mask1) = iter.next().unwrap();
        assert_eq!(vec1, Simd::from_array([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(mask1, Mask::splat(true));

        let (vec2, mask2) = iter.next().unwrap();
        assert_eq!(vec2, Simd::from_array([5.0, 6.0, 7.0, 8.0]));
        assert_eq!(mask2, Mask::splat(true));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_simd_iter_tail() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let or = Simd::splat(0.0f32);
        let mut iter = SimdIter::<'_, f32, 4>::from_slice_or(&data, or);

        let (vec1, mask1) = iter.next().unwrap();
        assert_eq!(vec1, Simd::from_array([1.0, 2.0, 3.0, 4.0]));
        assert_eq!(mask1, Mask::splat(true));

        let (vec2, mask2) = iter.next().unwrap();
        assert_eq!(vec2, Simd::from_array([5.0, 0.0, 0.0, 0.0]));
        assert_eq!(mask2, Mask::from_array([true, false, false, false]));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_gather_select_or_checked_u64() {
        let source = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let or = Simd::splat(0.0f32);

        // 1. All valid indices
        let idxs1 = Simd::from_array([0u64, 1, 2, 3]);
        let mask1 = Mask::splat(true);
        let result1 = gather_select_or_checked_u64(&source, idxs1, mask1, or);
        assert_eq!(result1, Simd::from_array([10.0, 20.0, 30.0, 40.0]));

        // 2. Some disabled lanes
        let idxs2 = Simd::from_array([0u64, 10, 2, 3]); // 10 is out of bounds. 3 is just disabled
        let mask2 = Mask::from_array([true, false, true, false]); // disable out of bounds lanes
        let result2 = gather_select_or_checked_u64(&source, idxs2, mask2, or);
        assert_eq!(result2, Simd::from_array([10.0, 0.0, 30.0, 0.0]));

        // 3. Test with u64::MAX indices, should be filtered by safe_cast_mask
        let idxs3 = Simd::from_array([0u64, 1, u64::MAX, 3]);
        let mask3 = Mask::splat(true);
        let result3 = gather_select_or_checked_u64(&source, idxs3, mask3, or);
        assert_eq!(result3, Simd::from_array([10.0, 20.0, 0.0, 40.0]));
    }

    #[test]
    fn test_simd_ops_f32() {
        let a = Simd::from_array([10.5f32, 0.5, -10.0, 1.0]);
        let b = Simd::from_array([20.0f32, 0.5, -5.5, -1.0]);

        let add_result = f32::add(a, b);
        assert_eq!(add_result, Simd::from_array([30.5, 1.0, -15.5, 0.0]));

        let horizontal_add_result = f32::horizontal_add(a);
        assert_eq!(horizontal_add_result, 10.5 + 0.5 - 10.0 + 1.0);
    }
}
