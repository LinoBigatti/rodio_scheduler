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
    let samples = vec![10i16, 20, 30];
    let input_sample = Some(5i16);

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(65i16));
}

#[test]
fn test_mix_samples_none_input() {
    let samples = vec![10i16, 20, 30];
    let input_sample = None;

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(60i16));
}

#[test]
fn test_mix_empty_samples() {
    let samples: Vec<i16> = vec![];
    let input_sample = Some(5i16);

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(5i16));
}

#[test]
fn test_mix_samples_saturating_add() {
    let samples = vec![i16::MAX, 1];
    let input_sample = Some(1i16);

    let result = simd::mix_samples(&samples, input_sample);

    assert_eq!(result, Some(i16::MAX));
}

#[test]
fn test_retrieve_and_mix_samples_basic() {
    let source = vec![10i16, 20, 30, 40, 50];
    let playback_schedule = vec![0, 2, 4];
    let queue_index = (0, 3);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);

    // The samples should be [50, 30, 10]
    assert_eq!(result, Some(90i16));
}

#[test]
fn test_retrieve_and_mix_samples_none_playing() {
    let source = vec![10i16, 20, 30, 40, 50];
    let playback_schedule = vec![10, 12, 14];

    // NOTE: This test contains a situation that shouldn't happen
    // (the window provided by queue_index contains samples from the future)
    // Nevertheless, its useful to test for this as we could want to introduce
    // some sort of grouping in the future.
    let queue_index = (0, 3);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);
    assert_eq!(result, Some(0i16));
}

#[test]
fn test_retrieve_and_mix_samples_empty_schedule() {
    let source = vec![10i16, 20, 30, 40, 50];
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
    let source = vec![10i16, 20, 30];
    let playback_schedule = vec![0, 5, 10];

    // NOTE: This test contains a situation that shouldn't happen
    // (the window provided by queue_index contains samples that have already ended)
    // Nevertheless, its useful to test for this as to prevent errors from the
    // SingleSourceScheduler window implementation from trickling down.
    let queue_index = (0, 3);
    let sample_n = 4;

    let result = simd::retrieve_and_mix_samples(&source, &playback_schedule, queue_index, sample_n);
    
    assert_eq!(result, Some(0i16));
}

#[cfg(feature = "simd")]
mod simd_tests {
    use rodio_scheduler::simd_utils::{gather_select_or_checked_u64, SimdIter, SimdOps};
    use std::simd::{Mask, Simd};

    #[test]
    fn test_simd_iter_exact() {
        let data = vec![1i16, 2, 3, 4, 5, 6, 7, 8];
        let or = Simd::splat(0i16);
        let mut iter = SimdIter::<'_, i16, 4>::from_slice_or(&data, or);

        let (vec1, mask1) = iter.next().unwrap();
        assert_eq!(vec1, Simd::from_array([1, 2, 3, 4]));
        assert_eq!(mask1, Mask::splat(true));

        let (vec2, mask2) = iter.next().unwrap();
        assert_eq!(vec2, Simd::from_array([5, 6, 7, 8]));
        assert_eq!(mask2, Mask::splat(true));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_simd_iter_tail() {
        let data = vec![1i16, 2, 3, 4, 5];
        let or = Simd::splat(0i16);
        let mut iter = SimdIter::<'_, i16, 4>::from_slice_or(&data, or);

        let (vec1, mask1) = iter.next().unwrap();
        assert_eq!(vec1, Simd::from_array([1, 2, 3, 4]));
        assert_eq!(mask1, Mask::splat(true));

        let (vec2, mask2) = iter.next().unwrap();
        assert_eq!(vec2, Simd::from_array([5, 0, 0, 0]));
        assert_eq!(mask2, Mask::from_array([true, false, false, false]));

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_gather_select_or_checked_u64() {
        let source = vec![10i16, 20, 30, 40, 50];
        let or = Simd::splat(0i16);

        // 1. All valid indices
        let idxs1 = Simd::from_array([0u64, 1, 2, 3]);
        let mask1 = Mask::splat(true);
        let result1 = gather_select_or_checked_u64(&source, idxs1, mask1, or);
        assert_eq!(result1, Simd::from_array([10, 20, 30, 40]));

        // 2. Some disabled lanes
        let idxs2 = Simd::from_array([0u64, 10, 2, 3]); // 10 is out of bounds. 3 is just disabled
        let mask2 = Mask::from_array([true, false, true, false]); // disable out of bounds lanes
        let result2 = gather_select_or_checked_u64(&source, idxs2, mask2, or);
        assert_eq!(result2, Simd::from_array([10, 0, 30, 0]));

        // 3. Test with u64::MAX indices, should be filtered by safe_cast_mask
        let idxs3 = Simd::from_array([0u64, 1, u64::MAX, 3]);
        let mask3 = Mask::splat(true);
        let result3 = gather_select_or_checked_u64(&source, idxs3, mask3, or);
        assert_eq!(result3, Simd::from_array([10, 20, 0, 40]));
    }

    #[test]
    fn test_simd_ops_i16() {
        let a = Simd::from_array([10i16, i16::MAX, -10, i16::MIN]);
        let b = Simd::from_array([20i16, 1, -5, -1]);

        let add_result = i16::add(a, b);
        assert_eq!(add_result, Simd::from_array([30, i16::MAX, -15, i16::MIN]));

        let mut sum: i16 = 0;
        sum = sum.saturating_add(10i16);
        sum = sum.saturating_add(i16::MAX);
        sum = sum.saturating_add(-10i16);
        sum = sum.saturating_add(i16::MIN);
        let horizontal_add_result = i16::horizontal_add(a);
        assert_eq!(horizontal_add_result, sum);
    }

    #[test]
    fn test_simd_ops_u16() {
        let a = Simd::from_array([10u16, u16::MAX, 0, <u16 as rodio::Sample>::zero_value()]);
        let b = Simd::from_array([20u16, 1, 20, 1]);

        let add_result = u16::add(a, b);
        assert_eq!(add_result, Simd::from_array([30, u16::MAX, 20, <u16 as rodio::Sample>::zero_value() + 1]));

        let mut sum: u16 = 0;
        sum = sum.saturating_add(10u16);
        sum = sum.saturating_add(u16::MAX);
        sum = sum.saturating_add(0u16);
        sum = sum.saturating_add(<u16 as rodio::Sample>::zero_value());
        let horizontal_add_result = u16::horizontal_add(a);
        assert_eq!(horizontal_add_result, sum);
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
