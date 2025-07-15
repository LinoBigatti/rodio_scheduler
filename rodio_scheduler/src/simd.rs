//! This module provides SIMD-accelerated and scalar fallback functions for audio processing.
//!
//! The functions in this module are used to retrieve and mix audio samples.
//! When the `simd` feature is enabled, SIMD instructions are used to process samples in parallel,
//! which can lead to significant performance improvements. Otherwise, a scalar fallback is used.

#[cfg(feature = "profiler")]
use time_graph::instrument;

#[cfg(feature = "simd")]
use std::simd::cmp::SimdPartialEq;
#[cfg(feature = "simd")]
use std::simd::cmp::SimdPartialOrd;
#[cfg(feature = "simd")]
use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount};

#[cfg(feature = "simd")]
use crate::simd_utils::SimdOps;
#[cfg(feature = "simd")]
use crate::simd_utils::{SimdIter, SimdIterator, gather_select_or_checked_u64};

use rodio::Sample;

/// Retrieves samples from a source based on a playback schedule.
///
/// This is a scalar fallback function used when the `simd` feature is not enabled.
#[inline]
#[cfg(not(feature = "simd"))]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_samples_scalar<'a>(
    source: &'a [Sample],
    playback_schedule: &[u64],
    queue_index: (usize, usize),
    sample_n: u64,
) -> Vec<Sample> {
    if playback_schedule.len() == 0 || queue_index.0 == queue_index.1 {
        return Vec::new();
    }

    let playback_queue: &[u64] = &playback_schedule[queue_index.0..queue_index.1];
    let mut output = Vec::with_capacity(playback_queue.len());

    if playback_queue.len() == 0 {
        output.push(0.0);

        return output;
    }

    for &timestamp in playback_queue {
        if timestamp > sample_n {
            output.push(0.0);

            continue;
        };

        let index = (sample_n - timestamp) as usize;
        if index >= source.len() {
            output.push(0.0);

            continue;
        }

        if let Some(sample) = source.get(index) {
            output.push(*sample);
        }
    }

    output
}

/// Mixes a slice of samples with an input sample.
///
/// This is a scalar fallback function used when the `simd` feature is not enabled.
#[inline]
#[cfg(not(feature = "simd"))]
#[cfg_attr(feature = "profiler", instrument)]
pub fn mix_samples_scalar(samples: &[Sample], input_sample: Option<Sample>) -> Option<Sample> {
    samples
        .into_iter()
        .fold(input_sample, |accumulator, sample| match accumulator {
            Some(s1) => Some(s1 + sample),
            // If you want to make scheduled playback stop after the input Source ended, return None here
            None => Some(*sample),
        })
}

/// Retrieves samples from a source based on a playback schedule using SIMD instructions.
///
/// This function is used when the `simd` feature is enabled.
#[inline]
#[cfg(feature = "simd")]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_samples_simd<'a, const N: usize>(
    source: &'a [Sample],
    playback_schedule: &'a [u64],
    queue_index: (usize, usize),
    sample_n: u64,
) -> impl SimdIterator<Sample, N> + 'a
where
    LaneCount<N>: SupportedLaneCount,
{
    let playback_queue: &'a [u64] = &playback_schedule[queue_index.0..queue_index.1];

    // Decompose the unaligned slice into a &[u64] prefix, a &[Simd<u64, N>] middle and a &[u64]
    // suffix. Afterwards, we can load the prefix and suffix into simd vectors.

    // IMPORTANT: We fill the default case with u64::MAX so that it is ignored when selecting
    // the samples (gather_select_or_checked_u64 will only retrieve indexes below usize::MAX).
    let out_of_bounds = Simd::splat(u64::MAX);

    let simd_iter: SimdIter<'a, u64, N> = SimdIter::from_slice_or(playback_queue, out_of_bounds);

    let f = move |(data, load_mask): (Simd<u64, N>, Mask<_, N>)| {
        let simd_sample_n = Simd::splat(sample_n);

        let idxs = simd_sample_n - data;

        // Safeguard: Dont gather indexes set as out of bounds or that happen after the current sample_n.
        let mask = !data.simd_eq(out_of_bounds) & data.simd_le(simd_sample_n) & load_mask;

        (
            gather_select_or_checked_u64(source, idxs, mask, Simd::splat(0.0)),
            Mask::splat(true),
        )
    };

    simd_iter.map(f)
}

/// Mixes a slice of samples with an input sample using SIMD instructions.
///
/// This function is used when the `simd` feature is enabled.
#[inline]
#[cfg(feature = "simd")]
#[cfg_attr(feature = "profiler", instrument)]
fn mix_samples_simd<const N: usize>(
    samples: impl SimdIterator<Sample, N>,
    input_sample: Option<Sample>,
) -> Option<Sample>
where
    LaneCount<N>: SupportedLaneCount,
{
    let res = samples
        .reduce(
            |acc: (Simd<Sample, N>, Mask<_, N>), data: (Simd<Sample, N>, Mask<_, N>)| {
                // Select values where the mask is 1 and zeros elsewhere
                let masked_data = data.1.select(data.0, Simd::splat(0.0));

                // Perform saturating addition
                let result = Sample::add(acc.0, masked_data);

                (result, Mask::splat(true))
            },
        )
        .map(|(data, _mask): (Simd<Sample, N>, Mask<_, N>)| Sample::horizontal_add(data));

    match input_sample {
        Some(s) => Some(s + res.unwrap_or(0.0)),
        None => res,
    }
}

/// Mixes a slice of samples with an input sample.
///
/// This function will use SIMD instructions if the `simd` feature is enabled, otherwise it will
/// use a scalar fallback.
#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn mix_samples(samples: &[Sample], input_sample: Option<Sample>) -> Option<Sample> {
    #[cfg(feature = "simd")]
    {
        let simd_iter: SimdIter<Sample, 4> = SimdIter::from_slice_or_default(samples);

        // SIMD algorithm
        mix_samples_simd::<4>(simd_iter, input_sample).map(|s: Sample| s.clamp(-1.0, 1.0))
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback scalar algorithm
        mix_samples_scalar(samples, input_sample).map(|s: Sample| s.clamp(-1.0, 1.0))
    }
}

/// Retrieves and mixes samples from a source.
///
/// This function will use SIMD instructions if the `simd` feature is enabled, otherwise it will
/// use a scalar fallback.
#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_and_mix_samples<'a>(
    source: &'a [Sample],
    playback_schedule: &[u64],
    queue_index: (usize, usize),
    sample_n: u64,
) -> Option<Sample> {
    #[cfg(feature = "simd")]
    {
        // SIMD algorithm
        let playing_samples =
            retrieve_samples_simd::<4>(source, playback_schedule, queue_index, sample_n);

        // Mix scheduled and input samples
        mix_samples_simd(playing_samples, None)
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback scalar algorithm
        let playing_samples =
            retrieve_samples_scalar(source, playback_schedule, queue_index, sample_n);

        // Mix scheduled and input samples
        mix_samples_scalar(playing_samples.as_slice(), None)
    }
}
