#[cfg(feature="profiler")]
use time_graph::instrument;

#[cfg(feature="simd")]
use std::simd::{Simd, Mask, LaneCount, SupportedLaneCount};
#[cfg(feature="simd")]
use std::simd::cmp::SimdPartialOrd;
#[cfg(feature="simd")]
use std::simd::cmp::SimdPartialEq;

#[cfg(feature="simd")]
use crate::simd_utils::{SimdIter, SimdIterator, gather_select_or_checked_u64};
use crate::simd_utils::SimdOps;

#[inline]
#[cfg(not(feature = "simd"))]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_samples_scalar<'a, D: rodio::Sample>(source: &'a [D], playback_schedule: &[u64], queue_index: (usize, usize), sample_n: u64) -> Vec<D> {
    let playback_queue: &[u64] = &playback_schedule[queue_index.0..queue_index.1];
    let mut output = Vec::with_capacity(playback_queue.len());

    for &timestamp in playback_queue {
        if timestamp > sample_n { continue };

        let index = (sample_n - timestamp) as usize;

        if let Some(sample) = source.get(index) {
            output.push(*sample);
        }
    }

    output
}

#[inline]
#[cfg(not(feature = "simd"))]
#[cfg_attr(feature = "profiler", instrument)]
pub fn mix_samples_scalar<D: rodio::Sample>(samples: &[D], input_sample: Option<D>) -> Option<D> {
    samples.into_iter().fold(input_sample, |accumulator, sample| match accumulator {
        Some(s1) => Some(s1.saturating_add(*sample)),
        // If you want to make scheduled playback stop after the input Source ended, return None here
        None => Some(*sample),
    })
}

#[inline]
#[cfg(feature = "simd")]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_samples_simd<'a, D, const N: usize>(source: &'a [D], playback_schedule: &'a [u64], queue_index: (usize, usize), sample_n: u64) -> impl SimdIterator<D, N> + 'a where 
    D: rodio::Sample + SimdOps,
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

        (gather_select_or_checked_u64(source, idxs, mask, Simd::splat(D::zero_value())), Mask::splat(true))
    };

    simd_iter.map(f)
}

#[inline]
#[cfg(feature = "simd")]
#[cfg_attr(feature = "profiler", instrument)]
fn mix_samples_simd<D, const N: usize>(samples: impl SimdIterator<D, N>, input_sample: Option<D>) -> Option<D>
where
    D: SimdOps,
    LaneCount<N>: SupportedLaneCount,
{
    let res = samples
        .reduce(|acc: (Simd<D, N>, Mask<_, N>), data: (Simd<D, N>, Mask<_, N>)| {
            // Select values where the mask is 1 and zeros elsewhere
            let masked_data = data.1.select(data.0, Simd::splat(D::zero_value()));

            // Perform saturating addition
            let result = D::add(acc.0, masked_data);

            (result, Mask::splat(true))
        })
        .map(|(data, _mask): (Simd<D, N>, Mask<_, N>)| D::horizontal_add(data));

    match input_sample {
        Some(s) => Some(s.saturating_add(res.unwrap_or(D::zero_value()))),
        None => res,
    }
    //if samples.is_empty() {
        //return input_sample;
    //}

    //if samples_to_process.is_empty() {
        //return Some(acc);
    //}

    //let (prefix, middle, suffix) = samples_to_process.as_simd::<N>();

    //let mut simd_sum = Simd::splat(D::zero_value());
    //for &chunk in middle {
        //simd_sum = D::add(simd_sum, chunk);
    //}

    //let mut scalar_sum = D::horizontal_add(simd_sum);

    //for sample in prefix.iter().chain(suffix.iter()) {
        //scalar_sum = scalar_sum.saturating_add(*sample);
    //}

    //acc = acc.saturating_add(scalar_sum);

    //Some(acc)
}

#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn mix_samples<D>(samples: &[D], input_sample: Option<D>) -> Option<D>
where
    D: SimdOps,
{
    #[cfg(feature = "simd")]
    {
        let simd_iter: SimdIter<D, 4> = SimdIter::from_slice_or_zero_value(samples);

        // SIMD algorithm
        mix_samples_simd::<D, 4>(simd_iter, input_sample)
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback scalar algorithm
        mix_samples_scalar(samples, input_sample)
    }
}


#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_and_mix_samples<'a, D>(source: &'a [D], playback_schedule: &[u64], queue_index: (usize, usize), sample_n: u64) -> Option<D> 
where
    D: rodio::Sample + SimdOps,
{
    #[cfg(feature = "simd")]
    {
        // SIMD algorithm
        let playing_samples = retrieve_samples_simd::<D, 4>(source, playback_schedule, queue_index, sample_n);

        // Mix scheduled and input samples
        mix_samples_simd(playing_samples, None)
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback scalar algorithm
        let playing_samples = retrieve_samples_scalar(source, playback_schedule, queue_index, sample_n);

        // Mix scheduled and input samples
        mix_samples_scalar(playing_samples.as_slice(), None)
    }
}
