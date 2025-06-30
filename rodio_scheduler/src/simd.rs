#[cfg(feature="profiler")]
use time_graph::instrument;

use std::simd::SimdElement; 
#[cfg(feature="simd")]
use std::simd::{Simd, Mask, LaneCount, SupportedLaneCount};
#[cfg(feature="simd")]
use std::simd::cmp::SimdPartialOrd;
#[cfg(feature="simd")]
use std::simd::cmp::SimdPartialEq;
#[cfg(feature="simd")]
use std::simd::num::SimdUint;

#[cfg(feature="simd")]
use crate::simd_iter::{SimdIter, SimdIterator};

#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_samples_scalar<'a, D: rodio::Sample + Default>(source: &'a [D], playback_schedule: &[u64], queue_index: (usize, usize), sample_n: u64) -> Vec<D> {
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

#[cfg(feature = "simd")]
#[cfg_attr(feature = "profiler", instrument)]
pub fn gather_select_or_default_checked<T, const N: usize>(source: &[T], idxs: Simd<u64, N>, mask: Mask<i64, N>) -> Simd<T, N> where 
    T: Default + SimdElement,
    LaneCount<N>: SupportedLaneCount,
{
    let safe_cast_mask = idxs.simd_le(Simd::splat(usize::MAX as u64));

    // This will perform lane-wise casting from u64 to usize (which is platform dependent)
    // We will only utilize the values below usize::MAX, thanks to the mask above, because 
    // the cast is safe under these circumstances.
    let idxs_usize: Simd<usize, N> = idxs.cast();
    let mask_isize: Mask<isize, N> = (mask & safe_cast_mask).cast();

    //println!("{:?}", idxs_usize);
    Simd::gather_select(source, mask_isize, idxs_usize, Simd::splat(T::default()))
}

#[inline]
#[cfg(feature = "simd")]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_samples_simd<'a, D, const N: usize>(source: &'a [D], playback_schedule: &'a [u64], queue_index: (usize, usize), sample_n: u64) -> impl SimdIterator<D, N> + 'a where 
    D: rodio::Sample + Default + SimdElement,
    LaneCount<N>: SupportedLaneCount,
{
    let playback_queue: &'a [u64] = &playback_schedule[queue_index.0..queue_index.1];

    // Decompose the unaligned slice into a &[u64] prefix, a &[Simd<u64, N>] middle and a &[u64]
    // suffix. Afterwards, we can load the prefix and suffix into simd vectors. 

    // IMPORTANT: We fill the default case with u64::MAX so that it is ignored when selecting
    // the samples (gather_select_or_default_checked will only retrieve indexes below usize::MAX).
    let out_of_bounds = Simd::splat(u64::MAX);

    let simd_iter: SimdIter<'a, u64, N> = SimdIter::from_slice_or(playback_queue, out_of_bounds);

    let f = move |(data, _): (Simd<u64, N>, Mask<_, N>)| {
        let simd_sample_n = Simd::splat(sample_n);

        let idxs = simd_sample_n - data;

        // Safeguard: Dont gather indexes set as out of bounds or that happen after the current sample_n.
        let mask = !data.simd_eq(out_of_bounds) & data.simd_le(simd_sample_n);

        (gather_select_or_default_checked(source, idxs, mask), Mask::splat(true))
    };

    simd_iter.map(f)
}

#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn retrieve_and_mix_samples<'a, D>(source: &'a [D], playback_schedule: &[u64], queue_index: (usize, usize), sample_n: u64) -> Option<D> 
where
    D: rodio::Sample + Default + SimdElement,
{
    #[cfg(feature = "simd")]
    {
        let playing_samples: Vec<D> = retrieve_samples_simd::<D, 4>(source, playback_schedule, queue_index, sample_n)
            .flat_map(|s| s.0.as_array().to_owned())
            .collect();

        // Mix scheduled and input samples
        simd_mix_samples(playing_samples.as_slice(), None)
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback scalar algorithm
        let playing_samples = retrieve_samples_scalar(source, playback_schedule, queue_index, sample_n);

        // Mix scheduled and input samples
        simd_mix_samples(playing_samples.as_slice(), None)
    }
}


#[inline]
#[cfg_attr(feature = "profiler", instrument)]
pub fn simd_mix_samples<D: rodio::Sample>(samples: &[D], input_sample: Option<D>) -> Option<D> {
    samples.into_iter().fold(input_sample, |accumulator, sample| match accumulator {
        Some(s1) => Some(s1.saturating_add(*sample)),
        // If you want to make scheduled playback stop after the input Source ended, return None here
        None => Some(*sample),
    })
}
