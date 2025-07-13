//! This module provides SIMD-related utilities and traits.
//!
//! When the `simd` feature is enabled, this module provides traits and structs for
//! working with SIMD vectors, including iterators and operations for different sample types.
//! When the `simd` feature is not enabled, it provides dummy traits to ensure the code compiles.

#[cfg(feature="profiler")]
use time_graph::instrument;

#[cfg(feature="simd")]
use std::simd::{Simd, Mask, SimdElement, LaneCount, SupportedLaneCount};

#[cfg(feature="simd")]
use std::simd::cmp::SimdPartialOrd;

#[cfg(feature="simd")]
use std::simd::num::{SimdFloat, SimdUint};

/// Gathers elements from a source slice into a SIMD vector, with a fallback for out-of-bounds indices.
///
/// This function is used when the `simd` feature is enabled.
#[cfg(feature="simd")]
#[cfg_attr(feature = "profiler", instrument)]
pub fn gather_select_or_checked_u64<T, const N: usize>(source: &[T], idxs: Simd<u64, N>, mask: Mask<i64, N>, or: Simd<T, N>) -> Simd<T, N> where 
    T: SimdOps,
    LaneCount<N>: SupportedLaneCount,
{
    let safe_cast_mask = idxs.simd_le(Simd::splat(usize::MAX as u64));

    // This will perform lane-wise casting from u64 to usize (which is platform dependent)
    // We will only utilize the values below usize::MAX, thanks to the mask above, because 
    // the cast is safe under these circumstances.
    let idxs_usize: Simd<usize, N> = idxs.cast();
    let mask_isize: Mask<isize, N> = (mask & safe_cast_mask).cast();

    // Gather the indices from the source slice
    Simd::gather_select(source, mask_isize, idxs_usize, or)
}

/// A trait for iterators that yield SIMD vectors.
#[cfg(feature="simd")]
pub trait SimdIterator<T, const N: usize>: Iterator<Item = (Simd<T, N>, Mask<T::Mask, N>)> 
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
{}

#[cfg(feature="simd")]
impl<I, T, const N: usize> SimdIterator<T, N> for I 
where
    I: Iterator<Item = (Simd<T, N>, Mask<T::Mask, N>)>,
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
{}

/// An iterator that yields SIMD vectors from a slice.
#[cfg(feature="simd")]
pub struct SimdIter<'a, T, const N: usize> 
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
{
    src: &'a [T],
    or: Simd<T, N>,
    i: usize,
}

#[cfg(feature="simd")]
impl<'a, T, const N: usize> SimdIter<'a, T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
{
    /// Creates a new `SimdIter` from a slice and a fallback SIMD vector.
    #[inline]
    #[allow(dead_code)]
    pub fn from_slice_or(src: &'a [T], or: Simd<T, N>) -> SimdIter<'a, T, N> {
        Self {
            src: src,
            or: or,
            i: 0,
        }
    }

    /// Creates a new `SimdIter` from a slice, with a default value as the fallback.
    #[inline]
    #[allow(dead_code)]
    pub fn from_slice_or_default(src: &'a [T]) -> SimdIter<'a, T, N> where T: Default {
        Self {
            src: src,
            or: Simd::splat(T::default()),
            i: 0,
        }
    }
}

#[cfg(feature="simd")]
impl<T, const N: usize> Iterator for SimdIter<'_, T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount,
{
    type Item = (Simd<T, N>, Mask<T::Mask, N>);

    #[inline]
    #[cfg_attr(feature = "profiler", instrument(name = "SimdIter::next"))]
    fn next(&mut self) -> Option<Self::Item> {
        if self.i * N >= self.src.len() {
            return None
        }

        // Check if our index is still within the body of the slice
        let result = if self.i < (self.src.len() / N) {
            // Load a vector from the body of the slice
            let start = self.i * N;
            let end = (self.i + 1) * N;

            Some(
                (Simd::from_slice(&self.src[start..end]), Mask::splat(true))
            )
        } else {
            // Load a vector from the tail of the slice, filling the out-of-bounds elements with
            // the values from self.or
            let start = self.i * N;
            let end = self.src.len();

            let indices: Vec<usize> = (0..N).collect();
            let indices_simd: Simd<usize, N> = Simd::from_slice(&indices);
            let valid_loads: Mask<T::Mask, N> = indices_simd.simd_lt(Simd::splat(end - start)).cast();

            Some(
                (Simd::load_or(&self.src[start..end], self.or), valid_loads)
            )
        };

        // Advance the iterator
        self.i += 1;
        result
    }

    #[inline]
    #[cfg_attr(feature = "profiler", instrument(name = "SimdIter::size_hint"))]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let body_size = self.src.len() / N;
        let tail_size = if self.src.len() % N == 0 { 0 } else { 1 };
        let size = body_size + tail_size;

        (size, Some(size))
    }
}

/// A trait for types that support SIMD operations.
///
/// When the `simd` feature is not enabled, this is a dummy trait.
#[cfg(not(feature = "simd"))]
pub trait SimdOps: Sized {}

#[cfg(not(feature = "simd"))]
impl<T> SimdOps for T
where
    T: Sized 
{}

/// A trait for types that support SIMD operations.
///
/// When the `simd` feature is enabled, this trait provides methods for SIMD addition,
/// horizontal addition, and clamping for different sample types.
#[cfg(feature = "simd")]
pub trait SimdOps: Sized + SimdElement {
    /// Adds two SIMD vectors.
    fn add<const N: usize>(a: Simd<Self, N>, b: Simd<Self, N>) -> Simd<Self, N>
    where
        LaneCount<N>: SupportedLaneCount;

    /// Horizontally adds the elements of a SIMD vector.
    fn horizontal_add<const N: usize>(a: Simd<Self, N>) -> Self
    where
        LaneCount<N>: SupportedLaneCount;
}

// rodio::Sample is f32 since rodio 0.21.0, so we only need to implement Simd Operations for floats.
#[cfg(feature = "simd")]
impl SimdOps for f32
{
    #[inline]
    fn add<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>) -> Simd<f32, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        a + b
    }

    #[inline]
    fn horizontal_add<const N: usize>(a: Simd<f32, N>) -> f32
    where
        LaneCount<N>: SupportedLaneCount,
    {
        a.reduce_sum()
    }
}
