#[cfg(feature="profiler")]
use time_graph::instrument;

#[cfg(feature="simd")]
use std::simd::{Simd, Mask, SimdElement, LaneCount, SupportedLaneCount};

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
    #[inline]
    pub fn from_slice_or(src: &'a [T], or: Simd<T, N>) -> SimdIter<'a, T, N> {
        Self {
            src: src,
            or: or,
            i: 0,
        }
    }

    #[inline]
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

            Some(
                (Simd::load_or(&self.src[start..end], self.or), Mask::splat(true))
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
