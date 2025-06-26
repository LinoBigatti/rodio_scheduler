use time_graph::instrument;

#[inline]
#[instrument]
pub fn simd_retrieve_samples<'a, D: rodio::Sample>(source: &'a [D], sample_n: u64, playback_queue: &[u64]) -> Vec<D> {
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
#[instrument]
pub fn simd_mix_samples<D: rodio::Sample>(samples: &[D], input_sample: Option<D>) -> Option<D> {
    samples.into_iter().fold(input_sample, |accumulator, sample| match accumulator {
        Some(s1) => Some(s1.saturating_add(*sample)),
        // If you want to make scheduled playback stop after the input Source ended, return None here
        None => Some(*sample),
    })
}
