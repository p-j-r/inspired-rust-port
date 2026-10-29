//! Fixed-chunk execution helpers.
//!
//! Every reduction in this crate is evaluated as a sum over *fixed-size* chunks
//! combined in index order. The serial and the `parallel` (rayon) paths use the
//! same chunking, so results are bit-for-bit identical for any thread count.

/// Elements per chunk for vector kernels (64 KiB of `f64`, an L2-friendly size).
pub const CHUNK: usize = 8192;

/// Below this length the parallel path is not worth its scheduling overhead.
#[cfg(feature = "parallel")]
const PAR_MIN: usize = 32 * 1024;

#[inline]
pub fn parallel_enabled(n: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        n >= PAR_MIN && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        false
    }
}

/// Sum of `f(start, end)` over the chunks of `0..n`, accumulated in chunk order.
#[inline]
pub fn sum_chunks(n: usize, chunk: usize, f: impl Fn(usize, usize) -> f64 + Sync) -> f64 {
    #[cfg(feature = "parallel")]
    if parallel_enabled(n) {
        use rayon::prelude::*;
        let nchunks = n.div_ceil(chunk);
        let parts: Vec<f64> = (0..nchunks)
            .into_par_iter()
            .map(|c| f(c * chunk, ((c + 1) * chunk).min(n)))
            .collect();
        return parts.iter().sum();
    }
    let mut acc = 0.0;
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        acc += f(s, e);
        s = e;
    }
    acc
}

/// Like [`sum_chunks`] for `N` simultaneous scalar reductions.
#[inline]
pub fn sum_chunks_n<const N: usize>(
    n: usize,
    chunk: usize,
    f: impl Fn(usize, usize) -> [f64; N] + Sync,
) -> [f64; N] {
    #[cfg(feature = "parallel")]
    if parallel_enabled(n) {
        use rayon::prelude::*;
        let nchunks = n.div_ceil(chunk);
        let parts: Vec<[f64; N]> = (0..nchunks)
            .into_par_iter()
            .map(|c| f(c * chunk, ((c + 1) * chunk).min(n)))
            .collect();
        let mut acc = [0.0; N];
        for p in parts {
            for i in 0..N {
                acc[i] += p[i];
            }
        }
        return acc;
    }
    let mut acc = [0.0; N];
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        let p = f(s, e);
        for i in 0..N {
            acc[i] += p[i];
        }
        s = e;
    }
    acc
}

/// A mutable slice that can be shared across chunk closures which each touch a *disjoint*
/// range. This is what lets fused kernels update several vectors and return reductions
/// in a single pass over [`sum_chunks_n`].
pub struct SharedMut {
    ptr: *mut f64,
    len: usize,
}

// SAFETY: access goes through `range`, whose contract requires disjoint ranges per thread.
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}

impl SharedMut {
    pub fn new(s: &mut [f64]) -> Self {
        Self { ptr: s.as_mut_ptr(), len: s.len() }
    }

    /// # Safety
    /// No two live slices obtained from the same `SharedMut` may overlap, and the original
    /// slice must not be used while they are alive.
    #[inline(always)]
    pub unsafe fn range(&self, start: usize, end: usize) -> &mut [f64] {
        assert!(start <= end && end <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), end - start) }
    }
}

/// Per-chunk vector-valued results (used for `k`-vector reductions such as `Vᵀw`),
/// returned in chunk order for the caller to fold deterministically.
#[inline]
pub fn map_chunks<R: Send>(n: usize, chunk: usize, f: impl Fn(usize, usize) -> R + Sync) -> Vec<R> {
    #[cfg(feature = "parallel")]
    if parallel_enabled(n) {
        use rayon::prelude::*;
        let nchunks = n.div_ceil(chunk);
        return (0..nchunks)
            .into_par_iter()
            .map(|c| f(c * chunk, ((c + 1) * chunk).min(n)))
            .collect();
    }
    let mut out = Vec::with_capacity(n.div_ceil(chunk));
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        out.push(f(s, e));
        s = e;
    }
    out
}

/// Apply `f(start, y_chunk)` to disjoint chunks of `y`.
#[inline]
pub fn for_chunks_mut(y: &mut [f64], chunk: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    #[cfg(feature = "parallel")]
    if parallel_enabled(y.len()) {
        use rayon::prelude::*;
        y.par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| f(c * chunk, yc));
        return;
    }
    for (c, yc) in y.chunks_mut(chunk).enumerate() {
        f(c * chunk, yc);
    }
}

/// Apply `f(start, a_chunk, b_chunk)` to matching disjoint chunks of two slices.
#[inline]
pub fn for_chunks2_mut(
    a: &mut [f64],
    b: &mut [f64],
    chunk: usize,
    f: impl Fn(usize, &mut [f64], &mut [f64]) + Sync,
) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "parallel")]
    if parallel_enabled(a.len()) {
        use rayon::prelude::*;
        a.par_chunks_mut(chunk)
            .zip(b.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(c, (ac, bc))| f(c * chunk, ac, bc));
        return;
    }
    for (c, (ac, bc)) in a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate() {
        f(c * chunk, ac, bc);
    }
}

/// Accumulate `k`-vector partials: `out[j] += Σ_chunks part_c[j]`, with partials folded in
/// chunk order (`f(start, end, part)` fills a zeroed `part` of length `k`).
#[inline]
pub fn fold_chunks_vec(
    n: usize,
    chunk: usize,
    out: &mut [f64],
    f: impl Fn(usize, usize, &mut [f64]) + Sync,
) {
    let k = out.len();
    #[cfg(feature = "parallel")]
    if parallel_enabled(n) {
        use rayon::prelude::*;
        let nchunks = n.div_ceil(chunk);
        let parts: Vec<Vec<f64>> = (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let mut part = vec![0.0; k];
                f(c * chunk, ((c + 1) * chunk).min(n), &mut part);
                part
            })
            .collect();
        for part in &parts {
            for (o, p) in out.iter_mut().zip(part) {
                *o += p;
            }
        }
        return;
    }
    let mut part = vec![0.0; k];
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        part.fill(0.0);
        f(s, e, &mut part);
        for (o, p) in out.iter_mut().zip(&part) {
            *o += p;
        }
        s = e;
    }
}
