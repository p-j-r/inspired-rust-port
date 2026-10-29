//! Column-major block of `n`-vectors with cache-blocked multi-vector kernels
//! (the role of SLEPc's `BV` object and PETSc's `VecMDot` / `VecMAXPY`).
//!
//! All row loops are blocked in [`CHUNK`]-sized pieces so the vector being
//! orthogonalised stays in L2 while every basis column streams through once.

use crate::par::{self, CHUNK};
use crate::vecops::dot_chunk;

/// `cap` columns of length `n`, stored contiguously column after column.
#[derive(Debug, Clone)]
pub struct Basis {
    n: usize,
    cap: usize,
    data: Vec<f64>,
}

impl Basis {
    pub fn new(n: usize, cap: usize) -> Self {
        Self { n, cap, data: vec![0.0; n * cap] }
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.n..(j + 1) * self.n]
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.n..(j + 1) * self.n]
    }

    /// Two distinct columns, `a` shared and `b` mutable.
    pub fn col_pair(&mut self, a: usize, b: usize) -> (&[f64], &mut [f64]) {
        assert_ne!(a, b);
        let n = self.n;
        if a < b {
            let (lo, hi) = self.data.split_at_mut(b * n);
            (&lo[a * n..(a + 1) * n], &mut hi[..n])
        } else {
            let (lo, hi) = self.data.split_at_mut(a * n);
            (&hi[..n], &mut lo[b * n..(b + 1) * n])
        }
    }

    pub fn copy_col(&mut self, from: usize, to: usize) {
        if from != to {
            let (src, dst) = self.col_pair(from, to);
            dst.copy_from_slice(src);
        }
    }

    /// Read-only view of all columns.
    pub fn view(&self) -> BasisView<'_> {
        BasisView { n: self.n, data: &self.data }
    }

    /// Columns `0..j` as a read-only view, plus column `j` mutably (for orthogonalising
    /// a freshly computed vector against its predecessors without copying).
    pub fn split_col(&mut self, j: usize) -> (BasisView<'_>, &mut [f64]) {
        let n = self.n;
        let (head, tail) = self.data.split_at_mut(j * n);
        (BasisView { n, data: head }, &mut tail[..n])
    }

    /// See [`BasisView::dots`].
    pub fn dots(&self, k: usize, w: &[f64], h: &mut [f64]) {
        self.view().dots(k, w, h)
    }
    /// See [`BasisView::sub_combination`].
    pub fn sub_combination(&self, k: usize, h: &[f64], w: &mut [f64]) {
        self.view().sub_combination(k, h, w)
    }
    /// See [`BasisView::add_combination`].
    pub fn add_combination(&self, k: usize, c: &[f64], y: &mut [f64]) {
        self.view().add_combination(k, c, y)
    }

    /// `V[:, first..first+q.ncols] ← V[:, first..first+q.nrows] · Q` for a column-major
    /// `Q` with `ldq` rows (`BVMultInPlace`). Rows are processed in cache-sized blocks with a
    /// small scratch tile so the multiply is done in a single pass over the basis.
    pub fn mult_in_place(&mut self, first: usize, nrows_q: usize, ncols_q: usize, q: &[f64], ldq: usize) {
        assert!(first + nrows_q.max(ncols_q) <= self.cap);
        assert!(q.len() >= ldq * ncols_q && ldq >= nrows_q);
        let n = self.n;
        const BLOCK: usize = 1024;
        // Blocks are independent, so this parallelises by splitting rows; columns are strided
        // in memory, so use raw access through a shared base pointer for the disjoint row ranges.
        let base = SharedPtr(self.data.as_mut_ptr());
        let nblocks = n.div_ceil(BLOCK);
        let work = |blk: usize| {
            let s = blk * BLOCK;
            let e = (s + BLOCK).min(n);
            let len = e - s;
            let mut tile = vec![0.0f64; len * ncols_q];
            for jc in 0..ncols_q {
                let out = &mut tile[jc * len..(jc + 1) * len];
                for jr in 0..nrows_q {
                    let qv = q[jc * ldq + jr];
                    // SAFETY: read-only view of rows s..e of column first+jr; no other
                    // block touches these rows.
                    let src = unsafe {
                        std::slice::from_raw_parts(base.get().add((first + jr) * n + s), len)
                    };
                    for (o, v) in out.iter_mut().zip(src) {
                        *o += qv * v;
                    }
                }
            }
            for jc in 0..ncols_q {
                // SAFETY: rows s..e of the destination columns belong to this block only.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(base.get().add((first + jc) * n + s), len)
                };
                dst.copy_from_slice(&tile[jc * len..(jc + 1) * len]);
            }
        };
        #[cfg(feature = "parallel")]
        if par::parallel_enabled(n) {
            use rayon::prelude::*;
            (0..nblocks).into_par_iter().for_each(work);
            return;
        }
        (0..nblocks).for_each(work);
    }
}

/// Borrowed leading columns of a [`Basis`] with the read-only multi-vector kernels.
#[derive(Clone, Copy)]
pub struct BasisView<'a> {
    n: usize,
    data: &'a [f64],
}

impl BasisView<'_> {
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.n..(j + 1) * self.n]
    }

    /// `h[j] = V[:, j]ᵀ w` for `j < k` (`VecMDot`).
    pub fn dots(&self, k: usize, w: &[f64], h: &mut [f64]) {
        let n = self.n;
        assert!((k * n) <= self.data.len() && w.len() == n && h.len() >= k);
        let h = &mut h[..k];
        h.fill(0.0);
        par::fold_chunks_vec(n, CHUNK, h, |s, e, part| {
            let wc = &w[s..e];
            for (j, p) in part.iter_mut().enumerate() {
                *p = dot_chunk(&self.data[j * n + s..j * n + e], wc);
            }
        });
    }

    /// `w ← w − V[:, :k] h` (`VecMAXPY` with negated coefficients).
    pub fn sub_combination(&self, k: usize, h: &[f64], w: &mut [f64]) {
        let n = self.n;
        assert!((k * n) <= self.data.len() && w.len() == n && h.len() >= k);
        par::for_chunks_mut(w, CHUNK, |s, wc| {
            let len = wc.len();
            for j in 0..k {
                let col = &self.data[j * n + s..j * n + s + len];
                let hj = h[j];
                for (wi, vi) in wc.iter_mut().zip(col) {
                    *wi -= hj * vi;
                }
            }
        });
    }

    /// `y ← y + V[:, :k] c`.
    pub fn add_combination(&self, k: usize, c: &[f64], y: &mut [f64]) {
        let n = self.n;
        assert!((k * n) <= self.data.len() && y.len() == n && c.len() >= k);
        par::for_chunks_mut(y, CHUNK, |s, yc| {
            let len = yc.len();
            for j in 0..k {
                let col = &self.data[j * n + s..j * n + s + len];
                let cj = c[j];
                for (yi, vi) in yc.iter_mut().zip(col) {
                    *yi += cj * vi;
                }
            }
        });
    }
}

#[derive(Clone, Copy)]
struct SharedPtr(*mut f64);
// SAFETY: used only for disjoint row blocks, see `mult_in_place`.
unsafe impl Send for SharedPtr {}
unsafe impl Sync for SharedPtr {}
impl SharedPtr {
    #[inline]
    fn get(self) -> *mut f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::random_vector;

    fn filled(n: usize, k: usize) -> Basis {
        let mut b = Basis::new(n, k);
        for j in 0..k {
            b.col_mut(j).copy_from_slice(&random_vector(n, j as u64 + 1));
        }
        b
    }

    #[test]
    fn dots_and_sub_match_naive() {
        let (n, k) = (20_001, 7);
        let b = filled(n, k);
        let w = random_vector(n, 99);
        let mut h = vec![0.0; k];
        b.dots(k, &w, &mut h);
        for j in 0..k {
            let r: f64 = b.col(j).iter().zip(&w).map(|(a, b)| a * b).sum();
            assert!((r - h[j]).abs() < 1e-9 * r.abs().max(1.0));
        }
        let mut w2 = w.clone();
        b.sub_combination(k, &h, &mut w2);
        for i in (0..n).step_by(997) {
            let r = w[i] - (0..k).map(|j| h[j] * b.col(j)[i]).sum::<f64>();
            assert!((r - w2[i]).abs() < 1e-12);
        }
        let mut y = w2.clone();
        b.add_combination(k, &h, &mut y);
        for i in (0..n).step_by(991) {
            assert!((y[i] - w[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn mult_in_place_matches_naive() {
        let (n, k) = (3000, 6);
        let mut b = filled(n, 8);
        let orig = b.clone();
        // Q is 6 x 4 (ldq = 6), applied to columns 1..7 giving columns 1..5.
        let q: Vec<f64> = (0..24).map(|i| (i as f64 * 0.37).sin()).collect();
        b.mult_in_place(1, k, 4, &q, 6);
        for jc in 0..4 {
            for i in (0..n).step_by(131) {
                let r: f64 = (0..k).map(|jr| q[jc * 6 + jr] * orig.col(1 + jr)[i]).sum();
                assert!((r - b.col(1 + jc)[i]).abs() < 1e-12);
            }
        }
        // Untouched columns stay untouched.
        assert_eq!(b.col(0), orig.col(0));
        assert_eq!(b.col(7), orig.col(7));
    }
}
