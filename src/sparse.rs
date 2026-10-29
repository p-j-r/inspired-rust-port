//! Compressed sparse row matrix (PETSc `MATSEQAIJ`) and test-problem generators.

use crate::operator::LinearOperator;
use crate::par;
use crate::rng::SplitMix64;
use std::fmt;

/// Rows per parallel/fused-reduction chunk in the SpMV kernels.
const ROW_CHUNK: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseError {
    BadIndptr(&'static str),
    LengthMismatch,
    ColumnOutOfRange { row: usize, col: usize },
    DimensionTooLarge,
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseError::BadIndptr(m) => write!(f, "invalid indptr: {m}"),
            SparseError::LengthMismatch => write!(f, "indices/values length does not match indptr"),
            SparseError::ColumnOutOfRange { row, col } => {
                write!(f, "column {col} out of range in row {row}")
            }
            SparseError::DimensionTooLarge => write!(f, "dimension exceeds u32 index range"),
        }
    }
}

impl std::error::Error for SparseError {}

/// CSR matrix with `u32` column indices (half the index bandwidth of `usize`).
///
/// The invariants checked in [`try_new`](Self::try_new) (monotone `indptr`, every column
/// index `< ncols`) are what make the unchecked indexing in the kernels sound; the fields
/// are private so they cannot be broken afterwards.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<f64>,
}

impl CsrMatrix {
    pub fn try_new(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        values: Vec<f64>,
    ) -> Result<Self, SparseError> {
        if ncols > u32::MAX as usize {
            return Err(SparseError::DimensionTooLarge);
        }
        if indptr.len() != nrows + 1 {
            return Err(SparseError::BadIndptr("length must be nrows + 1"));
        }
        if indptr[0] != 0 {
            return Err(SparseError::BadIndptr("must start at 0"));
        }
        if indptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(SparseError::BadIndptr("must be non-decreasing"));
        }
        let nnz = indptr[nrows];
        if indices.len() != nnz || values.len() != nnz {
            return Err(SparseError::LengthMismatch);
        }
        for row in 0..nrows {
            for &c in &indices[indptr[row]..indptr[row + 1]] {
                if c as usize >= ncols {
                    return Err(SparseError::ColumnOutOfRange { row, col: c as usize });
                }
            }
        }
        Ok(Self { nrows, ncols, indptr, indices, values })
    }

    /// Build from `(row, col, value)` triplets; duplicates are summed and columns sorted.
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Result<Self, SparseError> {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); nrows];
        for &(r, c, v) in triplets {
            if r >= nrows {
                return Err(SparseError::BadIndptr("row index out of range"));
            }
            if c >= ncols {
                return Err(SparseError::ColumnOutOfRange { row: r, col: c });
            }
            rows[r].push((c as u32, v));
        }
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0);
        for row in &mut rows {
            row.sort_by_key(|&(c, _)| c);
            let mut last: Option<u32> = None;
            for &(c, v) in row.iter() {
                if last == Some(c) {
                    *values.last_mut().unwrap() += v;
                } else {
                    indices.push(c);
                    values.push(v);
                    last = Some(c);
                }
            }
            indptr.push(indices.len());
        }
        Self::try_new(nrows, ncols, indptr, indices, values)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// `(columns, values)` of one row.
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let r = self.indptr[i]..self.indptr[i + 1];
        (&self.indices[r.clone()], &self.values[r])
    }

    /// Columns are strictly increasing within every row.
    pub fn has_sorted_unique_rows(&self) -> bool {
        (0..self.nrows).all(|i| self.row(i).0.windows(2).all(|w| w[0] < w[1]))
    }

    /// Diagonal entries (zero where structurally absent).
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.nrows.min(self.ncols);
        (0..n)
            .map(|i| {
                let (c, v) = self.row(i);
                c.iter().zip(v).filter(|&(&c, _)| c as usize == i).map(|(_, &v)| v).sum()
            })
            .collect()
    }

    pub fn transpose(&self) -> CsrMatrix {
        let mut count = vec![0usize; self.ncols + 1];
        for &c in &self.indices {
            count[c as usize + 1] += 1;
        }
        for i in 0..self.ncols {
            count[i + 1] += count[i];
        }
        let mut next = count.clone();
        let mut indices = vec![0u32; self.nnz()];
        let mut values = vec![0.0; self.nnz()];
        for i in 0..self.nrows {
            let (cols, vals) = self.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                let p = next[c as usize];
                next[c as usize] += 1;
                indices[p] = i as u32;
                values[p] = v;
            }
        }
        CsrMatrix { nrows: self.ncols, ncols: self.nrows, indptr: count, indices, values }
    }

    /// Exact structural + numerical symmetry up to `tol * max|a|`.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        let t = self.transpose();
        if t.indptr != self.indptr || t.indices != self.indices {
            return false;
        }
        let scale = self.values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        self.values.iter().zip(&t.values).all(|(a, b)| (a - b).abs() <= tol * scale)
    }

    /// Dense row-major copy (tests and tiny problems only).
    pub fn to_dense(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.nrows * self.ncols];
        for i in 0..self.nrows {
            let (c, v) = self.row(i);
            for (&c, &v) in c.iter().zip(v) {
                d[i * self.ncols + c as usize] += v;
            }
        }
        d
    }

    #[inline(always)]
    fn row_dot(&self, i: usize, x: &[f64]) -> f64 {
        let (lo, hi) = (self.indptr[i], self.indptr[i + 1]);
        let idx = &self.indices[lo..hi];
        let val = &self.values[lo..hi];
        let mut s = 0.0;
        for (&c, &v) in idx.iter().zip(val) {
            // SAFETY: `try_new` guarantees every stored column index is `< ncols`,
            // and callers assert `x.len() == ncols`.
            s += v * unsafe { *x.get_unchecked(c as usize) };
        }
        s
    }

    /// `y ← A x`.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        if par::parallel_enabled(self.nrows) {
            par::for_chunks_mut(y, ROW_CHUNK, |s, yc| {
                for (k, yi) in yc.iter_mut().enumerate() {
                    *yi = self.row_dot(s + k, x);
                }
            });
        } else {
            for (i, yi) in y.iter_mut().enumerate() {
                *yi = self.row_dot(i, x);
            }
        }
    }

    /// `y ← A x`, returning `xᵀy` (square matrices), fusing the reduction into the
    /// product so `y` is not re-read.
    pub fn spmv_dot(&self, x: &[f64], y: &mut [f64]) -> f64 {
        assert_eq!(self.nrows, self.ncols);
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        let n = self.nrows;
        if par::parallel_enabled(n) {
            use std::sync::atomic::{AtomicPtr, Ordering};
            // Disjoint row chunks write disjoint ranges of `y`; share the base pointer.
            let base = AtomicPtr::new(y.as_mut_ptr());
            par::sum_chunks(n, ROW_CHUNK, |s, e| {
                // SAFETY: chunks `[s, e)` are disjoint and within `y`.
                let yc = unsafe {
                    std::slice::from_raw_parts_mut(base.load(Ordering::Relaxed).add(s), e - s)
                };
                self.chunk_spmv_dot(s, x, yc)
            })
        } else {
            let mut acc = 0.0;
            let mut s = 0;
            while s < n {
                let e = (s + ROW_CHUNK).min(n);
                acc += self.chunk_spmv_dot(s, x, &mut y[s..e]);
                s = e;
            }
            acc
        }
    }

    #[inline(always)]
    fn chunk_spmv_dot(&self, start: usize, x: &[f64], yc: &mut [f64]) -> f64 {
        let mut acc = 0.0;
        for (k, yi) in yc.iter_mut().enumerate() {
            let v = self.row_dot(start + k, x);
            *yi = v;
            acc += v * x[start + k];
        }
        acc
    }
}

impl LinearOperator for CsrMatrix {
    fn dim(&self) -> usize {
        assert_eq!(self.nrows, self.ncols, "operator must be square");
        self.nrows
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.spmv(x, y)
    }
    fn apply_dot(&self, x: &[f64], y: &mut [f64]) -> f64 {
        self.spmv_dot(x, y)
    }
}

// ---------------------------------------------------------------------------------------
// Test problems
// ---------------------------------------------------------------------------------------

/// 1-D Laplacian `tridiag(-1, 2, -1)` of order `n`.
pub fn laplacian_1d(n: usize) -> CsrMatrix {
    let mut t = Vec::with_capacity(3 * n);
    for i in 0..n {
        if i > 0 {
            t.push((i, i - 1, -1.0));
        }
        t.push((i, i, 2.0));
        if i + 1 < n {
            t.push((i, i + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, &t).unwrap()
}

/// 2-D 5-point Laplacian on an `nx × ny` grid (Dirichlet, natural ordering, `i + nx*j`).
pub fn laplacian_2d(nx: usize, ny: usize) -> CsrMatrix {
    stencil_2d(nx, ny, 0.0, 0.0)
}

/// 2-D convection–diffusion `-Δu + (bx, by)·∇u` with first-order upwind convection
/// (grid spacing absorbed: `bx`, `by` are cell Péclet numbers). Non-symmetric for `b ≠ 0`.
pub fn convection_diffusion_2d(nx: usize, ny: usize, bx: f64, by: f64) -> CsrMatrix {
    stencil_2d(nx, ny, bx, by)
}

fn stencil_2d(nx: usize, ny: usize, bx: f64, by: f64) -> CsrMatrix {
    let n = nx * ny;
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(5 * n);
    let mut values = Vec::with_capacity(5 * n);
    indptr.push(0);
    // Upwind: for b > 0 the neighbour behind (i-1) gets -(1+b), diagonal gains b.
    let (wx, ex, dx) = (-(1.0 + bx.max(0.0)), -(1.0 + (-bx).max(0.0)), bx.abs());
    let (sy, ny_, dy) = (-(1.0 + by.max(0.0)), -(1.0 + (-by).max(0.0)), by.abs());
    for j in 0..ny {
        for i in 0..nx {
            let p = i + nx * j;
            if j > 0 {
                indices.push((p - nx) as u32);
                values.push(sy);
            }
            if i > 0 {
                indices.push((p - 1) as u32);
                values.push(wx);
            }
            indices.push(p as u32);
            values.push(4.0 + dx + dy);
            if i + 1 < nx {
                indices.push((p + 1) as u32);
                values.push(ex);
            }
            if j + 1 < ny {
                indices.push((p + nx) as u32);
                values.push(ny_);
            }
            indptr.push(indices.len());
        }
    }
    CsrMatrix::try_new(n, n, indptr, indices, values).unwrap()
}

/// 3-D 7-point Laplacian on an `nx × ny × nz` grid (index `i + nx*(j + ny*k)`).
pub fn laplacian_3d(nx: usize, ny: usize, nz: usize) -> CsrMatrix {
    let n = nx * ny * nz;
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(7 * n);
    let mut values = Vec::with_capacity(7 * n);
    indptr.push(0);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = i + nx * (j + ny * k);
                let mut push = |q: usize, v: f64| {
                    indices.push(q as u32);
                    values.push(v);
                };
                if k > 0 {
                    push(p - nx * ny, -1.0);
                }
                if j > 0 {
                    push(p - nx, -1.0);
                }
                if i > 0 {
                    push(p - 1, -1.0);
                }
                push(p, 6.0);
                if i + 1 < nx {
                    push(p + 1, -1.0);
                }
                if j + 1 < ny {
                    push(p + nx, -1.0);
                }
                if k + 1 < nz {
                    push(p + nx * ny, -1.0);
                }
                indptr.push(indices.len());
            }
        }
    }
    CsrMatrix::try_new(n, n, indptr, indices, values).unwrap()
}

/// Random sparse symmetric matrix with `~per_row` off-diagonals per row and a diagonal that
/// makes it strictly diagonally dominant (hence SPD). Deterministic in `seed`.
pub fn random_spd(n: usize, per_row: usize, seed: u64) -> CsrMatrix {
    let mut rng = SplitMix64::new(seed);
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    let mut rowsum = vec![0.0f64; n];
    for i in 0..n {
        for _ in 0..per_row / 2 {
            let j = rng.below(n);
            if j == i {
                continue;
            }
            let v = -rng.next_f64() - 0.1;
            t.push((i, j, v));
            t.push((j, i, v));
            rowsum[i] += v.abs();
            rowsum[j] += v.abs();
        }
    }
    for (i, s) in rowsum.iter().enumerate() {
        t.push((i, i, s + 1.0 + rng.next_f64()));
    }
    CsrMatrix::from_triplets(n, n, &t).unwrap()
}

/// Random sparse non-symmetric, diagonally dominant matrix (well-conditioned for GMRES).
pub fn random_nonsymmetric(n: usize, per_row: usize, seed: u64) -> CsrMatrix {
    let mut rng = SplitMix64::new(seed);
    let mut t: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        let mut s = 0.0;
        for _ in 0..per_row {
            let j = rng.below(n);
            if j == i {
                continue;
            }
            let v = rng.next_signed();
            t.push((i, j, v));
            s += v.abs();
        }
        t.push((i, i, s * 1.1 + 1.0));
    }
    CsrMatrix::from_triplets(n, n, &t).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_structure() {
        assert!(CsrMatrix::try_new(2, 2, vec![0, 1], vec![0], vec![1.0]).is_err());
        assert!(CsrMatrix::try_new(1, 2, vec![0, 1], vec![2], vec![1.0]).is_err());
        assert!(CsrMatrix::try_new(2, 2, vec![0, 2, 1], vec![0, 1], vec![1.0, 1.0]).is_err());
        assert!(CsrMatrix::try_new(1, 1, vec![0, 1], vec![0, 0], vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn triplets_sum_duplicates_and_sort() {
        let a = CsrMatrix::from_triplets(2, 3, &[(0, 2, 1.0), (0, 0, 2.0), (0, 2, 3.0), (1, 1, 5.0)]).unwrap();
        assert_eq!(a.row(0).0, &[0, 2]);
        assert_eq!(a.row(0).1, &[2.0, 4.0]);
        assert_eq!(a.to_dense(), vec![2.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
    }

    #[test]
    fn spmv_matches_dense() {
        let a = random_nonsymmetric(50, 6, 3);
        let x = crate::rng::random_vector(50, 9);
        let mut y = vec![0.0; 50];
        a.spmv(&x, &mut y);
        let d = a.to_dense();
        for i in 0..50 {
            let r: f64 = (0..50).map(|j| d[i * 50 + j] * x[j]).sum();
            assert!((r - y[i]).abs() < 1e-12);
        }
        let mut y2 = vec![0.0; 50];
        let xy = a.spmv_dot(&x, &mut y2);
        assert_eq!(y, y2);
        assert!((xy - crate::vecops::dot(&x, &y)).abs() < 1e-12);
    }

    #[test]
    fn transpose_and_symmetry() {
        let a = laplacian_2d(7, 5);
        assert!(a.is_symmetric(0.0));
        assert_eq!(a.transpose(), a);
        let c = convection_diffusion_2d(7, 5, 1.0, 0.5);
        assert!(!c.is_symmetric(1e-12));
        assert_eq!(c.transpose().transpose(), c);
    }

    #[test]
    fn laplacian_row_sums() {
        // Interior rows of the Dirichlet Laplacians sum to zero; boundary rows are positive.
        let a = laplacian_3d(4, 4, 4);
        let ones = vec![1.0; 64];
        let mut y = vec![0.0; 64];
        a.spmv(&ones, &mut y);
        assert_eq!(y[1 + 4 * (1 + 4 * 1)], 0.0);
        assert!(y[0] > 0.0);
        assert_eq!(a.nnz(), 7 * 64 - 6 * 16);
    }
}

impl CsrMatrix {
    /// Column-major dense copy, for tests of dense routines.
    #[doc(hidden)]
    pub fn to_dense_colmajor_for_test(&self) -> Vec<f64> {
        let d = self.to_dense();
        let (m, n) = (self.nrows, self.ncols);
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                c[j * m + i] = d[i * n + j];
            }
        }
        c
    }
}
