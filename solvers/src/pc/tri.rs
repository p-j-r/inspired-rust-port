//! Sparse triangular solves shared by ILU(0), IC(0) and SSOR.
//!
//! A natural-order triangular sweep is a chain of dependent multiply–subtract operations
//! (`z[i]` needs `z[i-1]`), so its speed is set by FP *latency*, not bandwidth. Two
//! reformulations shorten that chain compared with the textbook `z[i] = (r[i] − Σ) / dᵢᵢ`:
//!
//! * diagonal scalings are folded into the stored factors (unit-triangular form), so the only
//!   operations on the chain are the `mul` and `sub` of the last neighbour;
//! * the accumulator starts from `r[i]` and subtracts one product at a time, so the freshest
//!   neighbour (largest column, computed last) joins the chain at the very end.

/// `base − Σ val[k] · z[idx[k]]`, accumulated sequentially in index order.
///
/// # Safety
/// Every `idx[k]` must be `< z.len()`.
#[inline(always)]
unsafe fn row_sub(base: f64, idx: &[u32], val: &[f64], z: &[f64]) -> f64 {
    debug_assert!(idx.iter().all(|&c| (c as usize) < z.len()));
    let mut s = base;
    for (&c, &v) in idx.iter().zip(val) {
        s -= v * unsafe { *z.get_unchecked(c as usize) };
    }
    s
}

/// Like [`row_sub`] but visiting the row from its last entry to its first.
///
/// # Safety
/// Every `idx[k]` must be `< z.len()`.
#[inline(always)]
unsafe fn row_sub_rev(base: f64, idx: &[u32], val: &[f64], z: &[f64]) -> f64 {
    debug_assert!(idx.iter().all(|&c| (c as usize) < z.len()));
    let mut s = base;
    for (&c, &v) in idx.iter().zip(val).rev() {
        s -= v * unsafe { *z.get_unchecked(c as usize) };
    }
    s
}

/// Strict triangle in CSR form (rows `0..n`).
#[derive(Debug, Clone)]
struct Tri {
    ptr: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<f64>,
}

/// `z = U⁻¹ L⁻¹ r` for factors described by the constructor.
///
/// Forward:  `z_i = f_i · r_i − Σ_{j<i} l_ij z_j`   (`f_i = 1` when `fwd_scale` is `None`)
/// Backward: `z_i = b_i · z_i − Σ_{j>i} u_ij z_j`
#[derive(Debug, Clone)]
pub(crate) struct TriSolver {
    n: usize,
    lower: Tri,
    upper: Tri,
    fwd_scale: Option<Vec<f64>>,
    bwd_scale: Vec<f64>,
}

impl TriSolver {
    /// Build from strict-lower and strict-upper CSR triangles of a square `n × n` matrix whose
    /// column indices are all `< n`.
    pub(crate) fn new(
        n: usize,
        lower: (Vec<usize>, Vec<u32>, Vec<f64>),
        upper: (Vec<usize>, Vec<u32>, Vec<f64>),
        fwd_scale: Option<Vec<f64>>,
        bwd_scale: Vec<f64>,
    ) -> Self {
        for (p, i, v) in [&lower, &upper] {
            assert!(p.len() == n + 1 && i.len() == v.len() && p[n] == i.len());
            assert!(i.iter().all(|&c| (c as usize) < n), "column index out of range");
        }
        assert_eq!(bwd_scale.len(), n);
        if let Some(f) = &fwd_scale {
            assert_eq!(f.len(), n);
        }
        Self {
            n,
            lower: Tri { ptr: lower.0, idx: lower.1, val: lower.2 },
            upper: Tri { ptr: upper.0, idx: upper.1, val: upper.2 },
            fwd_scale,
            bwd_scale,
        }
    }

    pub(crate) fn solve(&self, r: &[f64], z: &mut [f64]) {
        let n = self.n;
        assert!(r.len() == n && z.len() == n);
        let l = &self.lower;
        // SAFETY (both sweeps): `new` verified every column index is < n == z.len().
        match &self.fwd_scale {
            None => {
                for i in 0..n {
                    let (lo, hi) = (l.ptr[i], l.ptr[i + 1]);
                    z[i] = unsafe { row_sub(r[i], &l.idx[lo..hi], &l.val[lo..hi], z) };
                }
            }
            Some(f) => {
                for i in 0..n {
                    let (lo, hi) = (l.ptr[i], l.ptr[i + 1]);
                    z[i] = unsafe { row_sub(f[i] * r[i], &l.idx[lo..hi], &l.val[lo..hi], z) };
                }
            }
        }
        let (u, b) = (&self.upper, &self.bwd_scale);
        for i in (0..n).rev() {
            let (lo, hi) = (u.ptr[i], u.ptr[i + 1]);
            // Rows are stored in ascending column order, i.e. nearest neighbour first, but this
            // sweep runs downward: visit the row in reverse so the freshest value joins last.
            z[i] = unsafe { row_sub_rev(b[i] * z[i], &u.idx[lo..hi], &u.val[lo..hi], z) };
        }
    }
}
