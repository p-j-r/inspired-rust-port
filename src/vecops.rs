//! Level-1 vector kernels (the `Vec*` layer of PETSc).
//!
//! Reductions are evaluated as sums over fixed-size chunks combined in index order
//! (see [`crate::par`]), and the per-chunk arithmetic lives in [`crate::kernels`], so results
//! do not depend on the thread count or on the CPU's vector extensions.

use crate::kernels;
use crate::par::{self, CHUNK};

pub(crate) use crate::kernels::dot as dot_chunk;

/// `aᵀb`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    par::sum_chunks(a.len(), CHUNK, |s, e| kernels::dot(&a[s..e], &b[s..e]))
}

/// Euclidean norm (no over/underflow protection beyond `f64` range; same as PETSc `NORM_2`).
pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `(aᵀb, aᵀa)` in a single pass over the data.
pub fn dot_and_sq(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let r = par::sum_chunks_n(a.len(), CHUNK, |s, e| {
        [kernels::dot(&a[s..e], &b[s..e]), kernels::dot(&a[s..e], &a[s..e])]
    });
    (r[0], r[1])
}

/// `y ← y + αx`.
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    par::for_chunks_mut(y, CHUNK, |s, yc| {
        let xc = &x[s..s + yc.len()];
        kernels::axpy(alpha, xc, yc);
    });
}

/// `y ← x + βy` (PETSc `VecAYPX`).
pub fn aypx(beta: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    par::for_chunks_mut(y, CHUNK, |s, yc| {
        let xc = &x[s..s + yc.len()];
        kernels::aypx(beta, xc, yc);
    });
}

/// `y ← αy`.
pub fn scale(alpha: f64, y: &mut [f64]) {
    par::for_chunks_mut(y, CHUNK, |_, yc| {
        for yi in yc {
            *yi *= alpha;
        }
    });
}

/// `y ← x`.
pub fn copy(x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    par::for_chunks_mut(y, CHUNK, |s, yc| yc.copy_from_slice(&x[s..s + yc.len()]));
}

/// `y ← 0`.
pub fn zero(y: &mut [f64]) {
    par::for_chunks_mut(y, CHUNK, |_, yc| yc.fill(0.0));
}

/// Fused CG update: `x ← x + αp`, `r ← r − αw` in one pass.
pub fn axpy2(alpha: f64, p: &[f64], x: &mut [f64], w: &[f64], r: &mut [f64]) {
    assert!(p.len() == x.len() && w.len() == r.len() && x.len() == r.len());
    par::for_chunks2_mut(x, r, CHUNK, |s, xc, rc| {
        let n = xc.len();
        kernels::axpy2(alpha, &p[s..s + n], xc, &w[s..s + n], rc);
    });
}

/// How a preconditioner acts element-wise, so solvers can fuse it into vector updates.
#[derive(Debug, Clone, Copy)]
pub enum Scaling<'a> {
    /// `z = r`.
    Identity,
    /// `z[i] = d[i] · r[i]`.
    Diagonal(&'a [f64]),
}

/// One fused pass of the CG update with an element-wise preconditioner:
/// `x += αp; r -= αw; z = d ⊙ r` and returns `(zᵀr, zᵀz, rᵀr)` of the new vectors.
/// For [`Scaling::Identity`], `z` is neither read nor written (it aliases `r`).
pub fn cg_update_scaled(
    alpha: f64,
    p: &[f64],
    x: &mut [f64],
    w: &[f64],
    r: &mut [f64],
    z: &mut [f64],
    scaling: Scaling<'_>,
) -> (f64, f64, f64) {
    let n = x.len();
    assert!(p.len() == n && w.len() == n && r.len() == n);
    let (xs, rs, zs) = (par::SharedMut::new(x), par::SharedMut::new(r), par::SharedMut::new(z));
    let out = par::sum_chunks_n::<3>(n, CHUNK, |s, e| {
        // SAFETY: chunk ranges are disjoint and `x`, `r`, `z` are not otherwise touched.
        let (xc, rc) = unsafe { (xs.range(s, e), rs.range(s, e)) };
        match scaling {
            Scaling::Identity => {
                let mut dummy: [f64; 0] = [];
                kernels::cg_update(alpha, &p[s..e], xc, &w[s..e], rc, &mut dummy, None)
            }
            Scaling::Diagonal(d) => {
                let zc = unsafe { zs.range(s, e) };
                kernels::cg_update(alpha, &p[s..e], xc, &w[s..e], rc, zc, Some(&d[s..e]))
            }
        }
    });
    (out[0], out[1], out[2])
}
