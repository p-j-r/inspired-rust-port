//! Level-1 vector kernels (the `Vec*` layer of PETSc).
//!
//! Reductions use eight independent accumulators per chunk so the compiler can
//! vectorise them without fast-math, and chunk partials are summed in order
//! (see [`crate::par`]) so results do not depend on the thread count.

use crate::par::{self, CHUNK};

#[inline(always)]
pub(crate) fn dot_chunk(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// `aᵀb`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    par::sum_chunks(a.len(), CHUNK, |s, e| dot_chunk(&a[s..e], &b[s..e]))
}

/// Euclidean norm (no over/underflow protection beyond `f64` range; same as PETSc `NORM_2`).
pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `(aᵀb, aᵀa)` in a single pass over the data.
pub fn dot_and_sq(a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let r = par::sum_chunks2(a.len(), CHUNK, |s, e| {
        [dot_chunk(&a[s..e], &b[s..e]), dot_chunk(&a[s..e], &a[s..e])]
    });
    (r[0], r[1])
}

/// `y ← y + αx`.
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    par::for_chunks_mut(y, CHUNK, |s, yc| {
        let xc = &x[s..s + yc.len()];
        for (yi, xi) in yc.iter_mut().zip(xc) {
            *yi += alpha * *xi;
        }
    });
}

/// `y ← x + βy` (PETSc `VecAYPX`).
pub fn aypx(beta: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    par::for_chunks_mut(y, CHUNK, |s, yc| {
        let xc = &x[s..s + yc.len()];
        for (yi, xi) in yc.iter_mut().zip(xc) {
            *yi = *xi + beta * *yi;
        }
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
        let (pc, wc) = (&p[s..s + n], &w[s..s + n]);
        for i in 0..n {
            xc[i] += alpha * pc[i];
            rc[i] -= alpha * wc[i];
        }
    });
}
