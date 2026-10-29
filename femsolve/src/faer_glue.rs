//! Everything that touches `faer`'s dense API.
//!
//! nyx pins its own faer version; if it differs from the one in this crate's `Cargo.toml`, the
//! two places to adapt are this file and `direct.rs` (sparse factorisations). The rest of the
//! crate only sees [`FaerScalar`].

use crate::dense::{Dense, DenseError};
use crate::scalar::{C64, Scalar};
use faer::traits::ComplexField;
use faer::{Mat, MatMut, Par, Side};

/// Scalars that `faer` can handle, with the conversions to its element type (identical
/// layouts: `f64` and `Complex<f64>` are faer's own `f64` and `c64`).
pub trait FaerScalar: Scalar {
    type Elem: ComplexField<Real = f64> + Copy;
    fn to_faer(self) -> Self::Elem;
    fn from_faer(x: Self::Elem) -> Self;
    /// Identity reinterpretations of slices (implemented per type, so no `unsafe` is needed).
    fn faer_slice(s: &[Self]) -> &[Self::Elem];
    fn faer_slice_mut(s: &mut [Self]) -> &mut [Self::Elem];
}

impl FaerScalar for f64 {
    type Elem = f64;
    #[inline]
    fn to_faer(self) -> f64 {
        self
    }
    #[inline]
    fn from_faer(x: f64) -> f64 {
        x
    }
    #[inline]
    fn faer_slice(s: &[f64]) -> &[f64] {
        s
    }
    #[inline]
    fn faer_slice_mut(s: &mut [f64]) -> &mut [f64] {
        s
    }
}

impl FaerScalar for C64 {
    type Elem = faer::c64;
    #[inline]
    fn to_faer(self) -> faer::c64 {
        self
    }
    #[inline]
    fn from_faer(x: faer::c64) -> C64 {
        x
    }
    #[inline]
    fn faer_slice(s: &[C64]) -> &[faer::c64] {
        s
    }
    #[inline]
    fn faer_slice_mut(s: &mut [C64]) -> &mut [faer::c64] {
        s
    }
}

pub(crate) fn hermitian_eig<T: FaerScalar>(a: &Dense<T>) -> Result<(Vec<f64>, Dense<T>), DenseError> {
    assert_eq!(a.rows(), a.cols());
    let n = a.rows();
    if !a.is_finite() {
        return Err(DenseError::NonFinite);
    }
    if n == 0 {
        return Ok((Vec::new(), Dense::zeros(0, 0)));
    }
    // Mirror the lower triangle so the input is exactly Hermitian whatever the caller stored.
    let m = Mat::<T::Elem>::from_fn(n, n, |i, j| if i >= j { a[(i, j)].to_faer() } else { a[(j, i)].conj().to_faer() });
    let evd = m.self_adjoint_eigen(Side::Lower).map_err(|_| DenseError::NoConvergence);
    clear_upper_vector_state();
    let evd = evd?;
    let (u, s) = (evd.U(), evd.S());
    let w: Vec<f64> = (0..n).map(|i| Scalar::re(T::from_faer(s.column_vector()[i]))).collect();
    let q = Dense::from_fn(n, n, |i, j| T::from_faer(u[(i, j)]));
    Ok((w, q))
}

/// `out = src · q` on one row tile with `faer`'s matrix product, sequential (one tile is one
/// thread's unit of work). `src` is `len × nrows` and `out` is `len × ncols`, contiguous;
/// `q` is `nrows × ncols` with leading dimension `ldq`.
#[cfg(not(feature = "portable-kernels"))]
pub(crate) fn tile_mult<T: FaerScalar>(
    src: &[T],
    len: usize,
    nrows: usize,
    q: &[T],
    ldq: usize,
    ncols: usize,
    out: &mut [T],
) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatRef};
    let lhs = MatRef::from_column_major_slice(T::faer_slice(&src[..len * nrows]), len, nrows);
    let rhs =
        MatRef::from_column_major_slice_with_stride(T::faer_slice(&q[..ldq * (ncols - 1) + nrows]), nrows, ncols, ldq);
    let dst = MatMut::from_column_major_slice_mut(T::faer_slice_mut(&mut out[..len * ncols]), len, ncols);
    matmul(dst, Accum::Replace, lhs, rhs, <T::Elem as faer::traits::ComplexField>::one_impl(), Par::Seq);
    clear_upper_vector_state();
}

/// Executes `vzeroupper` where available. On the benchmark VM (Cascade Lake class), one small
/// faer GEMM made *all later* baseline-SSE code about 3.6x slower (see
/// `examples/gemm_side_effect.rs`); the cause is dirty upper vector state left by wide kernels
/// without `vzeroupper`. Called after every faer call in this crate; it costs about one cycle.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn clear_upper_vector_state() {
    #[target_feature(enable = "avx")]
    fn zeroupper() {
        std::arch::x86_64::_mm256_zeroupper()
    }
    if std::arch::is_x86_feature_detected!("avx") {
        // SAFETY: AVX is available, checked just above; the instruction has no memory effects.
        unsafe { zeroupper() }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn clear_upper_vector_state() {}
