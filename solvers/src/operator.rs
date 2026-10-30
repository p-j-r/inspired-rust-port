//! Abstractions for matrix-free operators and preconditioners.

use crate::vecops;

/// A square linear operator `y = A x` (PETSc `Mat` / `MATSHELL` in the role of a matrix).
pub trait LinearOperator {
    /// Dimension `n` of the (square) operator.
    fn dim(&self) -> usize;

    /// `y ← A x`. Both slices have length [`dim`](Self::dim).
    fn apply(&self, x: &[f64], y: &mut [f64]);

    /// `y ← A x`, returning `xᵀy`. Implementations may fuse the reduction into the
    /// product; the default does the two passes separately.
    fn apply_dot(&self, x: &[f64], y: &mut [f64]) -> f64 {
        self.apply(x, y);
        vecops::dot(x, y)
    }
}

/// A preconditioner application `z ← M⁻¹ r` (PETSc `PCApply`).
pub trait Preconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]);

    /// `z ← M⁻¹ r`, returning `(zᵀr, zᵀz)`. Implementations may fuse the reductions
    /// into the application; the result must equal the unfused one bit-for-bit.
    fn apply_dots(&self, r: &[f64], z: &mut [f64]) -> (f64, f64) {
        self.apply(r, z);
        vecops::dot_and_sq(z, r)
    }

    /// Reports an element-wise structure so that solvers may fuse `M⁻¹` into their vector
    /// updates; `None` (the default) means a general operator.
    fn scaling(&self) -> Option<vecops::Scaling<'_>> {
        None
    }
}

/// `M = I` (PETSc `PCNONE`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

impl Preconditioner for Identity {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        vecops::copy(r, z);
    }
    fn scaling(&self) -> Option<vecops::Scaling<'_>> {
        Some(vecops::Scaling::Identity)
    }
}

/// Wrap a closure as a [`LinearOperator`] (the analogue of `MatShell`).
pub struct FnOperator<F> {
    n: usize,
    f: F,
}

impl<F: Fn(&[f64], &mut [f64])> FnOperator<F> {
    pub fn new(n: usize, f: F) -> Self {
        Self { n, f }
    }
}

impl<F: Fn(&[f64], &mut [f64])> LinearOperator for FnOperator<F> {
    fn dim(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        (self.f)(x, y)
    }
}

impl<T: LinearOperator + ?Sized> LinearOperator for &T {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        (**self).apply(x, y)
    }
    fn apply_dot(&self, x: &[f64], y: &mut [f64]) -> f64 {
        (**self).apply_dot(x, y)
    }
}

impl<T: Preconditioner + ?Sized> Preconditioner for &T {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        (**self).apply(r, z)
    }
    fn apply_dots(&self, r: &[f64], z: &mut [f64]) -> (f64, f64) {
        (**self).apply_dots(r, z)
    }
    fn scaling(&self) -> Option<vecops::Scaling<'_>> {
        (**self).scaling()
    }
}
