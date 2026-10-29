//! Krylov linear solvers (PETSc `KSP`): [`Cg`] and restarted [`Gmres`].
//!
//! Defaults, norm definitions and the convergence test follow PETSc 3.19
//! (`rtol = 1e-5`, `atol = 1e-50`, `dtol = 1e4`, `max_it = 10000`; CG monitors the
//! preconditioned residual norm, GMRES restarts every 30 iterations with left
//! preconditioning and classical Gram–Schmidt), so iteration counts are directly comparable.

mod cg;
mod gmres;

pub use cg::{Cg, CgNorm, CgOptions};
pub use gmres::{Gmres, GmresOptions, Orthogonalization, PcSide};

/// Why a solve stopped (PETSc `KSPConvergedReason`, the cases that can occur here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergedReason {
    /// `‖r‖ ≤ rtol · ‖r₀‖`.
    Rtol,
    /// `‖r‖ < atol`.
    Atol,
    /// Reached `max_it` without converging.
    DivergedIts,
    /// `‖r‖ ≥ dtol · ‖r₀‖`.
    DivergedDtol,
    /// Breakdown (GMRES) / non-finite scalar.
    DivergedBreakdown,
    /// CG saw `pᵀAp ≤ 0` or a sign change: `A` is not positive definite.
    DivergedIndefiniteMat,
    /// CG saw `rᵀM⁻¹r` change sign: `M` is not positive definite.
    DivergedIndefinitePc,
    /// A NaN or infinite residual norm.
    DivergedNanOrInf,
}

impl ConvergedReason {
    pub fn is_converged(self) -> bool {
        matches!(self, ConvergedReason::Rtol | ConvergedReason::Atol)
    }
}

/// Result of a linear solve.
#[derive(Debug, Clone)]
pub struct SolveInfo {
    pub iterations: usize,
    pub reason: ConvergedReason,
    /// The monitored residual norm at exit (see each solver's `norm_type`/`side`).
    pub residual_norm: f64,
    /// One entry per iteration (index 0 is the initial norm) if `record_history` is set.
    pub history: Vec<f64>,
    /// Number of operator applications performed.
    pub matvecs: usize,
    /// Number of preconditioner applications performed.
    pub pc_applies: usize,
}

/// Convergence tolerances (PETSc `KSPSetTolerances`).
#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    pub rtol: f64,
    pub atol: f64,
    pub dtol: f64,
    pub max_it: usize,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self { rtol: 1e-5, atol: 1e-50, dtol: 1e4, max_it: 10_000 }
    }
}

/// PETSc `KSPConvergedDefault`.
#[derive(Debug, Clone)]
pub(crate) struct ConvTest {
    tol: Tolerances,
    rnorm0: f64,
    ttol: f64,
}

impl ConvTest {
    pub(crate) fn new(tol: Tolerances) -> Self {
        Self { tol, rnorm0: 0.0, ttol: 0.0 }
    }

    /// Set the reference norm for the relative test (done at iteration 0).
    pub(crate) fn set_reference(&mut self, rnorm0: f64) {
        self.rnorm0 = rnorm0;
        self.ttol = (self.tol.rtol * rnorm0).max(self.tol.atol);
    }

    pub(crate) fn rnorm0(&self) -> f64 {
        self.rnorm0
    }

    pub(crate) fn check(&self, rnorm: f64) -> Option<ConvergedReason> {
        if !rnorm.is_finite() {
            Some(ConvergedReason::DivergedNanOrInf)
        } else if rnorm <= self.ttol {
            if rnorm < self.tol.atol {
                Some(ConvergedReason::Atol)
            } else {
                Some(ConvergedReason::Rtol)
            }
        } else if rnorm >= self.tol.dtol * self.rnorm0 {
            Some(ConvergedReason::DivergedDtol)
        } else {
            None
        }
    }
}

pub(crate) fn ensure_len(v: &mut Vec<f64>, n: usize) {
    if v.len() != n {
        v.clear();
        v.resize(n, 0.0);
    }
}
