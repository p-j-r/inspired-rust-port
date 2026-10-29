use super::{ConvTest, ConvergedReason, SolveInfo, Tolerances, ensure_len};
use crate::operator::{LinearOperator, Preconditioner};
use crate::vecops;

/// Which residual norm CG monitors (PETSc `KSPSetNormType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CgNorm {
    /// `‖M⁻¹r‖₂` (PETSc default for CG).
    #[default]
    Preconditioned,
    /// `‖r‖₂`.
    Unpreconditioned,
    /// `√(rᵀM⁻¹r)`, the energy-like norm.
    Natural,
}

#[derive(Debug, Clone, Default)]
pub struct CgOptions {
    pub tol: Tolerances,
    pub norm: CgNorm,
    /// Start from the contents of `x` instead of zero.
    pub initial_guess_nonzero: bool,
    pub record_history: bool,
}

/// Preconditioned conjugate gradients (Hestenes–Stiefel) for symmetric positive definite
/// operators, reusing its work vectors across solves.
#[derive(Debug, Clone, Default)]
pub struct Cg {
    pub opts: CgOptions,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    w: Vec<f64>,
}

impl Cg {
    pub fn new(opts: CgOptions) -> Self {
        Self { opts, ..Default::default() }
    }

    /// Solve `A x = b`. `x` is the initial guess if `opts.initial_guess_nonzero`, and is
    /// overwritten with the solution in every case.
    pub fn solve<A: LinearOperator, M: Preconditioner>(
        &mut self,
        a: &A,
        m: &M,
        b: &[f64],
        x: &mut [f64],
    ) -> SolveInfo {
        let n = a.dim();
        assert!(b.len() == n && x.len() == n, "dimension mismatch");
        for v in [&mut self.r, &mut self.z, &mut self.p, &mut self.w] {
            ensure_len(v, n);
        }
        let Self { opts, r, z, p, w } = self;
        let tol = opts.tol;
        let mut conv = ConvTest::new(tol);
        let mut info = SolveInfo {
            iterations: 0,
            reason: ConvergedReason::DivergedIts,
            residual_norm: 0.0,
            history: Vec::new(),
            matvecs: 0,
            pc_applies: 0,
        };

        // r ← b − A x
        if opts.initial_guess_nonzero {
            a.apply(x, r);
            info.matvecs += 1;
            vecops::aypx(-1.0, b, r);
        } else {
            vecops::zero(x);
            vecops::copy(b, r);
        }

        // Initial norm and β = zᵀr.
        let (mut beta, mut dp) = pc_and_norm(m, r, z, opts.norm, &mut info);
        // With a non-zero guess PETSc measures rtol against the preconditioned/natural
        // norm of the right-hand side instead of the initial residual.
        let reference = if opts.initial_guess_nonzero {
            let mut zb = vec![0.0; n];
            let (zr, zz) = m.apply_dots(b, &mut zb);
            info.pc_applies += 1;
            let snorm = match opts.norm {
                CgNorm::Preconditioned => zz.sqrt(),
                CgNorm::Unpreconditioned => vecops::norm2(b),
                CgNorm::Natural => zr.abs().sqrt(),
            };
            if snorm == 0.0 { dp } else { snorm }
        } else {
            dp
        };
        conv.set_reference(reference);
        info.residual_norm = dp;
        if opts.record_history {
            info.history.push(dp);
        }
        if let Some(reason) = conv.check(dp) {
            info.reason = reason;
            return info;
        }

        let (mut betaold, mut dpi_old) = (1.0f64, 0.0f64);
        let mut i = 0usize;
        loop {
            info.iterations = i + 1;
            if beta == 0.0 {
                info.reason = ConvergedReason::Atol;
                break;
            }
            if i > 0 && beta * betaold < 0.0 {
                info.reason = ConvergedReason::DivergedIndefinitePc;
                break;
            }
            if i == 0 {
                vecops::copy(z, p);
            } else {
                vecops::aypx(beta / betaold, z, p);
            }
            let dpi = a.apply_dot(p, w);
            info.matvecs += 1;
            betaold = beta;
            if dpi == 0.0 || (i > 0 && dpi.signum() * dpi_old.signum() < 0.0) {
                info.reason = ConvergedReason::DivergedIndefiniteMat;
                break;
            }
            if !dpi.is_finite() {
                info.reason = ConvergedReason::DivergedNanOrInf;
                break;
            }
            dpi_old = dpi;
            let alpha = beta / dpi;
            vecops::axpy2(alpha, p, x, w, r);
            (beta, dp) = pc_and_norm(m, r, z, opts.norm, &mut info);
            info.residual_norm = dp;
            if opts.record_history {
                info.history.push(dp);
            }
            if let Some(reason) = conv.check(dp) {
                info.reason = reason;
                break;
            }
            i += 1;
            if i >= tol.max_it {
                info.reason = ConvergedReason::DivergedIts;
                info.iterations = i;
                break;
            }
        }
        info
    }
}

/// Apply the preconditioner to `r`, returning `(β = zᵀr, monitored norm)`.
fn pc_and_norm<M: Preconditioner>(
    m: &M,
    r: &[f64],
    z: &mut [f64],
    norm: CgNorm,
    info: &mut SolveInfo,
) -> (f64, f64) {
    let (zr, zz) = m.apply_dots(r, z);
    info.pc_applies += 1;
    let dp = match norm {
        CgNorm::Preconditioned => zz.sqrt(),
        CgNorm::Unpreconditioned => vecops::norm2(r),
        CgNorm::Natural => zr.abs().sqrt(),
    };
    (zr, dp)
}
