use super::{ConvTest, ConvergedReason, SolveInfo, Tolerances, ensure_len};
use crate::operator::{LinearOperator, Preconditioner};
use crate::vecops::{self, Scaling};

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
    pub fn solve<A: LinearOperator + ?Sized, M: Preconditioner + ?Sized>(
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

        // Element-wise preconditioners (identity, Jacobi) are fused into the vector updates;
        // for the identity `z` aliases `r` and is never materialised.
        let scaling = m.scaling();
        let identity = matches!(scaling, Some(Scaling::Identity));

        let (mut beta, mut dp) = match scaling {
            Some(sc) => {
                let (zr, zz, rr) = scaled_stats(sc, r, z);
                info.pc_applies += 1;
                (zr, monitored(opts.norm, zr, zz, rr))
            }
            None => pc_and_norm(m, r, z, opts.norm, &mut info),
        };
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
            let zv: &[f64] = if identity { r } else { z };
            if i == 0 {
                vecops::copy(zv, p);
            } else {
                vecops::aypx(beta / betaold, zv, p);
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
            match scaling {
                Some(sc) => {
                    let (zr, zz, rr) = vecops::cg_update_scaled(alpha, p, x, w, r, z, sc);
                    info.pc_applies += 1;
                    beta = zr;
                    dp = monitored(opts.norm, zr, zz, rr);
                }
                None => {
                    vecops::axpy2(alpha, p, x, w, r);
                    (beta, dp) = pc_and_norm(m, r, z, opts.norm, &mut info);
                }
            }
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

fn monitored(norm: CgNorm, zr: f64, zz: f64, rr: f64) -> f64 {
    match norm {
        CgNorm::Preconditioned => zz.sqrt(),
        CgNorm::Unpreconditioned => rr.sqrt(),
        CgNorm::Natural => zr.abs().sqrt(),
    }
}

/// `(zᵀr, zᵀz, rᵀr)` for `z = M⁻¹r` with an element-wise `M⁻¹` (materialising `z` unless identity).
fn scaled_stats(sc: Scaling<'_>, r: &[f64], z: &mut [f64]) -> (f64, f64, f64) {
    match sc {
        Scaling::Identity => {
            let rr = vecops::dot(r, r);
            (rr, rr, rr)
        }
        Scaling::Diagonal(d) => {
            for ((zi, di), ri) in z.iter_mut().zip(d).zip(r) {
                *zi = di * ri;
            }
            let (zr, zz) = vecops::dot_and_sq(z, r);
            (zr, zz, vecops::dot(r, r))
        }
    }
}

/// Apply the preconditioner to `r`, returning `(β = zᵀr, monitored norm)`.
fn pc_and_norm<M: Preconditioner + ?Sized>(
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
