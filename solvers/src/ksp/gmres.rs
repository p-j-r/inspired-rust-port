use super::{ConvTest, ConvergedReason, SolveInfo, Tolerances, ensure_len};
use crate::basis::Basis;
use crate::operator::{LinearOperator, Preconditioner};
use crate::vecops;

/// Which side the preconditioner is applied on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PcSide {
    /// Solve `M⁻¹A x = M⁻¹b`; the monitored norm is `‖M⁻¹r‖` (PETSc default).
    #[default]
    Left,
    /// Solve `A M⁻¹ y = b`, `x = M⁻¹y`; the monitored norm is the true `‖r‖`.
    Right,
}

/// Gram–Schmidt variant used to orthogonalise each new Krylov vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Orthogonalization {
    /// Classical Gram–Schmidt, one pass (PETSc default; one fused multi-dot per step).
    #[default]
    ClassicalGramSchmidt,
    /// Classical GS with a DGKS second pass when the norm drops by more than 1/√2.
    ClassicalGramSchmidtIfNeeded,
    /// Modified Gram–Schmidt.
    ModifiedGramSchmidt,
}

#[derive(Debug, Clone)]
pub struct GmresOptions {
    pub tol: Tolerances,
    pub restart: usize,
    pub side: PcSide,
    pub orthog: Orthogonalization,
    pub initial_guess_nonzero: bool,
    pub record_history: bool,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            tol: Tolerances::default(),
            restart: 30,
            side: PcSide::Left,
            orthog: Orthogonalization::ClassicalGramSchmidt,
            initial_guess_nonzero: false,
            record_history: false,
        }
    }
}

/// Restarted GMRES(m) with Givens-rotation least-squares updates.
#[derive(Debug, Clone, Default)]
pub struct Gmres {
    pub opts: GmresOptions,
    v: Option<Basis>,
    r: Vec<f64>,
    t: Vec<f64>,
    u: Vec<f64>,
    h: Vec<f64>,
    cs: Vec<f64>,
    sn: Vec<f64>,
    rs: Vec<f64>,
}

const HAPPY_TOL: f64 = 1e-30;
const BREAKDOWN_TOL: f64 = 0.1;

impl Gmres {
    pub fn new(opts: GmresOptions) -> Self {
        Self { opts, ..Default::default() }
    }

    pub fn solve<A: LinearOperator + ?Sized, M: Preconditioner + ?Sized>(
        &mut self,
        a: &A,
        m: &M,
        b: &[f64],
        x: &mut [f64],
    ) -> SolveInfo {
        let n = a.dim();
        assert!(b.len() == n && x.len() == n, "dimension mismatch");
        let restart = self.opts.restart.max(1);
        let opts = self.opts.clone();
        if self.v.as_ref().is_none_or(|v| v.len() != n || v.capacity() != restart + 1) {
            self.v = Some(Basis::new(n, restart + 1));
        }
        for v in [&mut self.r, &mut self.t, &mut self.u] {
            ensure_len(v, n);
        }
        let ld = restart + 1;
        self.h.clear();
        self.h.resize(ld * restart, 0.0);
        self.cs.clear();
        self.cs.resize(restart, 0.0);
        self.sn.clear();
        self.sn.resize(restart, 0.0);
        self.rs.clear();
        self.rs.resize(ld, 0.0);
        let Self { v, r, t, u, h, cs, sn, rs, .. } = self;
        let v = v.as_mut().unwrap();

        let mut conv = ConvTest::new(opts.tol);
        let mut info = SolveInfo {
            iterations: 0,
            reason: ConvergedReason::DivergedIts,
            residual_norm: 0.0,
            history: Vec::new(),
            matvecs: 0,
            pc_applies: 0,
        };
        let left = opts.side == PcSide::Left;
        let mut guess_nonzero = opts.initial_guess_nonzero;
        if !guess_nonzero {
            vecops::zero(x);
        }
        let mut first_cycle = true;
        let mut last_rnorm = -1.0f64; // recursion estimate carried between cycles
        let mut its = 0usize;

        'outer: loop {
            // Initial residual of this cycle, in v[0]:  left: M⁻¹(b − A x),  right: b − A x.
            if guess_nonzero {
                a.apply(x, r);
                info.matvecs += 1;
                vecops::aypx(-1.0, b, r);
            } else {
                vecops::copy(b, r);
            }
            if left {
                m.apply(r, v.col_mut(0));
                info.pc_applies += 1;
            } else {
                v.col_mut(0).copy_from_slice(r);
            }
            let res = vecops::norm2(v.col(0));
            if !res.is_finite() {
                info.reason = ConvergedReason::DivergedNanOrInf;
                info.residual_norm = res;
                break;
            }
            if first_cycle {
                // Relative tolerance reference: the initial residual, or (with a non-zero
                // guess) the norm of the right-hand side measured the same way.
                let reference = if guess_nonzero {
                    if left {
                        m.apply(b, t);
                        info.pc_applies += 1;
                        let s = vecops::norm2(t);
                        if s == 0.0 { res } else { s }
                    } else {
                        let s = vecops::norm2(b);
                        if s == 0.0 { res } else { s }
                    }
                } else {
                    res
                };
                conv.set_reference(reference);
                if opts.record_history {
                    info.history.push(res);
                }
            } else if last_rnorm > 0.0 && (res - last_rnorm).abs() > BREAKDOWN_TOL * conv.rnorm0().max(f64::MIN_POSITIVE) {
                // Recursion-formula residual disagrees badly with the recomputed one.
                info.reason = ConvergedReason::DivergedBreakdown;
                info.residual_norm = res;
                break;
            }
            info.residual_norm = res;
            if res == 0.0 {
                info.reason = ConvergedReason::Atol;
                break;
            }
            if let Some(reason) = conv.check(res) {
                info.reason = reason;
                break;
            }
            vecops::scale(1.0 / res, v.col_mut(0));
            rs.fill(0.0);
            rs[0] = res;

            let mut k = 0usize; // columns completed this cycle
            let mut reason = None;
            let mut res_k = res;
            while k < restart && its < opts.tol.max_it {
                // w = v[k+1] ← B A v[k]  (left) or  A M⁻¹ v[k]  (right)
                {
                    let (vk, w) = v.col_pair(k, k + 1);
                    if left {
                        a.apply(vk, t);
                        m.apply(t, w);
                    } else {
                        m.apply(vk, t);
                        a.apply(t, w);
                    }
                }
                info.matvecs += 1;
                info.pc_applies += 1;
                let hcol = &mut h[k * ld..(k + 1) * ld];
                let tt = orthogonalize(v, k, opts.orthog, hcol);
                hcol[k + 1] = tt;
                if !(tt.is_finite()) {
                    reason = Some(ConvergedReason::DivergedNanOrInf);
                    break;
                }
                if tt != 0.0 {
                    vecops::scale(1.0 / tt, v.col_mut(k + 1));
                }
                // Happy breakdown: the Krylov space is invariant.
                let hapbnd = (tt / rs[k]).abs().min(HAPPY_TOL);
                let happy = tt < hapbnd;
                // Apply previous rotations to the new column.
                for j in 0..k {
                    let (c, s) = (cs[j], sn[j]);
                    let (hj, hj1) = (hcol[j], hcol[j + 1]);
                    hcol[j] = c * hj + s * hj1;
                    hcol[j + 1] = c * hj1 - s * hj;
                }
                if !happy {
                    let denom = (hcol[k] * hcol[k] + hcol[k + 1] * hcol[k + 1]).sqrt();
                    if denom == 0.0 {
                        reason = Some(ConvergedReason::DivergedBreakdown);
                        break;
                    }
                    cs[k] = hcol[k] / denom;
                    sn[k] = hcol[k + 1] / denom;
                    rs[k + 1] = -sn[k] * rs[k];
                    rs[k] = cs[k] * rs[k];
                    hcol[k] = cs[k] * hcol[k] + sn[k] * hcol[k + 1];
                    res_k = rs[k + 1].abs();
                } else {
                    res_k = 0.0;
                }
                hcol[k + 1] = 0.0;
                k += 1;
                its += 1;
                info.iterations = its;
                info.residual_norm = res_k;
                if opts.record_history {
                    info.history.push(res_k);
                }
                reason = conv.check(res_k);
                if happy && reason.is_none() {
                    reason = Some(ConvergedReason::DivergedBreakdown);
                }
                if reason.is_some() {
                    break;
                }
            }

            // Solve the k×k upper-triangular system H y = rs and update x.
            if k > 0 {
                let mut y = vec![0.0; k];
                for i in (0..k).rev() {
                    let mut s = rs[i];
                    for j in i + 1..k {
                        s -= h[j * ld + i] * y[j];
                    }
                    let d = h[i * ld + i];
                    if d == 0.0 {
                        reason = Some(ConvergedReason::DivergedBreakdown);
                        y[i] = 0.0;
                    } else {
                        y[i] = s / d;
                    }
                }
                if left {
                    v.add_combination(k, &y, x);
                } else {
                    // x += M⁻¹ (V y)
                    vecops::zero(t);
                    v.add_combination(k, &y, t);
                    m.apply(t, u);
                    info.pc_applies += 1;
                    vecops::axpy(1.0, u, x);
                }
            }
            last_rnorm = res_k;
            if let Some(reason) = reason {
                info.reason = reason;
                break 'outer;
            }
            if its >= opts.tol.max_it {
                info.reason = ConvergedReason::DivergedIts;
                break;
            }
            guess_nonzero = true;
            first_cycle = false;
        }
        info
    }
}

/// Orthogonalise `v[k+1]` against `v[0..=k]`, filling `h[0..=k]` and returning the
/// remaining norm.
fn orthogonalize(v: &mut Basis, k: usize, kind: Orthogonalization, h: &mut [f64]) -> f64 {
    let (head, w) = v.split_col(k + 1);
    match kind {
        Orthogonalization::ClassicalGramSchmidt => {
            head.dots(k + 1, w, h);
            head.sub_combination_norm_sq(k + 1, h, w).sqrt()
        }
        Orthogonalization::ClassicalGramSchmidtIfNeeded => {
            let before = vecops::norm2(w);
            head.dots(k + 1, w, h);
            head.sub_combination(k + 1, h, w);
            let mut after = vecops::norm2(w);
            if after < std::f64::consts::FRAC_1_SQRT_2 * before {
                let mut h2 = vec![0.0; k + 1];
                head.dots(k + 1, w, &mut h2);
                head.sub_combination(k + 1, &h2, w);
                for (a, b) in h.iter_mut().zip(&h2) {
                    *a += b;
                }
                after = vecops::norm2(w);
            }
            after
        }
        Orthogonalization::ModifiedGramSchmidt => {
            for j in 0..=k {
                let c = vecops::dot(head.col(j), w);
                h[j] = c;
                vecops::axpy(-c, head.col(j), w);
            }
            vecops::norm2(w)
        }
    }
}
