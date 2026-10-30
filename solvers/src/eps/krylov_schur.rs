use super::{EigenResult, EpsError, EpsOptions, EpsReason};
use crate::basis::Basis;
use crate::dense::sym_eig;
use crate::operator::LinearOperator;
use crate::rng::random_vector;
use crate::vecops;

/// Krylov–Schur (thick-restart Lanczos) for a real symmetric operator.
///
/// Follows SLEPc's `EPSKRYLOVSCHUR` for Hermitian problems with Ritz extraction: locking of
/// converged pairs, restart keeping `⌊(nv−k)·keep⌋` Ritz vectors (at least one), relative
/// residual convergence test `β|sᵢ| < tol·|θᵢ|`, and default `ncv`/`max_it`. The basis is fully
/// re-orthogonalised (classical Gram–Schmidt with a DGKS refinement pass when needed), as
/// SLEPc does by default.
///
/// `start` is the initial vector (SLEPc `EPSSetInitialSpace`); a fixed pseudo-random vector
/// is used when it is `None`. The operator is assumed symmetric; this is not checked.
pub fn krylov_schur<A: LinearOperator + ?Sized>(
    a: &A,
    start: Option<&[f64]>,
    opts: &EpsOptions,
) -> Result<EigenResult, EpsError> {
    let n = a.dim();
    let nev = opts.nev;
    if nev == 0 || nev > n {
        return Err(EpsError::InvalidArgument("nev must be in 1..=n"));
    }
    if !(opts.tol > 0.0) {
        return Err(EpsError::InvalidArgument("tol must be positive"));
    }
    if !(0.1..=0.9).contains(&opts.keep) {
        return Err(EpsError::InvalidArgument("keep must be in [0.1, 0.9]"));
    }
    let ncv = match opts.ncv {
        Some(c) => {
            if c > n || !(c >= nev + 1 || (c == nev && c == n)) {
                return Err(EpsError::InvalidArgument("ncv must satisfy nev + 1 <= ncv <= n"));
            }
            c
        }
        None => n.min((2 * nev).max(nev + 15)),
    };
    let max_it = opts.max_it.unwrap_or_else(|| 100.max(2 * n / ncv));
    if let Some(s) = start {
        if s.len() != n {
            return Err(EpsError::InvalidArgument("start vector has wrong length"));
        }
    }

    let ld = ncv + 1;
    let mut v = Basis::new(n, ncv + 1);
    let mut t = vec![0.0; ld * ld]; // projected matrix, column-major (upper triangle used)
    let mut evals = vec![0.0; ncv];
    let mut errs = vec![f64::INFINITY; ncv];
    let mut h = vec![0.0; ld];
    let mut h2 = vec![0.0; ld];

    // Starting vector.
    {
        let col = v.col_mut(0);
        match start {
            Some(s) => col.copy_from_slice(s),
            None => col.copy_from_slice(&random_vector(n, 0x5EED_0001)),
        }
        let nrm = vecops::norm2(col);
        if !(nrm > 0.0) || !nrm.is_finite() {
            return Err(EpsError::InvalidArgument("start vector must be non-zero and finite"));
        }
        vecops::scale(1.0 / nrm, col);
    }

    let (mut nconv, mut l) = (0usize, 0usize);
    let (mut its, mut matvecs) = (0usize, 0usize);
    let mut restart_seed = 0u64;
    let reason;
    loop {
        its += 1;
        let mut nv = ncv;
        let mut beta = 0.0;
        let mut breakdown = false;

        // Expand the factorisation from column nconv + l up to nv.
        let mut j = nconv + l;
        while j < nv {
            {
                let (vj, w) = v.col_pair(j, j + 1);
                a.apply(vj, w);
            }
            matvecs += 1;
            let (bj, lindep) = orthogonalize(&mut v, j, &mut h, &mut h2);
            t[j * ld..j * ld + j + 1].copy_from_slice(&h[..=j]);
            if lindep || bj == 0.0 {
                breakdown = true;
                beta = 0.0;
                nv = j + 1;
                break;
            }
            vecops::scale(1.0 / bj, v.col_mut(j + 1));
            beta = bj;
            j += 1;
        }

        // Projected problem on the active window [nconv, nv).
        let m = nv - nconv;
        let mut tact = vec![0.0; m * m];
        for c in 0..m {
            for r in 0..=c {
                tact[c * m + r] = t[(nconv + c) * ld + nconv + r];
            }
        }
        let (w, q) = sym_eig(&tact, m, m);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&x, &y| opts.which.key(w[x]).total_cmp(&opts.which.key(w[y])));

        // Convergence: leading run of converged Ritz pairs in "wanted" order.
        let mut k = nconv;
        let mut run = true;
        for (pos, &idx) in order.iter().enumerate() {
            let theta = w[idx];
            let resid = (beta * q[idx * m + m - 1]).abs();
            let err = if resid == 0.0 {
                0.0
            } else if theta != 0.0 {
                resid / theta.abs()
            } else {
                f64::MAX
            };
            evals[nconv + pos] = theta;
            errs[nconv + pos] = err;
            if run && err < opts.tol {
                k += 1;
            } else {
                run = false;
            }
        }

        let stop = if k >= nev {
            Some(EpsReason::Converged)
        } else if its >= max_it {
            Some(EpsReason::DivergedIts)
        } else {
            None
        };
        l = if stop.is_some() || breakdown || k == nv {
            0
        } else {
            1.max(((nv - k) as f64 * opts.keep) as usize)
        };

        // V[:, nconv..k+l] ← V[:, nconv..nv] · Q(:, wanted order).
        let cols = k - nconv + l;
        if cols > 0 {
            let mut qs = vec![0.0; m * cols];
            for (c, &idx) in order.iter().take(cols).enumerate() {
                qs[c * m..(c + 1) * m].copy_from_slice(&q[idx * m..(idx + 1) * m]);
            }
            v.mult_in_place(nconv, m, cols, &qs, m);
            // Rayleigh quotient of the kept/locked part is diagonal; the coupling to the
            // next basis vector is recomputed by the next expansion.
            for c in 0..cols {
                let p = nconv + c;
                t[p * ld..p * ld + p + 1].fill(0.0);
                t[p * ld + p] = w[order[c]];
                evals[p] = w[order[c]];
            }
        }
        let converged_before = nconv;
        nconv = k;

        if let Some(r) = stop {
            reason = r;
            break;
        }
        if breakdown || k == nv {
            // Invariant subspace but fewer than nev pairs: continue from a fresh vector.
            restart_seed += 1;
            match fresh_start_vector(&mut v, nconv, restart_seed) {
                Some(()) => {}
                None => {
                    reason = EpsReason::DivergedBreakdown;
                    break;
                }
            }
        } else {
            v.copy_col(nv, nconv + l);
        }
        let _ = converged_before;
    }

    // Collect results, ordered from most to least wanted.
    let mut idx: Vec<usize> = (0..nconv).collect();
    idx.sort_by(|&x, &y| opts.which.key(evals[x]).total_cmp(&opts.which.key(evals[y])));
    let mut vectors = Basis::new(n, nconv.max(1));
    for (dst, &src) in idx.iter().enumerate() {
        vectors.col_mut(dst).copy_from_slice(v.col(src));
    }
    Ok(EigenResult {
        reason,
        values: idx.iter().map(|&i| evals[i]).collect(),
        errors: idx.iter().map(|&i| errs[i]).collect(),
        vectors,
        nconv,
        iterations: its,
        matvecs,
    })
}

/// Orthogonalise column `j+1` against columns `0..=j` (classical Gram–Schmidt, DGKS
/// refinement when more than `1 − 1/√2` of the norm cancels). Returns the remaining norm
/// and whether the new vector is numerically dependent on the previous ones.
fn orthogonalize(v: &mut Basis, j: usize, h: &mut [f64], h2: &mut [f64]) -> (f64, bool) {
    let (head, w) = v.split_col(j + 1);
    let k = j + 1;
    head.dots(k, w, h);
    // The refinement product `Vᵀw` is formed in the same pass as the update; Lanczos vectors
    // lose most of their norm in the first pass (so refinement is almost always needed), and
    // when it is not, the product is simply discarded.
    let mut after = head.sub_combination_norm_sq_dots(k, h, w, h2).sqrt();
    let sumsq: f64 = h[..k].iter().map(|x| x * x).sum();
    let before = (after * after + sumsq).sqrt();
    if after < std::f64::consts::FRAC_1_SQRT_2 * before {
        after = head.sub_combination_norm_sq(k, h2, w).sqrt();
        for (a, b) in h[..k].iter_mut().zip(&h2[..k]) {
            *a += b;
        }
    }
    (after, after <= 100.0 * f64::EPSILON * before)
}

/// Place a unit vector orthogonal to columns `0..k` in column `k`; `None` if the space is exhausted.
fn fresh_start_vector(v: &mut Basis, k: usize, seed: u64) -> Option<()> {
    if k >= v.capacity() - 1 || k >= v.len() {
        return None;
    }
    for attempt in 0..5u64 {
        let cand = random_vector(v.len(), 0xB0B0_0000 + seed * 16 + attempt);
        v.col_mut(k).copy_from_slice(&cand);
        let mut h = vec![0.0; k.max(1)];
        let before = vecops::norm2(v.col(k));
        for _ in 0..2 {
            let (head, w) = v.split_col(k);
            if k > 0 {
                head.dots(k, w, &mut h);
                head.sub_combination(k, &h, w);
            }
        }
        let after = vecops::norm2(v.col(k));
        if after > 1e-8 * before {
            vecops::scale(1.0 / after, v.col_mut(k));
            return Some(());
        }
    }
    None
}
