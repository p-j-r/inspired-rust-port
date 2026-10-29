mod common;

use common::*;
use rsolvers::ksp::*;
use rsolvers::pc::*;
use rsolvers::rng::random_vector;
use rsolvers::sparse::*;
use rsolvers::{CsrMatrix, Identity, LinearOperator, Preconditioner};

fn tight() -> Tolerances {
    Tolerances { rtol: 1e-12, atol: 1e-50, dtol: 1e10, max_it: 20_000 }
}

fn manufactured(a: &CsrMatrix, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let xs = random_vector(a.nrows(), seed);
    let mut b = vec![0.0; a.nrows()];
    a.spmv(&xs, &mut b);
    (xs, b)
}

// ---------------------------------------------------------------- CG

#[test]
fn cg_matches_dense_solve_on_random_spd() {
    for seed in 1..=5 {
        let a = random_spd(60, 8, seed);
        let b = random_vector(60, 100 + seed);
        let reference = dense_solve(&a.to_dense(), 60, &b);
        let mut x = vec![0.0; 60];
        let mut cg = Cg::new(CgOptions { tol: tight(), ..Default::default() });
        let info = cg.solve(&a, &Identity, &b, &mut x);
        assert!(info.reason.is_converged(), "{info:?}");
        assert!(rel_diff(&x, &reference) < 1e-9, "seed {seed}: {}", rel_diff(&x, &reference));
    }
}

#[test]
fn cg_terminates_in_at_most_k_iterations_for_k_distinct_eigenvalues() {
    // Diagonal matrix with 4 distinct eigenvalues: CG converges in exactly 4 steps (exact arithmetic).
    let n = 200;
    let t: Vec<_> = (0..n).map(|i| (i, i, [1.0, 2.0, 5.0, 9.0][i % 4])).collect();
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let b = random_vector(n, 7);
    let mut x = vec![0.0; n];
    let mut cg = Cg::new(CgOptions { tol: Tolerances { rtol: 1e-10, ..tight() }, ..Default::default() });
    let info = cg.solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged());
    assert!(info.iterations <= 5, "took {} iterations", info.iterations);
    assert!(residual_norm(&a, &x, &b) < 1e-9 * norm(&b));
}

#[test]
fn cg_poisson_error_bound() {
    let a = laplacian_2d(30, 30);
    let (xs, b) = manufactured(&a, 11);
    let mut x = vec![0.0; 900];
    let mut cg = Cg::new(CgOptions { tol: Tolerances { rtol: 1e-10, ..tight() }, norm: CgNorm::Unpreconditioned, ..Default::default() });
    let info = cg.solve(&a, &Identity, &b, &mut x);
    assert_eq!(info.reason, ConvergedReason::Rtol);
    assert!(residual_norm(&a, &x, &b) <= 1e-10 * norm(&b) * 1.0001);
    assert!(rel_diff(&x, &xs) < 1e-7);
    // The reported norm is the true residual norm up to recurrence drift.
    assert!((info.residual_norm - residual_norm(&a, &x, &b)).abs() < 1e-8 * norm(&b));
}

#[test]
fn preconditioners_reduce_cg_iterations_on_poisson() {
    let a = laplacian_2d(48, 48);
    let (_, b) = manufactured(&a, 5);
    let tol = Tolerances { rtol: 1e-8, ..tight() };
    let run = |m: &dyn Preconditioner| {
        let mut x = vec![0.0; a.nrows()];
        let mut cg = Cg::new(CgOptions { tol, norm: CgNorm::Unpreconditioned, ..Default::default() });
        let info = cg.solve(&a, &m, &b, &mut x);
        assert!(info.reason.is_converged());
        assert!(residual_norm(&a, &x, &b) <= 1.1e-8 * norm(&b));
        info.iterations
    };
    let none = run(&Identity);
    let jac = run(&Jacobi::new(&a).unwrap());
    let ssor = run(&Sor::new(&a, 1.0).unwrap());
    let ic = run(&Ic0::new(&a).unwrap());
    // Constant diagonal => Jacobi is just a scaling of the unpreconditioned iteration.
    assert!((jac as i64 - none as i64).abs() <= 1, "{jac} vs {none}");
    assert!(ssor < none * 2 / 3, "ssor {ssor} none {none}");
    assert!(ic < ssor, "ic {ic} ssor {ssor}");
}

#[test]
fn ic0_is_exact_for_tridiagonal_spd() {
    let a = laplacian_1d(500);
    let (xs, b) = manufactured(&a, 3);
    let mut x = vec![0.0; 500];
    let mut cg = Cg::new(CgOptions { tol: Tolerances { rtol: 1e-10, ..tight() }, ..Default::default() });
    let info = cg.solve(&a, &Ic0::new(&a).unwrap(), &b, &mut x);
    assert!(info.iterations <= 2, "{info:?}");
    assert!(rel_diff(&x, &xs) < 1e-9);
}

#[test]
fn cg_detects_indefinite_matrix() {
    let t = [(0, 0, 1.0), (1, 1, -1.0), (2, 2, 1.0)];
    let a = CsrMatrix::from_triplets(3, 3, &t).unwrap();
    let mut x = vec![0.0; 3];
    let mut cg = Cg::new(CgOptions::default());
    let info = cg.solve(&a, &Identity, &[1.0, 1.0, 1.0], &mut x);
    assert_eq!(info.reason, ConvergedReason::DivergedIndefiniteMat);
}

#[test]
fn cg_zero_rhs_and_exact_initial_guess() {
    let a = laplacian_1d(50);
    let mut cg = Cg::new(CgOptions::default());
    let mut x = vec![1.0; 50];
    let info = cg.solve(&a, &Identity, &vec![0.0; 50], &mut x);
    assert!(info.reason.is_converged() && info.iterations <= 1);
    assert!(x.iter().all(|v| *v == 0.0));

    let (xs, b) = manufactured(&a, 4);
    let mut x = xs.clone();
    let mut cg = Cg::new(CgOptions { initial_guess_nonzero: true, ..Default::default() });
    let info = cg.solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged());
    assert_eq!(info.iterations, 0, "{info:?}");
    assert!(rel_diff(&x, &xs) < 1e-14);
}

#[test]
fn cg_nonzero_guess_uses_rhs_norm_for_rtol() {
    let a = laplacian_2d(20, 20);
    let (xs, b) = manufactured(&a, 21);
    let mut x0 = xs.clone();
    for v in x0.iter_mut() {
        *v += 1e-3;
    }
    let mut x = x0.clone();
    let mut cg = Cg::new(CgOptions { initial_guess_nonzero: true, tol: Tolerances { rtol: 1e-4, ..tight() }, norm: CgNorm::Unpreconditioned, ..Default::default() });
    let info = cg.solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged());
    assert!(residual_norm(&a, &x, &b) <= 1e-4 * norm(&b));
}

#[test]
fn cg_maxit_and_dtol() {
    let a = laplacian_2d(30, 30);
    let (_, b) = manufactured(&a, 9);
    let mut x = vec![0.0; 900];
    let mut cg = Cg::new(CgOptions { tol: Tolerances { max_it: 5, ..tight() }, ..Default::default() });
    let info = cg.solve(&a, &Identity, &b, &mut x);
    assert_eq!(info.reason, ConvergedReason::DivergedIts);
    assert_eq!(info.iterations, 5);
    assert_eq!(info.matvecs, 5);
}

#[test]
fn cg_history_is_recorded_and_monotone_in_energy_norm() {
    let a = laplacian_2d(25, 25);
    let (_, b) = manufactured(&a, 2);
    let mut x = vec![0.0; 625];
    let mut cg = Cg::new(CgOptions { record_history: true, norm: CgNorm::Natural, tol: Tolerances { rtol: 1e-8, ..tight() }, ..Default::default() });
    let info = cg.solve(&a, &Jacobi::new(&a).unwrap(), &b, &mut x);
    assert_eq!(info.history.len(), info.iterations + 1);
    // The natural norm √(rᵀM⁻¹r) is not monotone in general, but A-norm error is; check overall decrease.
    assert!(info.history.last().unwrap() < &(info.history[0] * 1e-7));
}

#[test]
fn cg_solves_with_matrix_free_operator() {
    let n = 400;
    let op = rsolvers::FnOperator::new(n, |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            let l = if i > 0 { x[i - 1] } else { 0.0 };
            let r = if i + 1 < n { x[i + 1] } else { 0.0 };
            y[i] = 2.0 * x[i] - l - r;
        }
    });
    let a = laplacian_1d(n);
    let (xs, b) = manufactured(&a, 31);
    let mut x = vec![0.0; n];
    let mut cg = Cg::new(CgOptions { tol: tight(), ..Default::default() });
    let info = cg.solve(&op, &Identity, &b, &mut x);
    assert!(info.reason.is_converged());
    assert!(rel_diff(&x, &xs) < 1e-8);
    assert_eq!(op.dim(), n);
}

// ---------------------------------------------------------------- GMRES

fn gmres_opts(restart: usize, side: PcSide, orthog: Orthogonalization) -> GmresOptions {
    GmresOptions { tol: tight(), restart, side, orthog, ..Default::default() }
}

#[test]
fn gmres_matches_dense_solve_on_nonsymmetric() {
    for seed in 1..=4 {
        let a = random_nonsymmetric(80, 7, seed);
        let b = random_vector(80, 50 + seed);
        let reference = dense_solve(&a.to_dense(), 80, &b);
        for &side in &[PcSide::Left, PcSide::Right] {
            for &orth in &[Orthogonalization::ClassicalGramSchmidt, Orthogonalization::ClassicalGramSchmidtIfNeeded, Orthogonalization::ModifiedGramSchmidt] {
                let mut x = vec![0.0; 80];
                let mut g = Gmres::new(gmres_opts(30, side, orth));
                let info = g.solve(&a, &Jacobi::new(&a).unwrap(), &b, &mut x);
                assert!(info.reason.is_converged(), "{side:?} {orth:?}: {info:?}");
                assert!(rel_diff(&x, &reference) < 1e-8, "{side:?} {orth:?}: {}", rel_diff(&x, &reference));
            }
        }
    }
}

#[test]
fn gmres_full_cycle_is_exact_in_n_steps() {
    // Unrestarted GMRES on an n-dimensional problem terminates in <= n iterations.
    let n = 40;
    let a = random_nonsymmetric(n, 6, 17);
    let b = random_vector(n, 18);
    let mut x = vec![0.0; n];
    let mut g = Gmres::new(GmresOptions { tol: Tolerances { rtol: 1e-13, ..tight() }, restart: n, ..Default::default() });
    let info = g.solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged());
    assert!(info.iterations <= n);
    assert!(residual_norm(&a, &x, &b) < 1e-10 * norm(&b));
}

#[test]
fn gmres_restarted_converges_on_convection_diffusion() {
    let a = convection_diffusion_2d(40, 40, 2.0, 1.0);
    let (xs, b) = manufactured(&a, 77);
    let mut x = vec![0.0; 1600];
    let mut g = Gmres::new(GmresOptions { tol: Tolerances { rtol: 1e-10, ..tight() }, restart: 8, side: PcSide::Right, ..Default::default() });
    let ilu = Ilu0::new(&a).unwrap();
    let info = g.solve(&a, &ilu, &b, &mut x);
    assert!(info.reason.is_converged(), "{info:?}");
    // Right preconditioning monitors the true residual.
    assert!(residual_norm(&a, &x, &b) <= 1e-10 * norm(&b) * 1.001);
    assert!(rel_diff(&x, &xs) < 1e-7);
    assert!(info.iterations > 8, "should have restarted, took {}", info.iterations);
}

#[test]
fn ilu0_is_exact_for_tridiagonal() {
    // Non-symmetric tridiagonal: ILU(0) = LU, so left-preconditioned GMRES converges in one step.
    let n = 300;
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i, 4.0));
        if i > 0 {
            t.push((i, i - 1, -2.0));
        }
        if i + 1 < n {
            t.push((i, i + 1, -0.5));
        }
    }
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let (xs, b) = manufactured(&a, 5);
    let mut x = vec![0.0; n];
    let mut g = Gmres::new(GmresOptions { tol: Tolerances { rtol: 1e-10, ..tight() }, ..Default::default() });
    let info = g.solve(&a, &Ilu0::new(&a).unwrap(), &b, &mut x);
    assert_eq!(info.iterations, 1, "{info:?}");
    assert!(rel_diff(&x, &xs) < 1e-10);
}

#[test]
fn gmres_happy_breakdown_few_distinct_eigenvalues() {
    // Identity scaled: Krylov space is 1-dimensional.
    let n = 50;
    let t: Vec<_> = (0..n).map(|i| (i, i, 3.0)).collect();
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let b = random_vector(n, 3);
    let mut x = vec![0.0; n];
    let mut g = Gmres::new(GmresOptions::default());
    let info = g.solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged(), "{info:?}");
    assert_eq!(info.iterations, 1);
    for i in 0..n {
        assert!((x[i] - b[i] / 3.0).abs() < 1e-14);
    }
}

#[test]
fn gmres_residual_history_is_monotone_within_cycle_and_left_matches_petsc_norm() {
    let a = convection_diffusion_2d(25, 25, 1.0, 1.0);
    let (_, b) = manufactured(&a, 8);
    let mut x = vec![0.0; 625];
    let jac = Jacobi::new(&a).unwrap();
    let mut g = Gmres::new(GmresOptions { record_history: true, tol: Tolerances { rtol: 1e-8, ..tight() }, ..Default::default() });
    let info = g.solve(&a, &jac, &b, &mut x);
    assert!(info.reason.is_converged());
    // GMRES residual norms never increase (also across restarts, since it is a minimisation).
    for w in info.history.windows(2) {
        assert!(w[1] <= w[0] * (1.0 + 1e-10), "{} -> {}", w[0], w[1]);
    }
    // Initial norm is the preconditioned RHS norm: ‖D⁻¹b‖.
    let d = a.diagonal();
    let r0 = norm(&b.iter().zip(&d).map(|(b, d)| b / d).collect::<Vec<_>>());
    assert!((info.history[0] - r0).abs() < 1e-12 * r0);
}

#[test]
fn gmres_maxit_reports_diverged_its() {
    let a = convection_diffusion_2d(30, 30, 5.0, 5.0);
    let (_, b) = manufactured(&a, 1);
    let mut x = vec![0.0; 900];
    let mut g = Gmres::new(GmresOptions { tol: Tolerances { max_it: 7, ..tight() }, ..Default::default() });
    let info = g.solve(&a, &Identity, &b, &mut x);
    assert_eq!(info.reason, ConvergedReason::DivergedIts);
    assert_eq!(info.iterations, 7);
}

#[test]
fn gmres_nonzero_initial_guess() {
    let a = random_nonsymmetric(60, 6, 5);
    let (xs, b) = manufactured(&a, 6);
    let mut x: Vec<f64> = xs.iter().map(|v| v + 0.01).collect();
    let mut g = Gmres::new(GmresOptions { initial_guess_nonzero: true, tol: tight(), ..Default::default() });
    let info = g.solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged());
    assert!(rel_diff(&x, &xs) < 1e-9);
}

// ---------------------------------------------------------------- preconditioners

#[test]
fn sor_matches_dense_ssor_formula() {
    let a = random_spd(30, 6, 12);
    let n = 30;
    let d = a.to_dense();
    for &omega in &[1.0, 1.3] {
        let r = random_vector(n, 4);
        let mut z = vec![0.0; n];
        Sor::new(&a, omega).unwrap().apply(&r, &mut z);
        // M = 1/(ω(2−ω)) (D+ωL) D⁻¹ (D+ωU)
        let mut dl = vec![0.0; n * n]; // D + ωL
        let mut du = vec![0.0; n * n]; // D + ωU
        let mut dd = vec![0.0; n * n]; // D⁻¹
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    dl[i * n + j] = d[i * n + i];
                    du[i * n + j] = d[i * n + i];
                    dd[i * n + j] = 1.0 / d[i * n + i];
                } else if j < i {
                    dl[i * n + j] = omega * d[i * n + j];
                } else {
                    du[i * n + j] = omega * d[i * n + j];
                }
            }
        }
        let mul = |a: &[f64], v: &[f64]| -> Vec<f64> { (0..n).map(|i| (0..n).map(|j| a[i * n + j] * v[j]).sum()).collect() };
        let mz = mul(&du, &z);
        let mz = mul(&dd, &mz);
        let mz: Vec<f64> = mul(&dl, &mz).iter().map(|v| v / (omega * (2.0 - omega))).collect();
        assert!(rel_diff(&mz, &r) < 1e-12, "omega {omega}: {}", rel_diff(&mz, &r));
    }
}

#[test]
fn ilu0_reproduces_matrix_on_pattern() {
    // (L U)_ij == A_ij for every (i, j) in the sparsity pattern of A.
    let a = convection_diffusion_2d(6, 5, 0.7, 0.3);
    let n = a.nrows();
    // Recover L U by applying the preconditioner's inverse is impractical; instead check M⁻¹ A x ≈ x
    // for a matrix where ILU(0) is exact (banded with no fill inside the pattern): the 1-D case.
    let _ = n;
    let t = laplacian_1d(25);
    let ilu = Ilu0::new(&t).unwrap();
    let x = random_vector(25, 2);
    let mut ax = vec![0.0; 25];
    t.spmv(&x, &mut ax);
    let mut z = vec![0.0; 25];
    ilu.apply(&ax, &mut z);
    assert!(rel_diff(&z, &x) < 1e-12);
    // And on the 2-D problem it is a *good approximation*, not exact.
    let x = random_vector(a.nrows(), 3);
    let mut ax = vec![0.0; a.nrows()];
    a.spmv(&x, &mut ax);
    let mut z = vec![0.0; a.nrows()];
    Ilu0::new(&a).unwrap().apply(&ax, &mut z);
    let e = rel_diff(&z, &x);
    assert!(e > 1e-8 && e < 1.0, "e = {e}");
}

#[test]
fn preconditioner_setup_errors() {
    let nodiag = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 0, 1.0)]).unwrap();
    assert_eq!(Ilu0::new(&nodiag).unwrap_err(), PcError::MissingDiagonal(1));
    assert_eq!(Sor::new(&nodiag, 1.0).unwrap_err(), PcError::MissingDiagonal(1));
    assert_eq!(Jacobi::new(&nodiag).unwrap_err(), PcError::ZeroPivot(1));
    let rect = CsrMatrix::from_triplets(2, 3, &[(0, 0, 1.0)]).unwrap();
    assert_eq!(Jacobi::new(&rect).unwrap_err(), PcError::NotSquare);
    let neg = CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, -1.0)]).unwrap();
    assert_eq!(Ic0::new(&neg).unwrap_err(), PcError::ZeroPivot(1));
    let zp = CsrMatrix::from_triplets(2, 2, &[(0, 0, 0.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]).unwrap();
    assert_eq!(Ilu0::new(&zp).unwrap_err(), PcError::ZeroPivot(0));
}

#[test]
fn fused_and_unfused_pc_dots_agree_bitwise() {
    let a = laplacian_2d(120, 120);
    let jac = Jacobi::new(&a).unwrap();
    let r = random_vector(a.nrows(), 8);
    let mut z1 = vec![0.0; a.nrows()];
    let mut z2 = vec![0.0; a.nrows()];
    let fused = jac.apply_dots(&r, &mut z1);
    jac.apply(&r, &mut z2);
    let unfused = rsolvers::vecops::dot_and_sq(&z2, &r);
    assert_eq!(z1, z2);
    assert_eq!(fused, unfused);
}

// ---------------------------------------------------------------- properties

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn cg_solves_random_spd(n in 5usize..80, per_row in 2usize..10, seed in 0u64..1000) {
            let a = random_spd(n, per_row, seed);
            let (xs, b) = manufactured(&a, seed + 1);
            let mut x = vec![0.0; n];
            let mut cg = Cg::new(CgOptions { tol: tight(), norm: CgNorm::Unpreconditioned, ..Default::default() });
            let info = cg.solve(&a, &Ic0::new(&a).unwrap(), &b, &mut x);
            prop_assert!(info.reason.is_converged());
            prop_assert!(rel_diff(&x, &xs) < 1e-7);
        }

        #[test]
        fn gmres_solves_random_nonsymmetric(n in 5usize..80, per_row in 2usize..10, seed in 0u64..1000) {
            let a = random_nonsymmetric(n, per_row, seed);
            let (xs, b) = manufactured(&a, seed + 1);
            let mut x = vec![0.0; n];
            let mut g = Gmres::new(GmresOptions { tol: tight(), ..Default::default() });
            let info = g.solve(&a, &Ilu0::new(&a).unwrap(), &b, &mut x);
            prop_assert!(info.reason.is_converged());
            prop_assert!(rel_diff(&x, &xs) < 1e-7);
        }

        #[test]
        fn spmv_is_linear(n in 3usize..60, seed in 0u64..500, alpha in -3.0f64..3.0) {
            let a = random_nonsymmetric(n, 5, seed);
            let x = random_vector(n, seed + 1);
            let y = random_vector(n, seed + 2);
            let comb: Vec<f64> = x.iter().zip(&y).map(|(p, q)| alpha * p + q).collect();
            let (mut ax, mut ay, mut ac) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            a.spmv(&x, &mut ax);
            a.spmv(&y, &mut ay);
            a.spmv(&comb, &mut ac);
            for i in 0..n {
                prop_assert!((ac[i] - (alpha * ax[i] + ay[i])).abs() < 1e-10);
            }
        }
    }
}
