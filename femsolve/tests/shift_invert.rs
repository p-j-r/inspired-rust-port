mod common;

use common::*;
use femsolve::csr::{laplacian_1d, laplacian_2d, mass_1d};
use femsolve::direct::{Definiteness, ShiftedSolver, Used};
use femsolve::problem::{EigenProblem, KrylovSchurShiftInvert, ShiftInvertEigenSolver, Symmetry};
use femsolve::{C64, Csr, Scalar};

fn fe_spectrum(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|k| {
            let c = (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            6.0 * (1.0 - c) / (2.0 + c)
        })
        .collect()
}

fn nearest(exact: &[f64], sigma: f64, k: usize) -> Vec<f64> {
    let mut v = exact.to_vec();
    v.sort_by(|a, b| (a - sigma).abs().total_cmp(&(b - sigma).abs()));
    v.truncate(k);
    v
}

#[test]
fn interior_shift_real_generalised_uses_lu() {
    let n = 400;
    let p = EigenProblem { a: laplacian_1d(n), b: Some(mass_1d(n, 1.0)), symmetry: Symmetry::Hermitian };
    let exact = fe_spectrum(n);
    let mut s = KrylovSchurShiftInvert::new(&p, Definiteness::General).unwrap();
    s.opts.tol = 1e-12;
    let sigma = 1.7;
    let r = s.solve_near(sigma, 5).unwrap();
    assert!(r.converged);
    let mut got: Vec<f64> = r.values.iter().take(5).map(|v| v.re).collect();
    let mut want = nearest(&exact, sigma, 5);
    got.sort_by(f64::total_cmp);
    want.sort_by(f64::total_cmp);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-10, "{g} vs {w}");
    }
    assert!(r.residuals.iter().take(5).all(|&x| x < 1e-9), "{:?}", r.residuals);
    assert_eq!(r.factor_stats.used, Used::Lu);
}

#[test]
fn shift_below_spectrum_uses_cholesky_and_reuses_the_symbolic_analysis() {
    let n = 300;
    let p = EigenProblem { a: laplacian_1d(n), b: Some(mass_1d(n, 1.0)), symmetry: Symmetry::Hermitian };
    let exact = fe_spectrum(n);
    let mut s = KrylovSchurShiftInvert::new(&p, Definiteness::PositiveDefinite).unwrap();
    s.opts.tol = 1e-12;
    for (sigma, k) in [(0.0, 4usize), (-0.5, 3), (-2.0, 3)] {
        let r = s.solve_near(sigma, k).unwrap();
        assert!(r.converged);
        for i in 0..k {
            assert!((r.values[i].re - exact[i]).abs() < 1e-10, "σ={sigma} {i}: {} vs {}", r.values[i].re, exact[i]);
        }
        assert_eq!(r.factor_stats.used, Used::Cholesky, "σ={sigma}");
    }
    let st = s.factor_stats();
    assert_eq!(st.numeric_count, 3);
    assert_eq!(st.symbolic_count, 1, "symbolic analysis must be done once for three shifts");
    assert_eq!(st.cholesky_fallbacks, 0);
}

#[test]
fn cholesky_falls_back_to_lu_when_the_shift_enters_the_spectrum() {
    let n = 200;
    let p = EigenProblem { a: laplacian_1d(n), b: None, symmetry: Symmetry::Hermitian };
    let exact = laplacian_1d_eigenvalues(n);
    let mut s = KrylovSchurShiftInvert::new(&p, Definiteness::PositiveDefinite).unwrap();
    s.opts.tol = 1e-12;
    // Inside the spectrum, so A − σ is indefinite. (σ = 1 would be exactly λ₆₇ of this matrix.)
    let r = s.solve_near(1.013, 3).unwrap();
    assert!(r.converged);
    let mut got: Vec<f64> = r.values.iter().map(|v| v.re).collect();
    got.sort_by(f64::total_cmp);
    let mut want = nearest(&exact, 1.013, 3);
    want.sort_by(f64::total_cmp);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-10);
    }
    assert_eq!(r.factor_stats.used, Used::Lu);
    assert_eq!(r.factor_stats.cholesky_fallbacks, 1);
}

#[test]
fn complex_symmetric_general_problem_with_complex_shift() {
    // (L + iγ M) x = λ x with M = I: λ = λ_k(L) + iγ. Shift off the real axis.
    let n = 300;
    let gamma = 0.4;
    let a = Csr::axpby(C64::ONE, &laplacian_1d(n).to_complex(), C64::new(0.0, gamma), &Csr::identity(n));
    let p = EigenProblem { a, b: None, symmetry: Symmetry::General };
    let exact = laplacian_1d_eigenvalues(n);
    let mut s = KrylovSchurShiftInvert::new(&p, Definiteness::General).unwrap();
    s.opts.tol = 1e-11;
    let sigma = C64::new(1.0, 0.1);
    let r = s.solve_near(sigma, 4).unwrap();
    assert!(r.converged);
    let mut want: Vec<C64> = exact.iter().map(|&e| C64::new(e, gamma)).collect();
    want.sort_by(|a, b| (a - sigma).norm().total_cmp(&(b - sigma).norm()));
    for i in 0..4 {
        assert!((r.values[i] - want[i]).norm() < 1e-9, "{i}: {} vs {}", r.values[i], want[i]);
    }
    assert!(r.residuals.iter().all(|&x| x < 1e-8), "{:?}", r.residuals);
}

#[test]
fn complex_hermitian_generalised() {
    // Gauge-transformed Laplacian with the (real) mass matrix as B: spectrum as in the real case.
    let n = 200;
    let ph = C64::new(0.6f64.cos(), 0.6f64.sin());
    let mut t = Vec::new();
    for i in 0..n {
        if i > 0 {
            t.push((i, i - 1, -ph));
        }
        t.push((i, i, C64::new(2.0, 0.0)));
        if i + 1 < n {
            t.push((i, i + 1, -ph.conj()));
        }
    }
    let a = Csr::from_triplets(n, n, &t).unwrap();
    // The gauge phase must act on the mass matrix too to keep the exact FE spectrum: use a
    // diagonal-free mass (identity-scaled) instead, i.e. B = 2·I, giving λ = λ_k(L)/2.
    let b = Csr::axpby(C64::new(2.0, 0.0), &Csr::identity(n), C64::ZERO, &Csr::identity(n));
    let p = EigenProblem { a, b: Some(b), symmetry: Symmetry::Hermitian };
    let exact: Vec<f64> = laplacian_1d_eigenvalues(n).iter().map(|e| e / 2.0).collect();
    let mut s = KrylovSchurShiftInvert::new(&p, Definiteness::General).unwrap();
    s.opts.tol = 1e-12;
    let r = s.solve_near(C64::new(0.513, 0.0), 4).unwrap();
    assert!(r.converged);
    let mut got: Vec<f64> = r.values.iter().map(|v| v.re).collect();
    got.sort_by(f64::total_cmp);
    let mut want = nearest(&exact, 0.513, 4);
    want.sort_by(f64::total_cmp);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-10, "{g} vs {w}");
    }
    assert!(s.solve_near(C64::new(0.513, 0.1), 2).is_err(), "Hermitian problems need a real shift");
}

#[test]
fn block_size_two_finds_degenerate_pair_near_the_shift() {
    let (nx, ny) = (20, 20);
    let p = EigenProblem { a: laplacian_2d(nx, ny), b: None, symmetry: Symmetry::Hermitian };
    let exact = laplacian_2d_eigenvalues(nx, ny);
    let mut s = KrylovSchurShiftInvert::new(&p, Definiteness::General).unwrap();
    s.opts.tol = 1e-11;
    s.opts.block_size = 2;
    let sigma = 3.0;
    let r = s.solve_near(sigma, 4).unwrap();
    assert!(r.converged);
    let mut want = nearest(&exact, sigma, 4);
    want.sort_by(f64::total_cmp);
    let mut got: Vec<f64> = r.values.iter().map(|v| v.re).collect();
    got.sort_by(f64::total_cmp);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
    }
}

#[test]
fn solver_solves_shifted_systems_exactly() {
    use femsolve::direct::ShiftedFactorization;
    use femsolve::rng::random_vector;
    let n = 120;
    let (a, b) = (laplacian_1d(n).to_complex(), mass_1d(n, 1.0).to_complex());
    let mut f = ShiftedFactorization::new(&a, Some(&b), Definiteness::General).unwrap();
    for sigma in [C64::new(0.3, 0.2), C64::new(-1.0, 0.0), C64::new(2.5, -0.7)] {
        f.set_shift(sigma).unwrap();
        let x = random_vector::<C64>(n * 3, 5);
        let mut y = x.clone();
        f.solve_block_in_place(&mut y, 3);
        // (A − σB) y = x
        let k = Csr::axpby(C64::ONE, &a, -sigma, &b);
        for c in 0..3 {
            let mut ky = vec![C64::ZERO; n];
            k.spmv(&y[c * n..(c + 1) * n], &mut ky);
            let err: f64 = ky.iter().zip(&x[c * n..(c + 1) * n]).map(|(p, q)| (p - q).norm_sqr()).sum::<f64>().sqrt();
            assert!(err < 1e-10 * norm(&x[c * n..(c + 1) * n]), "σ={sigma}: {err}");
        }
    }
    assert_eq!(f.stats().symbolic_count, 1);
    let _ = <C64 as Scalar>::ONE;
}
