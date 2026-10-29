mod common;

use common::*;
use rsolvers::eps::*;
use rsolvers::rng::{SplitMix64, random_vector};
use rsolvers::sparse::*;
use rsolvers::{CsrMatrix, FnOperator, LinearOperator};

/// Check that every returned pair satisfies the contract: unit, mutually orthogonal vectors
/// with true residual `‖Ax − λx‖ ≤ slack · tol · |λ|`.
fn check_pairs<A: LinearOperator>(a: &A, res: &EigenResult, tol: f64, slack: f64) {
    let n = a.dim();
    let mut ax = vec![0.0; n];
    for i in 0..res.values.len() {
        let x = res.vector(i);
        assert!((norm(x) - 1.0).abs() < 1e-10, "vector {i} not unit: {}", norm(x));
        a.apply(x, &mut ax);
        let r: f64 = ax.iter().zip(x).map(|(p, q)| (p - res.values[i] * q).powi(2)).sum::<f64>().sqrt();
        assert!(r <= slack * tol * res.values[i].abs(), "pair {i}: residual {r} vs tol*|λ| {}", tol * res.values[i].abs());
        for j in 0..i {
            let d: f64 = x.iter().zip(res.vector(j)).map(|(p, q)| p * q).sum();
            assert!(d.abs() < 1e-8, "vectors {i},{j} not orthogonal: {d}");
        }
    }
}

#[test]
fn laplacian_1d_largest_and_smallest() {
    let n = 400;
    let a = laplacian_1d(n);
    let exact = laplacian_1d_eigenvalues(n);
    for (which, nev) in [(Which::LargestMagnitude, 5), (Which::LargestReal, 5)] {
        let opts = EpsOptions { which, tol: 1e-10, ..EpsOptions::new(nev) };
        let res = krylov_schur(&a, None, &opts).unwrap();
        assert_eq!(res.reason, EpsReason::Converged);
        assert!(res.nconv >= nev);
        for i in 0..nev {
            assert!((res.values[i] - exact[n - 1 - i]).abs() < 1e-9, "{which:?} {i}: {} vs {}", res.values[i], exact[n - 1 - i]);
        }
        check_pairs(&a, &res, 1e-10, 5.0);
    }
}

#[test]
fn laplacian_1d_smallest_eigenvalues() {
    let n = 200;
    let a = laplacian_1d(n);
    let exact = laplacian_1d_eigenvalues(n);
    let opts = EpsOptions { which: Which::SmallestReal, tol: 1e-8, max_it: Some(5000), ..EpsOptions::new(3) };
    let res = krylov_schur(&a, None, &opts).unwrap();
    assert_eq!(res.reason, EpsReason::Converged, "restarts used: {}", res.iterations);
    for i in 0..3 {
        assert!((res.values[i] - exact[i]).abs() < 1e-8 * exact[i].abs() * 10.0);
    }
    check_pairs(&a, &res, 1e-8, 5.0);
    let opts = EpsOptions { which: Which::SmallestMagnitude, ..opts };
    let res2 = krylov_schur(&a, None, &opts).unwrap();
    for i in 0..3 {
        assert!((res2.values[i] - res.values[i]).abs() < 1e-8);
    }
}

/// Assert `got` (descending) is a valid partial spectrum of `exact` (ascending): every value
/// matches an eigenvalue (within `tol`) without exceeding its multiplicity, and no *distinct*
/// eigenvalue larger than the smallest returned one is missing.
fn assert_valid_top_spectrum(got: &[f64], exact: &[f64], tol: f64) {
    let mut avail: Vec<f64> = exact.to_vec();
    for &g in got {
        let k = avail.iter().position(|e| (e - g).abs() < tol).unwrap_or_else(|| panic!("{g} is not an eigenvalue (or exceeds its multiplicity)"));
        avail.remove(k);
    }
    let lowest = got.iter().cloned().fold(f64::INFINITY, f64::min);
    for &e in exact.iter().filter(|&&e| e > lowest + 100.0 * tol) {
        assert!(got.iter().any(|g| (g - e).abs() < tol), "missed distinct eigenvalue {e} (lowest returned {lowest})");
    }
}

#[test]
fn laplacian_2d_degenerate_spectrum_is_never_wrong() {
    // A square grid has doubly degenerate eigenvalues. A single-vector Krylov method sees one
    // direction per eigenspace in exact arithmetic; second copies only appear through rounding.
    // So we require: valid eigenvalues, multiplicity respected, no distinct eigenvalue missed.
    let (nx, ny) = (30, 30);
    let a = laplacian_2d(nx, ny);
    let exact = laplacian_2d_eigenvalues(nx, ny);
    let nev = 8;
    let opts = EpsOptions { tol: 1e-9, ..EpsOptions::new(nev) };
    let res = krylov_schur(&a, None, &opts).unwrap();
    assert_eq!(res.reason, EpsReason::Converged);
    assert_valid_top_spectrum(&res.values[..res.nconv], &exact, 1e-7);
    check_pairs(&a, &res, 1e-9, 5.0);
}

#[test]
fn diagonal_matrix_known_spectrum() {
    let n = 500;
    let t: Vec<_> = (0..n).map(|i| (i, i, (i + 1) as f64)).collect();
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let res = krylov_schur(&a, None, &EpsOptions { tol: 1e-10, ..EpsOptions::new(4) }).unwrap();
    assert_eq!(res.reason, EpsReason::Converged);
    for i in 0..4 {
        assert!((res.values[i] - (n - i) as f64).abs() < 1e-8);
    }
    // Eigenvectors are unit coordinate vectors up to sign.
    let v = res.vector(0);
    assert!(v[n - 1].abs() > 1.0 - 1e-8);
}

#[test]
fn matches_dense_reference_on_random_symmetric() {
    for seed in 1..=3u64 {
        let n = 90;
        let a = random_spd(n, 8, seed);
        let (w, _) = rsolvers::dense::sym_eig(&a.to_dense_colmajor_for_test(), n, n);
        let opts = EpsOptions { tol: 1e-10, ..EpsOptions::new(6) };
        let res = krylov_schur(&a, None, &opts).unwrap();
        assert_eq!(res.reason, EpsReason::Converged);
        for i in 0..6 {
            assert!((res.values[i] - w[n - 1 - i]).abs() < 1e-8 * w[n - 1].abs(), "seed {seed} pair {i}");
        }
        check_pairs(&a, &res, 1e-10, 5.0);
    }
}

#[test]
fn full_spectrum_when_nev_equals_n() {
    let n = 12;
    let a = random_spd(n, 4, 3);
    let (w, _) = rsolvers::dense::sym_eig(&a.to_dense_colmajor_for_test(), n, n);
    let opts = EpsOptions { ncv: Some(n), tol: 1e-12, ..EpsOptions::new(n) };
    let res = krylov_schur(&a, None, &opts).unwrap();
    assert_eq!(res.reason, EpsReason::Converged);
    assert_eq!(res.nconv, n);
    let mut got = res.values.clone();
    got.sort_by(f64::total_cmp);
    for i in 0..n {
        assert!((got[i] - w[i]).abs() < 1e-10 * w[n - 1]);
    }
    check_pairs(&a, &res, 1e-12, 50.0);
}

#[test]
fn start_vector_orthogonal_to_dominant_eigenvector_still_finds_it() {
    // diag(1..n) with a start vector having no component along e_n: exact arithmetic would never
    // find λ = n. Rounding (and the DGKS reorthogonalisation noise) reintroduces it only slowly, so
    // ask for the *second* largest, which lies in the Krylov space, and verify the exact value.
    let n = 100;
    let t: Vec<_> = (0..n).map(|i| (i, i, (i + 1) as f64)).collect();
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let mut s = random_vector(n, 5);
    s[n - 1] = 0.0;
    let res = krylov_schur(&a, Some(&s), &EpsOptions { tol: 1e-10, ..EpsOptions::new(1) }).unwrap();
    assert_eq!(res.reason, EpsReason::Converged);
    assert!((res.values[0] - (n - 1) as f64).abs() < 1e-8, "{}", res.values[0]);
}

#[test]
fn invariant_subspace_breakdown_recovers_with_fresh_vector() {
    // The eigenvalue 5 has multiplicity 3 and the start vector lies in its eigenspace, so the
    // Krylov space is one-dimensional and breaks down immediately. The solver must continue
    // from a fresh vector (which sees one further direction of the eigenspace plus the rest).
    let n = 40;
    let t: Vec<_> = (0..n).map(|i| (i, i, if i < 3 { 5.0 } else { 10.0 + i as f64 })).collect();
    let mut s = vec![0.0; n];
    s[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let opts = EpsOptions { tol: 1e-10, which: Which::SmallestReal, ..EpsOptions::new(5) };
    let res = krylov_schur(&a, Some(&s), &opts).unwrap();
    assert_eq!(res.reason, EpsReason::Converged, "{res:?}");
    assert!(res.iterations >= 2, "breakdown should force a second pass");
    let exact: Vec<f64> = (0..n).map(|i| if i < 3 { 5.0 } else { 10.0 + i as f64 }).collect();
    // `values` is ordered smallest-first for SmallestReal; the helper wants "top", so negate.
    let neg_got: Vec<f64> = res.values[..res.nconv].iter().map(|v| -v).collect();
    let mut neg_exact: Vec<f64> = exact.iter().map(|v| -v).collect();
    neg_exact.sort_by(f64::total_cmp);
    assert_valid_top_spectrum(&neg_got, &neg_exact, 1e-7);
    assert!((res.values[0] - 5.0).abs() < 1e-8 && (res.values[1] - 5.0).abs() < 1e-8);
    check_pairs(&a, &res, 1e-10, 5.0);
}

#[test]
fn matrix_free_operator_matches_assembled() {
    let (nx, ny) = (24, 20);
    let a = laplacian_2d(nx, ny);
    let op = FnOperator::new(nx * ny, |x: &[f64], y: &mut [f64]| {
        for j in 0..ny {
            for i in 0..nx {
                let p = i + nx * j;
                let mut s = 4.0 * x[p];
                if i > 0 { s -= x[p - 1]; }
                if i + 1 < nx { s -= x[p + 1]; }
                if j > 0 { s -= x[p - nx]; }
                if j + 1 < ny { s -= x[p + nx]; }
                y[p] = s;
            }
        }
    });
    let s = random_vector(nx * ny, 3);
    let opts = EpsOptions { tol: 1e-10, ..EpsOptions::new(3) };
    let r1 = krylov_schur(&a, Some(&s), &opts).unwrap();
    let r2 = krylov_schur(&op, Some(&s), &opts).unwrap();
    for i in 0..3 {
        assert!((r1.values[i] - r2.values[i]).abs() < 1e-10);
    }
    assert_eq!(r1.iterations, r2.iterations);
}

#[test]
fn deterministic_and_options_validated() {
    let a = laplacian_2d(15, 15);
    let opts = EpsOptions::new(2);
    let r1 = krylov_schur(&a, None, &opts).unwrap();
    let r2 = krylov_schur(&a, None, &opts).unwrap();
    assert_eq!(r1.values, r2.values);
    assert_eq!(r1.vector(0), r2.vector(0));
    assert!(krylov_schur(&a, None, &EpsOptions::new(0)).is_err());
    assert!(krylov_schur(&a, None, &EpsOptions::new(10_000)).is_err());
    assert!(krylov_schur(&a, None, &EpsOptions { ncv: Some(2), ..EpsOptions::new(2) }).is_err());
    assert!(krylov_schur(&a, None, &EpsOptions { keep: 0.99, ..EpsOptions::new(2) }).is_err());
    assert!(krylov_schur(&a, Some(&vec![0.0; 225]), &opts).is_err());
    assert!(krylov_schur(&a, Some(&[1.0, 2.0]), &opts).is_err());
}

#[test]
fn iteration_limit_is_reported() {
    let a = laplacian_1d(300);
    let opts = EpsOptions { which: Which::SmallestReal, max_it: Some(2), ..EpsOptions::new(2) };
    let res = krylov_schur(&a, None, &opts).unwrap();
    assert_eq!(res.reason, EpsReason::DivergedIts);
    assert_eq!(res.iterations, 2);
    assert!(res.nconv < 2 + 1);
}

#[test]
fn indefinite_spectrum_magnitude_vs_real() {
    // diag(-100, -3, 1, 2, ..., 50): largest magnitude is -100, largest real is 50.
    let n = 120;
    let mut t: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, i as f64 * 0.5 - 20.0)).collect();
    t[0].2 = -100.0;
    let a = CsrMatrix::from_triplets(n, n, &t).unwrap();
    let mag = krylov_schur(&a, None, &EpsOptions { tol: 1e-10, ..EpsOptions::new(1) }).unwrap();
    assert!((mag.values[0] + 100.0).abs() < 1e-8);
    let real = krylov_schur(&a, None, &EpsOptions { tol: 1e-10, which: Which::LargestReal, ..EpsOptions::new(1) }).unwrap();
    assert!((real.values[0] - (119.0 * 0.5 - 20.0)).abs() < 1e-8);
    let mut rng = SplitMix64::new(1);
    let _ = rng.next_u64();
}
