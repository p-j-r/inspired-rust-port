#![allow(clippy::needless_range_loop)]
mod common;

use common::*;
use femsolve::csr::{laplacian_1d, laplacian_2d};
use femsolve::direct::{Definiteness, ShiftedFactorization, ShiftedSolver};
use femsolve::ksp::{
    Cg, CgKind, CgNorm, CgOptions, ConvergedReason, Gmres, GmresOptions, Orthogonalization, PcSide, Tolerances,
};
use femsolve::linop::{Identity, LinearOperator, Preconditioner};
use femsolve::pc::Jacobi;
use femsolve::rng::random_vector;
use femsolve::{C64, Csr, Scalar};
use std::cell::Cell;

/// Helmholtz-like complex *symmetric* matrix: 2-D Laplacian − k² + iσ, with a non-constant
/// absorption so Jacobi is not trivially exact.
fn helmholtz(nx: usize, ny: usize, k2: f64, sigma: f64) -> Csr<C64> {
    let l = laplacian_2d(nx, ny).to_complex();
    let t: Vec<_> = (0..nx * ny)
        .map(|p| {
            let (i, j) = (p % nx, p / nx);
            let ramp = 1.0 + 0.5 * (i as f64 / nx as f64) + 0.25 * (j as f64 / ny as f64);
            (p, p, C64::new(-k2, sigma * ramp))
        })
        .collect();
    Csr::axpby(C64::ONE, &l, C64::ONE, &Csr::from_triplets(nx * ny, nx * ny, &t).unwrap())
}

fn direct_solution(a: &Csr<C64>, b: &[C64]) -> Vec<C64> {
    let mut f = ShiftedFactorization::new(a, None, Definiteness::General).unwrap();
    f.set_shift(C64::ZERO).unwrap();
    let mut x = b.to_vec();
    f.solve_block_in_place(&mut x, 1);
    x
}

fn true_rel_residual<T: Scalar>(a: &Csr<T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::ZERO; b.len()];
    a.spmv(x, &mut ax);
    let r: f64 = ax.iter().zip(b).map(|(p, q)| (*q - *p).modulus_sq()).sum::<f64>().sqrt();
    r / norm(b)
}

fn tight() -> Tolerances {
    Tolerances { rtol: 1e-10, ..Tolerances::default() }
}

#[test]
fn cocg_solves_complex_symmetric_systems() {
    let a = helmholtz(40, 40, 0.7, 0.6);
    assert!(a.is_symmetric(0.0) && !a.is_hermitian(1e-3));
    let b = random_vector::<C64>(a.nrows(), 1);
    let exact = direct_solution(&a, &b);
    for use_jacobi in [false, true] {
        let mut x = vec![C64::ZERO; b.len()];
        let opts = CgOptions { tol: tight(), kind: CgKind::Symmetric, record_history: true, ..Default::default() };
        let info = if use_jacobi {
            Cg::new(opts).solve(&a, &Jacobi::new(&a), &b, &mut x)
        } else {
            Cg::new(opts).solve(&a, &Identity, &b, &mut x)
        };
        assert!(info.reason.is_converged(), "jacobi={use_jacobi}: {:?} after {}", info.reason, info.iterations);
        assert_eq!(info.history.len(), info.iterations + 1);
        assert!(true_rel_residual(&a, &x, &b) < 1e-8, "{}", true_rel_residual(&a, &x, &b));
        let err = x.iter().zip(&exact).map(|(p, q)| (*p - *q).modulus_sq()).sum::<f64>().sqrt() / norm(&exact);
        assert!(err < 1e-7, "error {err}");
    }
}

#[test]
fn cocg_equals_cg_on_real_symmetric_and_hermitian_cg_works_on_complex_hermitian() {
    // Real scalars: both kinds are the same algorithm, bit for bit.
    let a = laplacian_2d(30, 30);
    let b = random_vector::<f64>(900, 3);
    let run = |kind| {
        let mut x = vec![0.0; 900];
        let info = Cg::new(CgOptions { kind, record_history: true, ..Default::default() }).solve(
            &a,
            &Jacobi::new(&a),
            &b,
            &mut x,
        );
        (x, info.history)
    };
    assert_eq!(run(CgKind::Hermitian), run(CgKind::Symmetric));
    // Complex Hermitian positive definite: the gauge Laplacian plus 0.1.
    let n = 400;
    let ph = C64::new(0.9f64.cos(), 0.9f64.sin());
    let mut t = vec![];
    for i in 0..n {
        if i > 0 {
            t.push((i, i - 1, -ph));
        }
        t.push((i, i, C64::new(2.1, 0.0)));
        if i + 1 < n {
            t.push((i, i + 1, -ph.conj()));
        }
    }
    let h = Csr::from_triplets(n, n, &t).unwrap();
    let b = random_vector::<C64>(n, 4);
    let mut x = vec![C64::ZERO; n];
    let info = Cg::new(CgOptions { tol: tight(), ..Default::default() }).solve(&h, &Jacobi::new(&h), &b, &mut x);
    assert!(info.reason.is_converged());
    assert!(true_rel_residual(&h, &x, &b) < 1e-8);
}

#[test]
fn hermitian_cg_detects_an_indefinite_matrix() {
    let n = 50;
    let t: Vec<_> = (0..n).map(|i| (i, i, if i % 2 == 0 { 1.0 } else { -1.0 })).collect();
    let a = Csr::from_triplets(n, n, &t).unwrap();
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    let info = Cg::new(CgOptions::default()).solve(&a, &Identity, &b, &mut x);
    assert!(
        matches!(info.reason, ConvergedReason::DivergedIndefiniteMat | ConvergedReason::DivergedBreakdown),
        "{:?}",
        info.reason
    );
}

#[test]
fn gmres_all_variants_solve_a_complex_symmetric_system() {
    let a = helmholtz(30, 30, 3.0, 0.4); // indefinite real part: COCG is risky, GMRES is not
    let b = random_vector::<C64>(a.nrows(), 2);
    let exact = direct_solution(&a, &b);
    let jac = Jacobi::new(&a);
    for (side, orthog, flexible) in [
        (PcSide::Left, Orthogonalization::ClassicalGramSchmidt, false),
        (PcSide::Left, Orthogonalization::ModifiedGramSchmidt, false),
        (PcSide::Right, Orthogonalization::ClassicalGramSchmidtIfNeeded, false),
        (PcSide::Right, Orthogonalization::ClassicalGramSchmidt, true),
    ] {
        let opts = GmresOptions {
            tol: tight(),
            restart: 60,
            side,
            orthog,
            flexible,
            record_history: true,
            ..Default::default()
        };
        let mut x = vec![C64::ZERO; b.len()];
        let info = Gmres::new(opts).solve(&a, &jac, &b, &mut x);
        assert!(
            info.reason.is_converged(),
            "{side:?} {orthog:?} flexible={flexible}: {:?} after {}",
            info.reason,
            info.iterations
        );
        assert!(
            info.history.windows(2).all(|w| w[1] <= w[0] * (1.0 + 1e-12)),
            "GMRES residual must not increase within a cycle"
        );
        let err = x.iter().zip(&exact).map(|(p, q)| (*p - *q).modulus_sq()).sum::<f64>().sqrt() / norm(&exact);
        assert!(err < 1e-6, "{side:?} {orthog:?} flexible={flexible}: error {err}");
    }
}

/// A preconditioner that is a *different* operator at every application: a damped Jacobi whose
/// damping changes from call to call. Plain right-preconditioned GMRES is wrong for this;
/// FGMRES is exact because it stores the preconditioned vectors.
struct Wobbly<'a> {
    jac: &'a Jacobi<C64>,
    calls: Cell<usize>,
}

impl Preconditioner<C64> for Wobbly<'_> {
    fn apply(&self, r: &[C64], z: &mut [C64]) {
        let c = self.calls.get();
        self.calls.set(c + 1);
        self.jac.apply(r, z);
        let damp = 0.6 + 0.4 * ((c % 7) as f64 / 6.0);
        for v in z.iter_mut() {
            *v = v.scale(damp);
        }
    }
}

#[test]
fn fgmres_handles_a_preconditioner_that_changes_every_call() {
    let a = helmholtz(30, 30, 1.0, 0.5);
    let b = random_vector::<C64>(a.nrows(), 9);
    let jac = Jacobi::new(&a);
    let opts =
        |flexible| GmresOptions { tol: tight(), restart: 80, side: PcSide::Right, flexible, ..Default::default() };
    let mut x = vec![C64::ZERO; b.len()];
    let pc = Wobbly { jac: &jac, calls: Cell::new(0) };
    let info = Gmres::new(opts(true)).solve(&a, &pc, &b, &mut x);
    assert!(info.reason.is_converged(), "{:?}", info.reason);
    assert!(true_rel_residual(&a, &x, &b) < 1e-8, "{}", true_rel_residual(&a, &x, &b));
    // The non-flexible version trusts a linear preconditioner and ends with a wrong solution.
    let mut x2 = vec![C64::ZERO; b.len()];
    let pc2 = Wobbly { jac: &jac, calls: Cell::new(0) };
    let _ = Gmres::new(opts(false)).solve(&a, &pc2, &b, &mut x2);
    assert!(
        true_rel_residual(&a, &x2, &b) > 1e-6,
        "plain right-preconditioned GMRES should not survive a varying preconditioner"
    );
}

#[test]
fn edge_cases_zero_rhs_exact_guess_and_norms() {
    let a = laplacian_1d(100).to_complex();
    let b = vec![C64::ZERO; 100];
    let mut x = random_vector::<C64>(100, 1);
    let info = Gmres::new(GmresOptions::default()).solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged() && x.iter().all(|v| *v == C64::ZERO));
    let info = Cg::new(CgOptions::default()).solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged() && info.iterations == 0);
    // Exact initial guess: converges immediately.
    let b = random_vector::<C64>(100, 6);
    let mut x = direct_solution(&a, &b);
    let info = Cg::new(CgOptions {
        initial_guess_nonzero: true,
        tol: Tolerances { rtol: 1e-6, ..Default::default() },
        ..Default::default()
    })
    .solve(&a, &Identity, &b, &mut x);
    assert!(info.reason.is_converged() && info.iterations <= 1, "{info:?}");
    for norm in [CgNorm::Preconditioned, CgNorm::Unpreconditioned, CgNorm::Natural] {
        let mut x = vec![C64::ZERO; 100];
        let info =
            Cg::new(CgOptions { norm, tol: tight(), ..Default::default() }).solve(&a, &Jacobi::new(&a), &b, &mut x);
        assert!(info.reason.is_converged(), "{norm:?}");
    }
}

#[test]
fn operator_fusion_hooks_are_exercised_and_exact() {
    // Fused SpMV+dot and unfused agree bit for bit for both dot kinds on a complex matrix.
    let a = helmholtz(60, 50, 0.5, 0.3);
    let x = random_vector::<C64>(a.nrows(), 8);
    let (mut y1, mut y2, mut y3) = (vec![C64::ZERO; a.nrows()], vec![C64::ZERO; a.nrows()], vec![C64::ZERO; a.nrows()]);
    let dc = a.apply_dotc(&x, &mut y1);
    a.apply(&x, &mut y2);
    assert_eq!(dc, femsolve::vecops::dotc(&x, &y2));
    let du = a.apply_dotu(&x, &mut y3);
    assert_eq!(du, femsolve::vecops::dotu(&x, &y2));
    assert_eq!(y1, y2);
}
