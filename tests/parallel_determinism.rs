//! Results must be bit-for-bit identical for every thread count (fixed-chunk reductions).
#![cfg(feature = "parallel")]

use rsolvers::eps::*;
use rsolvers::ksp::*;
use rsolvers::pc::*;
use rsolvers::rng::random_vector;
use rsolvers::sparse::*;
use rsolvers::Identity;

fn with_threads<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

#[test]
fn cg_is_bitwise_identical_across_thread_counts() {
    let a = laplacian_2d(300, 200); // 60k unknowns, above the parallel threshold
    let b = random_vector(a.nrows(), 1);
    let ic = Jacobi::new(&a).unwrap();
    let run = |threads| {
        with_threads(threads, || {
            let mut x = vec![0.0; a.nrows()];
            let mut cg = Cg::new(CgOptions { tol: Tolerances { rtol: 1e-8, ..Default::default() }, record_history: true, ..Default::default() });
            let info = cg.solve(&a, &ic, &b, &mut x);
            (x, info.iterations, info.history)
        })
    };
    let (x1, it1, h1) = run(1);
    for t in [2, 3, 4] {
        let (x, it, h) = run(t);
        assert_eq!(it, it1);
        assert_eq!(h, h1, "residual history differs with {t} threads");
        assert_eq!(x, x1, "solution differs with {t} threads");
    }
}

#[test]
fn gmres_is_bitwise_identical_across_thread_counts() {
    let a = convection_diffusion_2d(250, 250, 1.0, 0.5);
    let b = random_vector(a.nrows(), 2);
    let run = |threads| {
        with_threads(threads, || {
            let mut x = vec![0.0; a.nrows()];
            let mut g = Gmres::new(GmresOptions { tol: Tolerances { rtol: 1e-6, ..Default::default() }, ..Default::default() });
            let info = g.solve(&a, &Identity, &b, &mut x);
            (x, info.iterations)
        })
    };
    let r1 = run(1);
    assert_eq!(run(4), r1);
}

#[test]
fn krylov_schur_is_bitwise_identical_across_thread_counts() {
    let a = laplacian_2d(190, 180);
    let run = |threads| {
        with_threads(threads, || {
            let r = krylov_schur(&a, None, &EpsOptions { tol: 1e-4, ..EpsOptions::new(2) }).unwrap();
            (r.values.clone(), r.vector(0).to_vec(), r.iterations)
        })
    };
    let r1 = run(1);
    assert_eq!(run(4), r1);
    assert_eq!(run(3), r1);
}
