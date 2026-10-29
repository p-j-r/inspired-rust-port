//! Keeps the README usage snippets compiling and correct.
use rsolvers::eps::{EpsOptions, krylov_schur};
use rsolvers::ksp::{Cg, CgOptions};
use rsolvers::pc::Ic0;
use rsolvers::sparse::laplacian_2d;

#[test]
fn readme_usage() {
    let a = laplacian_2d(64, 64);
    let m = Ic0::new(&a).unwrap();
    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; b.len()];
    let info = Cg::new(CgOptions::default()).solve(&a, &m, &b, &mut x);
    assert!(info.reason.is_converged());

    let res = krylov_schur(&a, None, &EpsOptions::new(4)).unwrap();
    assert_eq!(res.nconv.min(4), 4);
}
