//! `fs-gen <spec> <dir>`: writes a test problem bundle (A.mtx, B.mtx, v0.mtx, meta.txt).
//!
//! Specs (all use a fixed start vector so Rust and SLEPc runs start identically):
//!   lap1d:n  lap2d:nx:ny  lap3d:n          real symmetric, standard
//!   fe1d:n                                 real symmetric, A x = λ M x (exact FE spectrum)
//!   gauge1d:n:phi                          complex Hermitian (gauge-transformed Laplacian)
//!   pml1d:n:gamma                          complex symmetric, A = L + iγI
//!   pml2d:nx:ny:smax                       complex symmetric pencil A x = λ B x with an absorbing
//!                                          layer (B = diag(1 − iσ(x,y))), complex eigenvalues

use femsolve::bundle::Bundle;
use femsolve::csr::{Csr, laplacian_1d, laplacian_2d, laplacian_3d, mass_1d};
use femsolve::rng::random_vector;
use femsolve::{C64, Scalar};
use std::path::Path;

fn num<T: std::str::FromStr>(p: &[&str], i: usize) -> T {
    p.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        eprintln!("bad or missing numeric field {i} in spec");
        std::process::exit(2)
    })
}

fn finish<T: Scalar>(mut b: Bundle<T>, name: &str, dir: &Path, nev: usize, sigma: (f64, f64)) {
    let n = b.a.nrows();
    b.v0 = Some(random_vector::<T>(n, 0xC0FFEE));
    b.set("name", name);
    b.set("nev", nev);
    b.set("sigma_re", sigma.0);
    b.set("sigma_im", sigma.1);
    b.write(dir).expect("write bundle");
    println!("wrote {} (n = {n}, nnz(A) = {}) to {}", name, b.a.nnz(), dir.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: fs-gen <spec> <dir>   (see the module docs of src/bin/fs_gen.rs)");
        std::process::exit(2);
    }
    let parts: Vec<&str> = args[1].split(':').collect();
    let dir = Path::new(&args[2]);
    match parts[0] {
        "lap1d" => {
            let n = num(&parts, 1);
            finish(Bundle::new(laplacian_1d(n), None), &args[1], dir, 6, (0.0, 0.0));
        }
        "lap2d" => {
            let (nx, ny) = (num(&parts, 1), num(&parts, 2));
            finish(Bundle::new(laplacian_2d(nx, ny), None), &args[1], dir, 8, (0.0, 0.0));
        }
        "lap3d" => {
            let n = num(&parts, 1);
            finish(Bundle::new(laplacian_3d(n), None), &args[1], dir, 8, (0.0, 0.0));
        }
        "fe1d" => {
            let n = num(&parts, 1);
            finish(Bundle::new(laplacian_1d(n), Some(mass_1d(n, 1.0))), &args[1], dir, 6, (0.0, 0.0));
        }
        "gauge1d" => {
            let (n, phi): (usize, f64) = (num(&parts, 1), num(&parts, 2));
            let ph = C64::new(phi.cos(), phi.sin());
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
            finish(Bundle::new(Csr::from_triplets(n, n, &t).unwrap(), None), &args[1], dir, 6, (0.0, 0.0));
        }
        "pml1d" => {
            let (n, g): (usize, f64) = (num(&parts, 1), num(&parts, 2));
            let a = Csr::axpby(C64::ONE, &laplacian_1d(n).to_complex(), C64::new(0.0, g), &Csr::identity(n));
            finish(Bundle::new(a, None), &args[1], dir, 6, (0.5, 0.0));
        }
        "pml2d" => {
            let (nx, ny, smax): (usize, usize, f64) = (num(&parts, 1), num(&parts, 2), num(&parts, 3));
            let layer = (nx.min(ny) / 6).max(2) as f64;
            let sigma = |i: usize, j: usize| {
                let d = (i.min(nx - 1 - i).min(j.min(ny - 1 - j))) as f64;
                if d < layer { smax * ((layer - d) / layer).powi(2) } else { 0.0 }
            };
            let a = laplacian_2d(nx, ny).to_complex();
            let t: Vec<_> = (0..nx * ny).map(|p| (p, p, C64::new(1.0, -sigma(p % nx, p / nx)))).collect();
            let b = Csr::from_triplets(nx * ny, nx * ny, &t).unwrap();
            finish(Bundle::new(a, Some(b)), &args[1], dir, 6, (1.0, 0.0));
        }
        other => {
            eprintln!("unknown spec '{other}'");
            std::process::exit(2);
        }
    }
}
