//! Benchmark driver: generates problem files shared with the C/PETSc/SLEPc harnesses and
//! times the Rust solvers on them. Output is one JSON object per line.

use rsolvers::eps::{EpsOptions, Which, krylov_schur};
use rsolvers::ksp::*;
use rsolvers::pc::*;
use rsolvers::rng::random_vector;
use rsolvers::sparse::*;
use rsolvers::{CsrMatrix, Identity, LinearOperator, Preconditioner, vecops};
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"RSBENCH1";

struct Problem {
    a: CsrMatrix,
    b: Vec<f64>,
    v0: Vec<f64>,
}

fn write_problem(path: &str, p: &Problem) -> std::io::Result<()> {
    let n = p.a.nrows();
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(p.a.nnz() as u64).to_le_bytes())?;
    for &x in p.a.indptr() {
        w.write_all(&(x as i32).to_le_bytes())?;
    }
    for &x in p.a.indices() {
        w.write_all(&(x as i32).to_le_bytes())?;
    }
    for &x in p.a.values().iter().chain(&p.b).chain(&p.v0) {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}

fn read_problem(path: &str) -> Problem {
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert_eq!(&buf[..8], MAGIC);
    let rd = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()) as usize;
    let (n, nnz) = (rd(8), rd(16));
    let mut o = 24;
    let i32s = |o: &mut usize, cnt: usize| -> Vec<i32> {
        let v = (0..cnt).map(|k| i32::from_le_bytes(buf[*o + 4 * k..*o + 4 * k + 4].try_into().unwrap())).collect();
        *o += 4 * cnt;
        v
    };
    let f64s = |o: &mut usize, cnt: usize| -> Vec<f64> {
        let v = (0..cnt).map(|k| f64::from_le_bytes(buf[*o + 8 * k..*o + 8 * k + 8].try_into().unwrap())).collect();
        *o += 8 * cnt;
        v
    };
    let indptr: Vec<usize> = i32s(&mut o, n + 1).into_iter().map(|x| x as usize).collect();
    let indices: Vec<u32> = i32s(&mut o, nnz).into_iter().map(|x| x as u32).collect();
    let values = f64s(&mut o, nnz);
    let b = f64s(&mut o, n);
    let v0 = f64s(&mut o, n);
    Problem { a: CsrMatrix::try_new(n, n, indptr, indices, values).unwrap(), b, v0 }
}

fn make_matrix(spec: &str) -> CsrMatrix {
    // poisson1d:N | poisson2d:NXxNY | poisson3d:N | convdiff2d:NXxNY:BX:BY | randspd:N:PERROW
    let mut it = spec.split(':');
    let kind = it.next().unwrap();
    let dims = |s: &str| -> Vec<usize> { s.split('x').map(|v| v.parse().unwrap()).collect() };
    match kind {
        "poisson1d" => laplacian_1d(it.next().unwrap().parse().unwrap()),
        "poisson2d" => {
            let d = dims(it.next().unwrap());
            laplacian_2d(d[0], *d.get(1).unwrap_or(&d[0]))
        }
        "poisson3d" => {
            let d = dims(it.next().unwrap());
            laplacian_3d(d[0], *d.get(1).unwrap_or(&d[0]), *d.get(2).unwrap_or(&d[0]))
        }
        "convdiff2d" => {
            let d = dims(it.next().unwrap());
            let bx: f64 = it.next().unwrap().parse().unwrap();
            let by: f64 = it.next().unwrap().parse().unwrap();
            convection_diffusion_2d(d[0], *d.get(1).unwrap_or(&d[0]), bx, by)
        }
        "randspd" => random_spd(it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap(), 42),
        _ => panic!("unknown problem {spec}"),
    }
}

struct Args {
    pos: Vec<String>,
    kv: HashMap<String, String>,
}

impl Args {
    fn parse(raw: &[String]) -> Self {
        let (mut pos, mut kv) = (Vec::new(), HashMap::new());
        let mut i = 0;
        while i < raw.len() {
            if let Some(k) = raw[i].strip_prefix("--") {
                kv.insert(k.to_string(), raw[i + 1].clone());
                i += 2;
            } else {
                pos.push(raw[i].clone());
                i += 1;
            }
        }
        Self { pos, kv }
    }
    fn get<T: std::str::FromStr>(&self, k: &str, default: T) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.kv.get(k).map(|v| v.parse().unwrap()).unwrap_or(default)
    }
    fn str(&self, k: &str, default: &str) -> String {
        self.kv.get(k).cloned().unwrap_or_else(|| default.to_string())
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn fmt_f(v: f64) -> String {
    if v.is_finite() { format!("{v:.17e}") } else { "null".into() }
}

fn run_in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

fn true_rel_residual(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.spmv(x, &mut ax);
    let r: f64 = ax.iter().zip(b).map(|(p, q)| (q - p) * (q - p)).sum::<f64>().sqrt();
    r / vecops::norm2(b)
}

fn cmd_ksp(args: &Args) {
    let p = read_problem(&args.pos[1]);
    let solver = args.str("solver", "cg");
    let pc = args.str("pc", "none");
    let reps: usize = args.get("reps", 5);
    let threads: usize = args.get("threads", 1);
    let tol = Tolerances { rtol: args.get("rtol", 1e-5), atol: 1e-50, dtol: 1e4, max_it: args.get("maxit", 10_000) };
    let restart: usize = args.get("restart", 30);
    let (a, b) = (&p.a, &p.b);
    let n = a.nrows();

    let out = run_in_pool(threads, || {
        let t0 = Instant::now();
        let m: Box<dyn Preconditioner + '_> = match pc.as_str() {
            "none" => Box::new(Identity),
            "jacobi" => Box::new(Jacobi::new(a).unwrap()),
            "sor" => Box::new(Sor::new(a, 1.0).unwrap()),
            "ilu" => Box::new(Ilu0::new(a).unwrap()),
            "icc" => Box::new(Ic0::new(a).unwrap()),
            other => panic!("unknown pc {other}"),
        };
        let setup = t0.elapsed().as_secs_f64();
        let mut x = vec![0.0; n];
        let mut cg = Cg::new(CgOptions { tol, ..Default::default() });
        let mut gm = Gmres::new(GmresOptions { tol, restart, ..Default::default() });
        let mut once = |x: &mut Vec<f64>| -> SolveInfo {
            match solver.as_str() {
                "cg" => cg.solve(a, &*m, b, x),
                "gmres" => gm.solve(a, &*m, b, x),
                other => panic!("unknown solver {other}"),
            }
        };
        let _ = once(&mut x); // warm-up (page-faults the work vectors)
        let mut times = Vec::new();
        let mut info = None;
        for _ in 0..reps {
            let t = Instant::now();
            let i = once(&mut x);
            times.push(t.elapsed().as_secs_f64());
            info = Some(i);
        }
        let info = info.unwrap();
        let res = true_rel_residual(a, &x, b);
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let med = median(&mut times);
        format!(
            "{{\"impl\":\"rust\",\"kind\":\"ksp\",\"solver\":\"{solver}\",\"pc\":\"{pc}\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"setup_s\":{},\"solve_min_s\":{},\"solve_med_s\":{},\"iterations\":{},\"reason\":\"{:?}\",\"true_rel_resid\":{},\"matvecs\":{}}}",
            a.nnz(), fmt_f(setup), fmt_f(min), fmt_f(med), info.iterations, info.reason, fmt_f(res), info.matvecs
        )
    });
    println!("{out}");
}

fn cmd_eps(args: &Args) {
    let p = read_problem(&args.pos[1]);
    let reps: usize = args.get("reps", 3);
    let threads: usize = args.get("threads", 1);
    let nev: usize = args.get("nev", 1);
    let ncv: Option<usize> = args.kv.get("ncv").map(|v| v.parse().unwrap());
    let which = match args.str("which", "largest").as_str() {
        "largest" => Which::LargestMagnitude,
        "smallest" => Which::SmallestMagnitude,
        "largest_real" => Which::LargestReal,
        "smallest_real" => Which::SmallestReal,
        other => panic!("unknown which {other}"),
    };
    let opts = EpsOptions { ncv, tol: args.get("tol", 1e-8), which, max_it: args.kv.get("maxit").map(|v| v.parse().unwrap()), ..EpsOptions::new(nev) };
    let a = &p.a;
    let n = a.nrows();
    let out = run_in_pool(threads, || {
        let _ = krylov_schur(a, Some(&p.v0), &opts).unwrap();
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..reps {
            let t = Instant::now();
            let r = krylov_schur(a, Some(&p.v0), &opts).unwrap();
            times.push(t.elapsed().as_secs_f64());
            last = Some(r);
        }
        let r = last.unwrap();
        // True residuals of the returned pairs.
        let mut ax = vec![0.0; n];
        let mut resid = Vec::new();
        for i in 0..r.values.len().min(nev) {
            a.apply(r.vector(i), &mut ax);
            let rr: f64 = ax.iter().zip(r.vector(i)).map(|(p, q)| (p - r.values[i] * q).powi(2)).sum::<f64>().sqrt();
            resid.push(rr / r.values[i].abs());
        }
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let med = median(&mut times);
        let list = |v: &[f64]| v.iter().map(|x| fmt_f(*x)).collect::<Vec<_>>().join(",");
        format!(
            "{{\"impl\":\"rust\",\"kind\":\"eps\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"nev\":{nev},\"which\":\"{}\",\"solve_min_s\":{},\"solve_med_s\":{},\"iterations\":{},\"matvecs\":{},\"nconv\":{},\"reason\":\"{:?}\",\"values\":[{}],\"true_rel_resid\":[{}]}}",
            a.nnz(), args.str("which", "largest"), fmt_f(min), fmt_f(med), r.iterations, r.matvecs, r.nconv, r.reason, list(&r.values[..r.values.len().min(nev)]), list(&resid)
        )
    });
    println!("{out}");
}

/// Micro-benchmarks of the building blocks.
fn cmd_micro(args: &Args) {
    let p = read_problem(&args.pos[1]);
    let reps: usize = args.get("reps", 50);
    let threads: usize = args.get("threads", 1);
    let a = &p.a;
    let n = a.nrows();
    let out = run_in_pool(threads, || {
        let x = random_vector(n, 1);
        let mut y = vec![0.0; n];
        let time = |f: &mut dyn FnMut()| -> f64 {
            f();
            let mut ts = Vec::new();
            for _ in 0..reps {
                let t = Instant::now();
                f();
                ts.push(t.elapsed().as_secs_f64());
            }
            ts.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        let spmv = time(&mut || a.spmv(&x, &mut y));
        let spmv_dot = time(&mut || {
            std::hint::black_box(a.spmv_dot(&x, &mut y));
        });
        let dot = time(&mut || {
            std::hint::black_box(vecops::dot(&x, &p.b));
        });
        let mut z = x.clone();
        let axpy = time(&mut || vecops::axpy(0.5, &p.b, &mut z));
        let sym = a.is_symmetric(0.0);
        let mut pcs: Vec<(&str, f64)> = Vec::new();
        {
            let jac = Jacobi::new(a).unwrap();
            pcs.push(("pc_jacobi_s", time(&mut || jac.apply(&x, &mut y))));
            let sor = Sor::new(a, 1.0).unwrap();
            pcs.push(("pc_sor_s", time(&mut || sor.apply(&x, &mut y))));
            let ilu = Ilu0::new(a).unwrap();
            pcs.push(("pc_ilu_s", time(&mut || ilu.apply(&x, &mut y))));
            if sym {
                let icc = Ic0::new(a).unwrap();
                pcs.push(("pc_icc_s", time(&mut || icc.apply(&x, &mut y))));
            }
        }
        let mut basis = rsolvers::basis::Basis::new(n, 16);
        for j in 0..16 {
            basis.col_mut(j).copy_from_slice(&random_vector(n, j as u64 + 10));
        }
        let mut h = vec![0.0; 16];
        let mdot = time(&mut || basis.dots(16, &x, &mut h));
        let mut w = x.clone();
        let hh = vec![1e-9; 16];
        let maxpy = time(&mut || basis.sub_combination(16, &hh, &mut w));
        let pcs: String = pcs.iter().map(|(k, v)| format!(",\"{k}\":{}", fmt_f(*v))).collect();
        format!(
            "{{\"impl\":\"rust\",\"kind\":\"micro\",\"n\":{n},\"nnz\":{},\"threads\":{threads},\"spmv_s\":{},\"spmv_dot_s\":{},\"dot_s\":{},\"axpy_s\":{}{pcs},\"mdot16_s\":{},\"maxpy16_s\":{}}}",
            a.nnz(), fmt_f(spmv), fmt_f(spmv_dot), fmt_f(dot), fmt_f(axpy), fmt_f(mdot), fmt_f(maxpy)
        )
    });
    println!("{out}");
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = Args::parse(&raw);
    match args.pos.first().map(String::as_str) {
        Some("gen") => {
            // gen <out.bin> <problem-spec>
            let a = make_matrix(&args.pos[2]);
            let n = a.nrows();
            let p = Problem { b: random_vector(n, 1234), v0: random_vector(n, 5678), a };
            write_problem(&args.pos[1], &p).unwrap();
            eprintln!("wrote {} (n = {n}, nnz = {})", args.pos[1], p.a.nnz());
        }
        Some("ksp") => cmd_ksp(&args),
        Some("eps") => cmd_eps(&args),
        Some("micro") => cmd_micro(&args),
        _ => {
            eprintln!("usage: rsolvers-bench gen <out.bin> <spec> | ksp|eps|micro <file.bin> [--opt val]...");
            std::process::exit(2);
        }
    }
}
