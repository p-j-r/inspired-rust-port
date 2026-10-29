# femsolve

Solver building blocks for finite-element eigenproblems, written once for `f64` and
`Complex<f64>`. It is the staging area for porting the solvers listed in `solver_port_list.md`
into nyx `femcore`; see [`../docs/STATUS.md`](../docs/STATUS.md) for what is done and what was
measured, and [`../docs/NYX_INTEGRATION.md`](../docs/NYX_INTEGRATION.md) for how to move it into nyx.

| Module | What |
|---|---|
| `eps` | thick-restart **Krylov–Schur** with locking: Hermitian (real symmetric, complex Hermitian, B-weighted) and general complex (complex symmetric PML pencils), block size ≥ 1, DGKS refinement, breakdown handling, per-phase timers, per-cycle history |
| `direct` | `(A − σB)⁻¹` by sparse Cholesky/LU with **symbolic reuse** across shifts, Cholesky→LU fallback, explicit thread policy |
| `problem` | the interface models use: `ShiftInvertEigenSolver::solve_near(σ, nev)`, with independently recomputed residuals |
| `mm`, `bundle`, `bin/fs-gen`, `bin/fs-eps` | Matrix Market export/import, problem bundles, CLIs |
| `linop` | `LinearOperator` / `Preconditioner` traits with fusion hooks (`apply_dotc`, `apply_block`, `apply_block_with_bx`) |
| `basis`, `kernels`, `vecops`, `par` | deterministic, cache-blocked multi-vector kernels (fixed row chunks, ordered folds: identical bits for any thread count) |
| `dense` | Hermitian EVD (faer), complex Schur + reordering (own code) |
| `ksp`, `pc` | **CG / COCG / GMRES / FGMRES** (real and complex, PETSc 3.19 defaults), Jacobi preconditioner, `fs-ksp` |
| `amg` | smoothed-aggregation **AMG** for real and complex-symmetric matrices, shifted-Laplacian helper for Helmholtz/PML |
| `contour` | **contour-integral** eigensolver (block Sakurai–Sugiura / CIRR), `fs-contour` |
| `jd` | **Jacobi–Davidson** with harmonic/standard extraction, locking and Schur deflation for interior eigenvalues |
| `snes` | **nonlinear solvers**: Newton (line search, Jacobian lag), JFNK, Picard, Anderson |
| `sweep` | **frequency sweeps** by greedy reduced-order models of `Σ f_j(ω) A_j x = b` |
| `nep`, `toar` | **nonlinear eigenproblems** (residual inverse iteration, Beyn contour method) and the **quadratic eigenproblem** (TOAR with compressed Krylov–Schur restart) |
| `ts` | time steppers: variable-step **BDF 1–5**, **Newmark / generalised-α**, θ-method, **Rosenbrock ROS2** |
| `ams`, `maxwell` | **AMS** preconditioner for lowest-order H(curl) systems (Hiptmair–Xu), and the Nédélec cube test generator |
| `harness/` | SLEPc driver (`c/slepc_eps.c`, real and complex PETSc builds) and `run.py` comparison runner |

```rust
use femsolve::csr::{laplacian_1d, mass_1d};
use femsolve::direct::Definiteness;
use femsolve::problem::{EigenProblem, KrylovSchurShiftInvert, ShiftInvertEigenSolver, Symmetry};

let n = 400;
let problem = EigenProblem { a: laplacian_1d(n), b: Some(mass_1d(n, 1.0)), symmetry: Symmetry::Hermitian };
let mut solver = KrylovSchurShiftInvert::new(&problem, Definiteness::PositiveDefinite).unwrap();
solver.opts.tol = 1e-10;
let r = solver.solve_near(-0.5, 4).unwrap();           // 4 eigenvalues nearest σ = −0.5
assert!(r.converged && r.residuals.iter().all(|&x| x < 1e-9));
let r2 = solver.solve_near(-1.0, 4).unwrap();          // new shift: numeric factorisation only
assert_eq!(solver.factor_stats().symbolic_count, 1);
```

## Tests and checks

```
cargo test --release                                  # unit + integration (analytic spectra, dense references)
cargo test --release --features parallel              # also: bitwise equality for 1 vs 4 threads
cargo clippy --all-targets --features parallel
cargo build --release -p femsolve && make -C femsolve/harness/c
python3 femsolve/harness/run.py run --suite quick --out femsolve/harness/results/quick.jsonl
python3 femsolve/harness/run.py report femsolve/harness/results/quick.jsonl
cargo run --release --example symbolic_share          # how much of a factorisation is symbolic
cargo run --release --example solve_scaling           # per-vector solve cost vs block width
```

The harness needs PETSc/SLEPc 3.19 in real *and* complex builds:
`apt install libpetsc-real-dev libslepc-real-dev libpetsc-complex-dev libslepc-complex-dev libopenmpi-dev pkg-config`.

## Licence

BSD-2-Clause, like the rest of this repository. The SLEPc start-vector and convergence rules
were reproduced after reading SLEPc 3.19's sources (BSD-2-Clause; see `../THIRD_PARTY_NOTICES.md`);
no SLEPc code is copied.
