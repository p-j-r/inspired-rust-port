# Benchmarks: `rsolvers` vs. calling PETSc 3.19.6 / SLEPc 3.19.2

**Short answer.** After the fixes described below, `rsolvers` is **level with or ahead of
PETSc/SLEPc on every linear-solver case measured, and level on eigensolves**, in a single
process on one 4-vCPU VM. It is not a different league: these solvers are memory-bandwidth
bound, so both implementations hit the same wall. The wins (up to ~1.4×) come from fusing
memory passes (CG update + preconditioner + dot products; SpMV + dot; Gram-Schmidt) and from a
leaner triangular solve, not from the language itself.

| Workload (best Rust build vs. best PETSc/SLEPc build) | Result |
|---|---|
| CG, 1M unknowns, 1 core | **Rust 1.08–1.38× faster** (SSOR 1.36–1.38×, no PC 1.27×, Jacobi 1.15–1.18×, IC(0) 1.08–1.13×) |
| CG, cache-resident 65k unknowns, and random sparsity | tie (−26 % … +20 %, i.e. noise) |
| GMRES(30), 1 core | **Rust 1.00–1.17× faster** |
| Krylov–Schur, 1 core | tie: 1.01–1.09× slower on the 2-D Laplacian, 1.06–1.12× faster on the random matrix |
| 4 threads vs. 4 MPI ranks, CG | tie (1.01× slower … 1.15× faster) |
| 4 threads vs. 4 MPI ranks, GMRES | **Rust 1.17–1.32× faster** |
| 4 threads vs. 4 MPI ranks, Krylov–Schur | **Rust 1.06–1.11× faster** |
| SSOR application | **Rust 1.5–1.8× faster** on grids, 1.3× on random sparsity |
| ILU(0)/IC(0) application | tie on grids; PETSc ~20 % faster on random sparsity |
| Correctness | iteration counts identical to PETSc/SLEPc in every case; eigenvalues agree to ≤ 2e-12 relative |

What PETSc/SLEPc still have and `rsolvers` does not: MPI across nodes, GPUs, and a far larger
set of solvers, preconditioners and matrix formats. What `rsolvers` has that they do not:
bitwise-reproducible results for any thread count or SIMD level, no C/MPI/BLAS dependency, and
memory safety.

## Setup

* 4 vCPU Intel Xeon @ 2.8 GHz (Cascade Lake class, L2 1 MiB/core, L3 33 MiB shared), Ubuntu
  24.04, gcc 13.3, rustc 1.97.
* Same binary problem files for both sides (`bench/data`, from `rsolvers-bench gen`).
* Each number is the **minimum of repeated solves** after a warm-up (2–3 reps).
  `OPENBLAS_NUM_THREADS=1`. Parallel runs: 4 `rsolvers` threads vs. `mpiexec --bind-to core -n 4`.
* Four variants:
  * `petsc`: Ubuntu packages (`libpetsc-real-dev`, `libslepc-real-dev`), `-O2`, generic x86-64.
  * `petsc-native`: PETSc/SLEPc built from source with `-O3 -march=native` (AVX-512 kernels
    *disabled*, see caveats).
  * `rust`: portable x86-64 build with runtime-dispatched AVX2/AVX-512 kernels.
  * `rust-native`: `RUSTFLAGS=-C target-cpu=native`.
  The last column picks the fastest of each pair.
* Solver settings mirror PETSc defaults (rtol 1e-5, GMRES(30) with left PC and classical
  Gram–Schmidt, CG on the preconditioned norm; EPS: Krylov–Schur, tol 1e-8, ncv = max(2·nev, nev+15)).
* **The VM is noisy (±15 % run to run, more for sub-100 ms cases).** Treat anything inside
  ±10 % as a tie.

Full table: [bench/results/full.md](bench/results/full.md); raw data:
[bench/results/full.jsonl](bench/results/full.jsonl). The first round of results (before the fixes below)
is kept in `bench/results/before/`.

## Selected results (seconds, lower is better)

| problem | case | iters | PETSc | PETSc native | Rust | Rust native |
|---|---|---|---|---|---|---|
| p2d_1024 (1M) | CG, no PC | 2161 | 36.10 | 37.94 | 28.43 | 28.29 |
| p2d_1024 | CG + SSOR | 781 | 31.74 | 31.36 | 23.13 | 24.10 |
| p3d_100 (1M) | CG + SSOR | 85 | 3.95 | 4.02 | 2.87 | 3.23 |
| cd2d_512 | GMRES + SSOR | 159 | 2.23 | 2.12 | 1.83 | 1.84 |
| cd2d_512, 4 threads/ranks | GMRES, no PC | 1085 | 3.17 | 2.94 | 2.22 | 2.48 |
| p2d_256x250 | EPS largest nev=4 | 341 restarts | 6.34 | 6.38 | 7.67 | 6.39 |
| p2d_256x250 | EPS largest nev=10 | 252 restarts | 7.60 | 7.58 | 8.26 | 8.41 |
| spd_rand | EPS largest nev=10 | 14 restarts | 3.80 | 3.74 | 3.33 | 3.82 |
| p2d_256x250, 4 threads/ranks | EPS largest nev=10 | 252 restarts | 2.29 | 2.24 | 2.11 | 2.24 |

## What was slower in the first round, and why

The first round (see `bench/results/before/`) had Rust behind on eigensolves (10–46 % serially,
up to 2.5× with 4 threads), on threaded GMRES (1.3–1.6×), and had the `target-cpu=native` build up to 2×
slower than the portable one. Root causes, found with assembly inspection, `callgrind` and
per-phase timers:

1. **A hot kernel compiled to scalar code.** The 4-column dot-product kernel behind Gram-Schmidt
   (`Vᵀw`) used eight-lane accumulator arrays; LLVM's SLP vectorizer left them as 32 scalar
   locals spilled to the stack. Under `-C target-cpu=native` this made the kernel ~5× slower
   (and the whole of Krylov–Schur and GMRES 1.5–2× slower); in the portable build it was merely
   slow. It is now written with explicit AVX2 / AVX-512 / SSE2 intrinsics. The summation order is
   unchanged, so results are bit-identical across tiers (unit-tested). This is also why `rust-native`
   is no longer systematically slower.
2. **Too many passes over the Krylov basis.** Krylov–Schur orthogonalization is memory-bound and
   the DGKS refinement fires on ~99 % of steps (Lanczos vectors lose most of their norm), so each
   step ran four sweeps over `V` (dot, update, dot, update). The second dot is now computed in the
   same row block as the first update, while that block of `V` is in L2: three sweeps. The
   restart multiply (`V ← V·Q`) also now uses the vectorised kernels (it was SSE2 scalar loops).
3. **Thread wake-up latency.** Parallel regions in a solver are short (a dot product, a
   Gram-Schmidt pass), and waking a sleeping rayon worker cost **50–150 µs** per region on this VM
   (measured with an empty `broadcast`), as much as the work. With ~6 regions per Lanczos step
   that dominated and 4 threads gave only 1.8× over one. `par` now uses a small persistent team
   that polls for ~400 µs between jobs (≈1 µs dispatch) and then parks; rayon is no longer a
   dependency. MPI ranks spin in the same way, which is why PETSc/SLEPc did not suffer.
   Chunk boundaries and fold order are unchanged, so results still do not depend on the thread count.

Things that were tried and did **not** help: shrinking the row block of the basis kernels, giving
each thread a fixed row range (cache affinity), and level-scheduled triangular solves (they hurt
on large problems).

## Caveats and findings worth knowing

* **PETSc configured with AVX-512 kernels was ~5.7× slower for `MatMult` on this VM**, so the
  source build disables them (the Ubuntu package is unaffected). Both PETSc builds above are the
  fair ones.
* **Spinning has a cost**: the team burns CPU for up to ~400 µs after the last parallel region and
  assumes the cores are yours. On an oversubscribed machine, set `RSOLVERS_THREADS` lower.
* `rsolvers` never uses FMA, so that results are identical on every CPU. FMA would speed up the dense
  kernels but break that guarantee.
* Remaining known gaps: ILU(0)/IC(0) application on matrices with random sparsity (~20 % slower
  than PETSc), and a 1–9 % deficit on Krylov–Schur for the 2-D Laplacian. The dense eigensolver
  inside Krylov–Schur is a cyclic Jacobi method and would dominate for very small problems.
* Eigenvalue agreement PETSc/Rust is 1e-14 to 2e-12 (relative) on every benchmark; iteration and
  restart counts are identical.
* Single-vector Lanczos (here and in SLEPc) does not reliably find multiple copies of a
  degenerate eigenvalue; tests account for that.
* Scope: only the solvers listed in the README were ported and compared; nothing here says
  anything about other solvers, GPUs, or runs beyond one node.

## Reproducing

See the README. `python3 bench/run.py run --suite full --out bench/results/full.jsonl` takes
about two hours on 4 cores; `--suite recheck` covers the previously slower cases in ~30 minutes.
