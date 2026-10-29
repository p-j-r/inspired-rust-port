# Benchmarks: `rsolvers` vs. calling PETSc 3.19.6 / SLEPc 3.19.2

**Short answer.** The Rust port is *not* a dramatic speed-up. These solvers are memory-bandwidth
bound, so both implementations hit the same wall. Where Rust wins, it wins by roughly 5–40 %,
and the wins come from fusing memory passes (CG update + preconditioner + dot products;
SpMV + dot) and from a leaner triangular solve, not from the language itself. Where it loses
(Krylov–Schur eigensolves, threaded GMRES/EPS), it loses by 10–150 %.

| Workload | Verdict (best Rust build vs. best PETSc/SLEPc build) |
|---|---|
| CG, large problems (1M unknowns), 1 core | **Rust 10–40 % faster** (p3d_100 SSOR 1.41×, p2d_1024 SSOR 1.32×, no-PC 1.22–1.24×) |
| CG, cache-resident problem (65k unknowns) | parity (0–7 %) |
| CG, random sparsity (spd_rand) | parity (±8 %) |
| GMRES(30) | parity: −22 % to +15 % |
| Krylov–Schur, 1 core | **SLEPc 10–46 % faster** (10–13 % on spd_rand, 31–46 % on the 2-D Laplacian; 1.7× on a 10 ms toy problem) |
| 4 threads (rayon) vs. 4 MPI ranks, CG | parity (±10 %) |
| 4 threads vs. 4 ranks, GMRES | **PETSc 1.3–1.6× faster** |
| 4 threads vs. 4 ranks, Krylov–Schur | **SLEPc 1.1–2.5× faster** |
| Preconditioner application (SSOR) | **Rust 1.5–1.9× faster** on grids, 1.4× on random sparsity |
| Preconditioner application (ILU0 / ICC0, Jacobi) | parity to 1.5× faster |
| Correctness | iteration counts identical to PETSc/SLEPc on every case; eigenvalues agree to ≤ 2e-12 relative |

## Setup

* 4 vCPU Intel Xeon @ 2.8 GHz (L2 1 MiB/core, L3 33 MiB shared), Ubuntu 24.04, gcc 13.3, rustc 1.97.
* Same binary problem files for both sides (`bench/data`, from `rsolvers-bench gen`).
* Each number is the **minimum of repeated solves** after a warm-up (`--reps` / `-reps`, 2–3 for
  the long cases). `OPENBLAS_NUM_THREADS=1`. Parallel runs: `RAYON_NUM_THREADS=4` vs.
  `mpiexec --bind-to core -n 4`.
* Four variants:
  * `petsc`: Ubuntu packages (`libpetsc-real-dev`, `libslepc-real-dev`), `-O2`, generic x86-64.
  * `petsc-native`: PETSc/SLEPc built from source with `-O3 -march=native` (AVX-512 kernels
    *disabled*, see below).
  * `rust`: portable x86-64 build with runtime-dispatched AVX2/AVX-512 kernels.
  * `rust-native`: `RUSTFLAGS=-C target-cpu=native`.
  The "best Rust vs best PETSc" column picks the fastest of each pair.
* Solver settings mirror PETSc defaults (rtol 1e-5, GMRES(30) with left PC and classical
  Gram–Schmidt, CG on the preconditioned norm; EPS: Krylov–Schur, tol 1e-8, ncv = max(2·nev, nev+15)).
* The VM is noisy (±15 % run to run). Treat anything inside ±10 % as a tie.

Full table: [bench/results/full.md](bench/results/full.md); raw data:
[bench/results/full.jsonl](bench/results/full.jsonl).

## Selected results (serial, seconds, lower is better)

| problem | case | iters | PETSc | PETSc native | Rust | Rust native |
|---|---|---|---|---|---|---|
| p2d_1024 (1M) | CG, no PC | 2161 | 36.40 | 35.07 | 28.37 | 28.83 |
| p2d_1024 | CG + SSOR | 781 | 30.11 | 29.72 | 23.55 | 22.52 |
| p2d_1024 | CG + IC(0) | 661 | 21.80 | 20.63 | 18.29 | 18.77 |
| p3d_100 (1M) | CG + SSOR | 85 | 3.80 | 3.85 | 2.69 | 2.94 |
| spd_rand (200k) | CG + IC(0) | 5 | 0.105 | 0.094 | 0.103 | 0.102 |
| cd2d_512 | GMRES + ILU(0) | 98 | 1.28 | 1.11 | 1.34 | 1.66 |
| p2d_256x250 | EPS largest nev=10 | 252 restarts | 6.88 | 6.99 | 10.06 | 19.25 |
| spd_rand | EPS largest nev=10 | 14 restarts | 3.16 | 3.29 | 3.48 | 4.96 |

## Why the numbers look like this

* **CG wins come from fusion.** PETSc runs CG as separate `VecAXPY`, `PCApply`, `VecDot`,
  `VecNorm` calls, each streaming the vectors from memory. `rsolvers` does the update, the
  diagonal/identity preconditioner and all three reductions in one pass, and SpMV returns `p·Ap`
  directly. With ~20 bytes of vector traffic per flop saved, that is worth 10–25 %.
* **SSOR wins come from the triangular kernel.** Both triangles are applied in unit-triangular
  form with the scaling folded into the sweep, which shortens the floating-point dependency
  chain. Level scheduling was tried and was slower on large problems (cache locality), so it
  is not used.
* **GMRES is a tie.** Its cost is dominated by orthogonalization against up to 30 basis
  vectors, a bandwidth-bound dense kernel that both sides implement with similar blocking.
* **Krylov–Schur is where SLEPc is ahead.** Orthogonalization (full CGS plus DGKS refinement,
  which triggers often) dominates; the likely reason (not profiled against SLEPc) is that SLEPc's BLAS-backed
  orthogonalization beats our 4-column blocked kernels once the basis is large relative to cache. The gap is 10–15 % on
  spd_rand and 30–45 % on the 2-D Laplacian, which needs 250+ restarts.
* **Threads:** `rsolvers` parallelizes SpMV and BLAS-1/basis kernels with rayon using fixed
  chunking (bitwise deterministic for any thread count). That matches MPI for CG, but GMRES and
  Krylov–Schur have serial or poorly scaling pieces (probably the small dense work and some basis
  kernels; not profiled in detail) that do not scale as well as PETSc/SLEPc's distributed vectors.

## Caveats and findings worth knowing

* **PETSc with AVX-512 kernels was ~5.7× slower for `MatMult` on this VM**, so the source build
  disables them (the Ubuntu package is unaffected). Comparing against that configuration would
  have made Rust look absurdly good; both PETSc builds above are the fair ones.
* **`target-cpu=native` is not better for Rust here** and is sometimes much worse
  (Krylov–Schur 2×, GMRES 1.5× slower; `mdot16` micro-kernel 5× slower). Runtime dispatch already
  picks AVX-512 in the portable build. Cause not diagnosed (likely a different inlining or
  unrolling choice in the 4-column kernels); use the portable build.
* Rust results are **bitwise reproducible** across thread counts and across ISA tiers (tested);
  PETSc reductions vary with rank count.
* Eigenvalue agreement PETSc/Rust is 1e-14 to 2e-12 (relative) on every benchmark; iteration and
  restart counts are identical.
* Single-vector Lanczos (here and in SLEPc) does not reliably find multiple copies of a
  degenerate eigenvalue; tests account for that.
* Scope: only the solvers listed in the README were ported and compared. Nothing here says
  anything about other PETSc/SLEPc solvers, GPUs, or runs beyond one node, where PETSc/SLEPc
  have capabilities `rsolvers` does not.

## Reproducing

See the README. `python3 bench/run.py run --suite full --out bench/results/full.jsonl` takes
about two hours on 4 cores.
