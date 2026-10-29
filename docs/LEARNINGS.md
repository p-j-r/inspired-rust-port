# What the port taught us

Findings from building and testing `femsolve` (2026-10-29). Each item says **how sure** it is,
where the evidence is, and what it means for nyx. Machine for all timings: the 4-vCPU Cascade
Lake class VM of the session (about 8–9 GB/s single-core streaming bandwidth, ±15 % run-to-run
noise). Nothing here has been measured on paul's Windows machines or on a cluster.

## 1. faer's *parallel* multi-RHS sparse solves are pathologically slow  (measured, reproducible)

`faer` 0.24.4, 64 000-unknown 3-D Laplacian, sparse Cholesky, solve of `nrhs` columns
(`examples/solve_scaling.rs`, `harness/results/` has the raw lines):

| nrhs | parallel (global rayon, any thread count ≥ 2) | sequential |
|---|---|---|
| 1 | 50 ms | 48 ms |
| 2 | **1560 ms** | 58 ms |
| 4 | 1540 ms | 66 ms |
| 8 | 1460 ms | 85 ms (10.6 ms per vector) |

The parallel cost is a fixed ~1.5 s per call whatever the width; single-vector solves gain
nothing from threads either. Sparse LU shows a smaller version of the same cliff (nrhs 2: 400 ms
parallel vs 118 ms sequential). **Consequence:** `femsolve::direct::ThreadPolicy` runs solves
sequentially by default and factorisations on the shared pool. **For nyx:** if its block solves go
through faer's global parallel solve with more than one right-hand side, they may be paying this
cost; check with `examples/solve_scaling.rs` on nyx's matrices. Not checked on 8+ threads.

## 2. Multi-RHS solves amortise memory traffic, so blocks can be *faster*  (measured)

With sequential solves the cost per vector falls steeply with block width (complex 2-D LU,
40 000 unknowns: 15.7 → 9.6 → 6.2 → 4.2 ms per vector for widths 1, 2, 4, 8; real 3-D Cholesky:
47.6 → 29.0 → 16.6 → 10.6 ms). In the eigensolver (`harness/results/block_sweep.txt`):

* real SPD 3-D, `ncv = 32`: block 1: 6.3 s; block 2: 3.8 s; block 4: 3.7 s; block 8: 4.4 s.
* complex symmetric 2-D pencil (slow convergence): block 1: 12.6 s (440 matvecs), block 2: 32.8 s
  (1496), block 4: 42.7 s (2544). Blocks need many more vectors there, which outweighs the cheaper solves.

So block size is a per-problem choice (it also finds degenerate eigenvalues, see 6); it is not
a free speed-up. A default of 1 is right until a model says otherwise.

## 3. One faer GEMM call made later, unrelated code 3.6× slower  (measured, root cause very likely)

After a single small `faer::linalg::matmul` call, a baseline-SSE kernel (`dots_block`, complex,
n = 3000, k = 19) went from 58 µs to 209 µs; every phase of an eigensolve ran ~3× slower
(`examples/gemm_side_effect.rs`). Executing `vzeroupper` right after the faer call removes the
effect completely (ratio 0.99). The usual cause is dirty upper vector state left by a wide
(AVX-512) micro-kernel; it matches the earlier `rsolvers` observation that AVX-512 code behaved
badly on this VM class. `femsolve` calls `clear_upper_vector_state()` after every faer call it
makes. **For nyx:** nyx calls faer for dense and sparse work and runs portable (SSE2) code
between the calls; on Intel AVX-512 CPUs it may be hitting this. Cheap test: time a portable
kernel before and after a faer call (the example does exactly that). Not tested on AMD or on
Windows.

## 4. What SLEPc 3.19 actually does (read from the source, BSD-2; reproduced, not copied)

Sources fetched from `gitlab.com/slepc/slepc` tag v3.19.2 (reachable from the sandbox):

* **Start vector.** For generalised Hermitian problems (and positive-definite `B`), `EPSGetStartVector`
  replaces the start vector by `Op·v₀` ("force the vector to be in the range of OP"). This costs
  one extra operator application and removes null-space components of a singular `B`. It was
  the whole reason the first comparison was off by one matvec and by the error estimates.
  `EpsOptions::start_in_range` reproduces it (on by default for weighted Hermitian problems).
* **Convergence test.** `EPSKrylovConvergence` does *not* back-transform the Ritz value for shift-and-invert
  (that branch is only for `STSHIFT`), so the default relative test is `‖r‖/|θ|` in the transformed
  space (`EPSConvergedRelative`). With `θ = 1/(λ−σ)` this guarantees only
  `‖Ax−λBx‖/|λ| ≲ tol/|λ|` for the original problem: loose when `|λ|` is small. Measured: tol 1e-8
  gave a worst residual 5.3e-5; testing the original problem gave 6.4e-10 for 6 more matvecs.
  `EpsOptions::original_shift` / `KrylovSchurShiftInvert` (default) test the original problem; the
  parity runs use SLEPc's test.
* **Sorting** back-transforms the Ritz values through the ST before comparing (`SlepcMap_ST`), so
  `TARGET_MAGNITUDE` on `λ` and `LARGEST_MAGNITUDE` on `θ` agree when the target equals the shift.
* **Purification** is on by default for generalised Hermitian shift-and-invert; not reproduced
  (it changes the returned vectors, not the restart logic). With it off SLEPc's residuals are
  still smaller than ours for the last pairs: unexplained, not investigated further.

## 5. Parity results  (measured; `harness/results/full.md`)

Eleven cases against SLEPc 3.19.2 / PETSc 3.19.6 (real and complex builds) on identical Matrix
Market files and start vector: real generalised Hermitian (with Cholesky), complex Hermitian,
complex symmetric (non-Hermitian path), complex pencil with complex `B`, direct (no shift) Hermitian.
Every non-degenerate case has **identical restart and operator-application counts**, eigenvalues
to 1e-14 or better, and per-cycle error estimates equal to 4–8 digits (checked by hand with
`-eps_monitor_all` against `--history`). The longest identical run is 178 restarts / 1790 matvecs.

## 6. Where parity cannot be asked for  (measured)

* **Degenerate spectra** (square or cubic grids): a single-vector Krylov method sees one vector
  per eigenspace; further copies emerge from rounding, which differs between PETSc's and faer's
  factorisations, so counts differ (3/3 restarts but 40 vs 42 matvecs). Only tier B applies.
  The block variant finds every copy deterministically (`tests/eps.rs`).
* **Long non-normal runs.** The same 200 × 200 absorbing-layer problem took 108, 121 and 128 restarts
  in three configurations of *our own* code that differed only in solve threading, against 100
  for SLEPc; eigenvalues agree to 3e-10. After ~30 restarts of an ill-conditioned problem,
  rounding differences are amplified. Do not use restart counts as a pass criterion there.

## 7. Symbolic reuse is small, buffer reuse is not  (measured, `harness/results/symbolic_share.txt`)

Symbolic analysis is 1.6–7.5 % of the first factorisation (2-D complex LU 62 500 unknowns: 0.034 s
of 0.62 s; 3-D Cholesky 64 000: 0.09 s of 1.23 s; 3-D LU: 0.14 s of 8.5 s). The *first* numeric
factorisation is 30–50 % slower than later ones (8.5 s vs 5.6 s for the 3-D LU), consistent with
allocation and page faults of the factor storage, which `ShiftedFactorization` reuses. So S1 is
worth having for the buffers and the API shape, but it will not change the 4–10 s factorisation
times quoted for the Brillouin example.

## 8. Where the time goes in a shift-and-invert eigensolve  (measured, per-phase timers)

200 × 200 complex pencil (40 000 unknowns, ncv 40): solves 62 %, orthogonalisation 31 %, restarts
1.5 %, dense projected problems 0.3 %. 3-D Cholesky (64 000): solves 99 %. Per solve, faer (sequential)
and PETSc's built-in LU are level (18.8 vs ~17 ms all-in for the complex 2-D case; 54 vs ~49 ms
for the 3-D Cholesky case). So the eigensolver is not the bottleneck; the direct solver's solve
speed is (the port list's own conclusion that faer ≈ PETSc/MUMPS holds for solves too).

## 9. Kernel bandwidth  (measured, `examples/basis_bench.rs`)

Streaming sum: 8–9 GB/s. `sub_block` runs at or above that, the fused update+dot is about
1.5× better than two passes, `dots_block` reaches ~70 % of streaming (portable SSE2 code), so
runtime-dispatched AVX2 kernels could recover at most ~25 % of the dot passes (≤ 5 % of an
eigensolve); not done. The restart multiply `V ← V·Q` was compute-bound (≈ 7 GFlop/s) and is now
a `faer` GEMM per row tile (≈ 29 GFlop/s single thread) — with the side effect of item 3, fixed.
DGKS refinement fired on 100 % of the orthogonalisations in the Hermitian direct (Lanczos-like)
case (1790 of 1791), as `rsolvers` saw (99 %), but on only 62–77 % in shift-and-invert runs
(real generalised, complex Hermitian, complex symmetric, complex pencil; `ncv` 12), so the fused
"update + second dot" pass is wasted in a quarter to a third of the steps there; still a net win.
With `--features parallel` the same kernels reach 20–40 GB/s effective on 4 vCPUs (3.5–4×).

## 10. Smaller facts

* PETSc's built-in sequential LU does **not pivot**; integer-valued shifted matrices hit an exact
  zero pivot (`σ = 1.0` on a 5-point grid, `σ = 2.0` on another) and abort. faer's LU pivots.
  An exactly singular shift gives an infinite operator, reported as `EpsError::Dense(NonFinite)`.
* Complex *symmetric* Matrix Market files use `symmetric` = `a_ij = a_ji` **without** conjugation
  (PML matrices); `hermitian` conjugates. The Rust writer and the C reader both follow this.
* faer exposes Hermitian EVD but not Schur reordering; the complex Schur form, reordering and
  triangular eigenvectors are written here (about 250 lines) and tested against faer's eigenvalues.
* `Which` for complex spectra needs a target-aware key; real PETSc builds cannot take an imaginary shift.
* Environment: `apt` (PETSc/SLEPc real + complex 3.19), crates.io and `gitlab.com` raw files are
  reachable from the sandbox; `sparse.tamu.edu` returns 403 (so no SuiteSparse breadth set yet).

## 11. Krylov linear solvers (K1) — what PETSc really does  (read from PETSc 3.19 `cg.c`, `iterativ.c`; measured)

* GMRES/FGMRES reproduce PETSc's iteration count exactly once the same default stopping rules
  are used (`rtol` 1e-5, `atol` 1e-50, `dtol` 1e4, restart 30, left preconditioning with the
  *preconditioned* residual norm, CGS orthogonalisation with refinement only on request).
  Right preconditioning changes the monitored norm; compare like with like.
* **PETSc's COCG aborts at iteration 2 on complex symmetric systems.** It tests the sign of the real part of
  the bilinear form `(r, z)ᵀ`, which is meaningless for complex values. Our COCG has no such test; the
  option `petsc_sign_test` reproduces the abort. COCG is therefore compared with PETSc only on real SPD.
* The CG update `x += αp; r -= αAp; z = M⁻¹r; (r,z)` is fused with a diagonal preconditioner into
  one pass; the saving shows in memory-bound sizes only.
* FGMRES history in PETSc has duplicated entries at restarts; the judge collapses them.

## 12. AMG (P2) and contour eigensolver (E3)  (measured)

* Aggregation: all-neighbours aggregation gave operator complexity > 2; MIS-2 aggregation with a
  maximum-relative strength measure gives 1.3–1.6 and a grid-independent PCG count.
* For complex *symmetric* matrices the Galerkin product must use the plain transpose `PᵀAP`; `Pᴴ` destroys
  the symmetry that COCG/the symmetric Gauss–Seidel V-cycle rely on. `hermitian: true` uses `Pᴴ`.
* Shifted-Laplacian preconditioning is only judged fairly against an exact shifted LU on the same
  problem and the same Krylov method: easy problems (small n) make anything look good, hard ones
  (n > 100 per side at this frequency) break FGMRES whatever the preconditioner.
* Contour method: a quadrature of 24 nodes leaks eigenvalues just outside the ellipse into the moments at
  ≈ 1e-6, above any sensible `rank_tol`, so `rank ≥ count` and the Rayleigh–Ritz step must be followed by
  region and residual filtering (done: `rejected` counts what was dropped). The stochastic count
  estimate `tr(Vᴴ P V)/block` has a standard error of a few eigenvalues for block ≈ 10 and was *useless*
  on the non-Hermitian pencil (−2 ± 46 for 17 eigenvalues): it is a warning light (found ≪ estimate − 3σ),
  not a count.
* faer 0.24.4 `generalized_eigen` panics for n = 1; special-cased in `faer_glue::generalized_eig_c64`.
* Worker-parallel nodes (one factorisation each) are bit-identical to the serial loop because reductions
  are done in node order.

## 13. Jacobi–Davidson (E2)  (measured while debugging)

* **Harmonic extraction: the right-hand matrix is `Wᴴ B V` with `W = (A − τB)V`, i.e. `(AV)ᴴ(BV) − τ̄(BV)ᴴ(BV)`.**
  Using `(BV)ᴴ(AV)` instead is invisible for `B = I` and Hermitian standard problems but gives
  erratic convergence and wrong "nearest" sets for generalised ones.
* Harmonic Ritz vectors are not orthogonal to a just-locked Ritz vector, so after locking (and at every
  restart) the kept basis columns must be re-orthogonalised against the locked set, with the image
  `A·x_removed` replaced by `λ B x` (Hermitian) or zero (when `A V` is already Schur-deflated).
  Without this, the non-Hermitian standard test did not converge in 400 iterations.
* Standard Rayleigh–Ritz is faster than harmonic for the non-Hermitian complex-symmetric test with a target
  off the real axis (37 vs ~60 outer iterations), so `JdOptions::new` sets `harmonic = hermitian`
  (harmonic for the Hermitian, interior-mode case, standard otherwise). Override it per problem.
* JD is a *local* method: it converges to eigenvalues in the neighbourhood of the target, not the exact
  nearest `nev`. If the nearest set is required use Krylov–Schur shift-and-invert or the contour solver.
* Complex symmetric (non-Hermitian) standard problems: locking must keep a Schur factor (`T`), eigenvectors
  are recovered from its triangular eigenvectors.

## 14. AMS and the Nédélec test generator (P1)  (measured)

* The gradient correction is what matters: at `A = K + 1e-4·M` (gradient kernel almost exact) AMS-PCG needs 10 iterations,
  Jacobi 1078, none 1187; at `β = 1` AMS needs 12–13 iterations on 4³–12³ cells, Jacobi 145 → 682.
* **The cycle must be symmetric for COCG.** Applying the three `Π_c` corrections one after another
  (multiplicatively, x then y then z) made the operator slightly non-symmetric: real CG still converged (11
  iterations) but COCG stagnated at 1e-6 relative. Computing the three corrections from the *same* residual
  (additive, i.e. the block-diagonal vector Laplacian) restores symmetry and costs 2 extra iterations (13).
* The interpolation `Π` needs the *edge vectors*, not only the gradient and interior-node coordinates: under PEC an
  end node of an edge may be removed, and its coordinate is then not recoverable from `G`. `Ams::new` takes
  `edge_vectors` explicitly (the generator provides them; nyx must export them with `G`).
* Solving `K − τM` (indefinite) with AMS built for `K + τM` works for JD with **standard** extraction (3 modes near
  2π² in 33 outer iterations, no factorisation). With **harmonic** extraction JD stalls in the gradient null space
  (θ ≈ 1e-15, relative residual `‖r‖/|θ|` meaningless there): the preconditioner amplifies exactly the null-space components
  and the search space never contains the cavity modes. JD now (a) drops linearly dependent columns on restart instead
  of dividing by zero (this was a `NonFinite` error), (b) confirms the locked set with *recomputed* true
  residuals and reports `converged = false` otherwise (before this it could report garbage as converged).
* Faer's complex generalised eigensolver (`generalized_eigen`) takes minutes for ~300 × 300 pencils with a large
  infinite/zero-eigenvalue cluster; a Cholesky reduction plus Hermitian EVD took milliseconds. Tests use the latter;
  `faer_glue::generalized_eig_c64` is only used for the tiny projected problems.
* The Kuhn mesh is not symmetric under axis permutations: the triple 2π² mode splits by about 8 % at 3 cells per side.
