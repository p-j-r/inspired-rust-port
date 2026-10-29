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

## 15. Time stepping (T1)  (measured)

* BDF in the quasi-constant-step (backward-difference) form needs only two small matrices `R`, `U` to change the
  step; the unit tests (factor 1 is the identity; a cubic polynomial history is reproduced on a new grid) catch
  every indexing slip. Written from the Shampine–Reichelt paper; no library code was read.
* A *linear* problem needs no Newton iteration: `(K + M/c) d = f − K y_pred − Mψ/c` is solved exactly, and a
  numeric factorisation is only needed when `c = h/γ_order` changes. The controller changes `h` or the order at most
  once per `order + 1` steps, so a run of 141 steps used 16–25 factorisations (one symbolic analysis).
* Mostly order 5 is selected on smooth heat problems (118 of 141 steps).
* **Reusing a stale factorisation by refinement does not pay** at ±30 % on `c`: 3-D heat, 64 000 unknowns: 26 → 17
  factorisations (−8.5 s) but the refinement solves cost more than that (34.4 s → 37.2 s). The option exists
  (`BdfOptions::reuse_tolerance`, default off); it would only win when one factorisation costs ≫ 30 solves.
* Per-step error control gives *global* error ∝ tol^(p/(p+1)): ROS2 (p = 2) reached 4e-3, 1.9e-4, 8.7e-6 for
  rtol 1e-3, 1e-5, 1e-7 on the logistic problem, i.e. much more than the tolerance at the loose end. BDF5 stayed within
  0.5–5 × rtol. Use tighter tolerances for the low-order Rosenbrock method (the port list warns about inaccurate default tolerances:
  all tolerances are mandatory arguments here).
* Generalised-α with ρ∞ = 0.3 reduces the energy of the highest (ωh ≈ 10) mode by > 1e4 over 20 steps; average-acceleration
  Newmark keeps it, as it must; both are second order.
* Factorisation time dominates time stepping in 2-D (2.4 of 3.6 s) and absolutely in 3-D (1 s each); the 3-D Cholesky of the
  unit cube takes 1 s vs 50 ms per solve, as in item 8.
* `FactorStats::solves` is not updated by `solve_block_in_place(&self)` (shared reference), so per-solve counters and
  times read zero; use wall time minus factorisation time.

## 16. Nonlinear and quadratic eigenproblems (N2, N1)  (measured)

* Beyn's method in the block-Hankel form needs *no* projected nonlinear problem: all post-processing is small
  dense (SVD of the `Kℓ × Kℓ` Hankel matrix, one dense eigenproblem). With 48–64 nodes the eigenvalues in the
  region came out to 1e-7 or better even though the nearest outside eigenvalues were 0.02–0.03 from the contour
  (relative to a contour radius of 0.15–0.2). The `ζ = (z − c)/ρ` scaling of the moments is what keeps the Hankel
  matrix usable.
* A shift that sits on an eigenvalue makes `T(σ)` numerically singular, and residual inverse iteration then gets *worse*
  (residual 4e-12 → 1e-5). The routine returns the best iterate; polish with `σ` slightly off the eigenvalue.
* `T(z)` is assembled by `axpby` over the union pattern (explicit zeros kept), so every node and every Rosenbrock
  Jacobian has the same pattern and `ShiftedFactorization::replace_matrices` avoids all but the first symbolic analysis.
* **TOAR compressed restart pitfalls.** (1) `Q` can need more than `ncv + 2` columns after a restart (here 53 for 50
  steps; bound `2·ncv + 2`), so `Q` must be growable and the coefficient columns sized for the bound: with a fixed
  capacity, new directions were silently dropped and the *estimates said converged while the true residual was 2e-3*.
  Always check the explicit residual. (2) Within a cycle `r₁ = σ r₂` exactly (because `w₁ = x₂ + σ w₂` and `x₂ ∈ range Q`),
  so each step adds one column, which is where the memory halving comes from.
* The linearised Krylov–Schur run spends most of its time in a 2n LU and its solves; TOAR factorises `P(σ)` (size n)
  but needs more orthogonalisation per step in the compressed space, which cancelled the gain on the 2-D test: do not
  expect speed-ups, expect memory.
* faer's `generalized_eigen` on small pencils is fine (≤ 50×50); it was only slow for ~300×300 pencils with a huge zero/infinite cluster (item 14).

## 17. Frequency sweeps (W1)  (measured)

* A reduced basis needs about **one vector per resonance in the band**, so the saving is (grid points)/(resonances): 400
  points over 23 resonances → 35 solves (11×); 300 points over ~70 resonances → 68 solves (2.4× in time). Choose or split
  the band accordingly; there is no free lunch for modally dense bands.
* The residual of the reduced solution can be evaluated from Gram matrices of `A_j V` without any `n`-vector work
  (`terms²·r²` flops per frequency), which made the greedy loop essentially free, but it has a noise floor of about
  1e-8 relative (squared cancellation). The routine therefore switches to the exact residual (one sparse pass per
  frequency) as soon as the indicator claims convergence or its worst point is an expansion frequency.
* Rejecting a new basis vector because it is "already in the span" must use a threshold far below the residual
  target: near a resonance `‖x‖/‖b‖ ~ 1/(γω)`, so a 1e-10 relative out-of-span component already means a 1e-7 residual
  (this stopped the greedy loop at 1.5e-7 instead of 1e-8). The threshold is now 1e-13 after two Gram–Schmidt passes.
* Residual-based certification says the *residual* is below tolerance, not the error; at a resonance the error can be
  `‖T⁻¹‖` times larger (1e-9 output error against 1e-8 residual in the modal test, both fine because of the damping).

## 18. Nonlinear solvers (L1)  (measured)

* **JFNK must use right preconditioning.** With left-preconditioned GMRES the forcing tolerance applied to the
  *preconditioned* residual, and with an AMG preconditioner the resulting steps were not descent directions: the
  line search failed at the second iteration on the Bratu problem. Right preconditioning (true linear residual)
  fixed it: 5 Newton iterations, 25 Krylov iterations in total.
* A fixed preconditioner is only as good as the Jacobian it came from: LU(J(x₀)) at `x₀ = 0` (where the cubic term vanishes)
  was *worse* than none (252 vs 204 Krylov iterations); built at `0.6 u*` it cut them to 19.
* PETSc's `snes_stol` test reports convergence when the step is tiny even if the residual is large. Here a tiny step with
  a large residual is `DivergedStagnation`. In the arctan example the full-step iteration blows up to 1e150, where the Jacobian underflows to
  zero; the old test reported "converged by step size".
* Exact-Newton Armijo (`‖F_new‖² ≤ (1 − 2cα)‖F‖²`) with quadratic interpolation needs a few lines but is sensitive to the
  clamping of the interpolated step to `[0.1α, 0.5α]`; the arctan case needs it.
* Jacobian lag 3 gave 12 iterations and 4 factorisations against 7 and 7: worth it when a factorisation costs more than
  ~2 residual evaluations, i.e. nearly always for 3-D problems.
* Anderson acceleration with depth 6 turned a Picard iteration that did not reach 1e-10 in 3000 steps into 33 steps; the least-squares problem is
  solved by modified Gram–Schmidt with dropping of dependent columns (normal equations would square the condition number).

## 19. Storage formats and incomplete factorisations (M1, P3)  (measured)

* **SpMV is bandwidth-bound already in CSR** (49 MB / 5 ms ≈ 9.8 GB/s, the VM's streaming rate), so a format can gain at most
  the ratio of bytes moved: block CSR 3×3 moves 1.44× fewer bytes and ran 1.1–1.2× faster; 2×2 moves 1.35× fewer and ran 1.4–1.6×
  faster (timing noise ±10 %). Symmetric (upper-triangle) storage halves the matrix traffic but needs a scatter into two
  rows, which cost more than the saved traffic: 1.7× less memory, 1.06–1.1× *slower*. The 1.5–2× of the port list needs larger
  blocks, f32 values, or a machine where CSR is latency-bound.
* ILU(k)/ICC(k) with the level-of-fill rule `lev(i,m) = lev(i,j) + lev(j,m) + 1 ≤ k` in natural ordering gives *exactly* PETSc's iteration counts, so
  the sequential PETSc `PCILU`/`PCICC` are a valid tier-A reference. The symbolic phase needs a heap over the lower columns to visit them
  in ascending order; a dense level array with reset-by-list keeps it O(fill).
* ILUT(1e-3, lfil 20) cost 3.8× the nonzeros of `A` and cut GMRES(50) from 37 (ILU(0)) to 10 iterations on convection–diffusion
  (n = 3600): it only pays when the factors are applied many times.
* Incomplete Cholesky breaks down on matrices that are SPD but not an H-matrix (a banded matrix with 0.7 off-diagonals): the diagonal
  shift `A + α diag(A)`, `α = 1e-3·2^j`, restores a usable factor; the shift is reported so callers can see it happened.
* `Scalar` gained `sqrt_principal` for the complex-symmetric `LLᵀ` factorisation.

## 20. Schwarz preconditioners (P4)  (measured)

* PETSc's sequential `PCASM` with `-pc_asm_blocks N` uses contiguous index ranges, overlap by graph layers and (with `-sub_pc_type lu`) exact local solves: that
  reproduces its iteration counts exactly (43/34 for CG, 12/16 for GMRES+RAS), so it is a tier-A reference; the type must be set explicitly (`-pc_asm_type basic` for CG).
* **The Nicolaides coarse space must be smooth.** Indicator functions of the non-overlapping parts gave 28/41/54 iterations (4/16/64
  subdomains), no better than one level (24/39/63). A graph-distance partition of unity on the overlapping subdomains gave 27/37/39. Both are far from
  the theoretical subdomain-independent bound because the overlap is small (2 layers) and the problem is tiny (64² unknowns); expect better behaviour with
  generous overlap or a larger problem.
* RAS needed fewer GMRES iterations than ASM (21 vs 26) on convection–diffusion, and more overlap helped RAS (13 with 3 layers), as in the literature.
* Subdomain solves are embarrassingly parallel and summed in subdomain order, so scoped threads give bit-identical results for any worker count.

## 21. Sensitivities, tracking and optimisation (N3, O1)  (measured)

* The right eigenvector is **not** a substitute for the left one once the matrix is non-normal: on a 12×12 tridiagonal
  with 5:1 asymmetry the derivative formula with `y = x` was off by more than 1e-3 relative, with an inverse-iteration left vector it matched finite
  differences to 1e-5. The finite-difference reference itself degrades with the eigenvalue condition number (n = 30 gave 1.4e-3 noise, n = 12 gave 1e-5).
* For complex *symmetric* problems the left vector is the conjugate of the right one (`yᴴ = xᵀ`): no extra solve.
* Mode tracking by `B`-overlap with an optimal (Hungarian) assignment follows an exact crossing; sorting by eigenvalue loses the
  branch (deviation > 0.5 after the crossing in the test). Near-degenerate pairs give overlaps of about ½ each: that is what the
  `uncertain` flag is for.
* **Strong-Wolfe line searches fail at the resolution of `f`**: with `gtol = 1e-10` on a quadratic whose minimum value is about −60 the
  Armijo test `f ≤ f₀ + c₁αg·d` cannot be satisfied once the expected decrease is below `1e-14|f|`; adding a rounding allowance of `1e-14·max(1,|f|)` fixed it.
* Nonlinear CG needs the step-length heuristic `α₀ = α_prev·(g_prevᵀd_prev)/(gᵀd)` (N&W 3.60): without it Rosenbrock(10) needed 4839 function evaluations, with it 637;
  L-BFGS needed 98.
* MMA: the penalised dual (`λ_i ≤ c`, exact penalty form of Svanberg's artificial variables) with coordinate bisection is exact for one constraint
  and keeps an infeasible start workable; the topology problem converged to a 10× compliance reduction with volume exactly at the limit, but 23 % of the
  elements stay grey without a filter or projection.
* The adjoint gradient of the compliance needs *no second solve* for a self-adjoint problem with `g = f`; `adjoint_gradient` still does both solves on one factorisation (2 triangular solves) so
  non-self-adjoint objectives work.

## 22. More Krylov solvers, field splits, matrix exponential, LOBPCG, Chebyshev filtering  (measured)

* MINRES and BiCGStab reproduce PETSc's iteration counts when the problem is benign (SPD with ICC: 98/98; BiCGStab with ILU: 26/26 and 24/24).
  On indefinite or poorly preconditioned problems the *residual history* is chaotic in rounding (788 vs 811 iterations, history differences of 78 %) while
  the outcome (true residual 1e-8) agrees: the harness has a tier-B verdict for these.
* With exact diagonal blocks and an exact Schur complement the preconditioned saddle-point matrix has 3 (diag), 2 (triangular) or 1 (full) distinct eigenvalues: GMRES
  needed exactly 3/2/2/1 iterations, a cheap end-to-end check of every sign and transposition in the Schur code. "selfp" (`A11 − A10 diag(A00)⁻¹ A01`) with exact `A00` gave 12–25 iterations
  where nothing converged in 100.
* A stiffly coupled 3-field problem: block Gauss–Seidel halves the GMRES iterations of block Jacobi (15 vs 28); the symmetric sweep gains 2 more.
* `expv` with 30-dimensional Krylov spaces took a single sub-step for `‖tA‖ ≲ 20` on a 400-unknown heat problem and 12 sub-steps at `t = 300` (the solution has decayed to
  nothing; the step control still sees the full operator norm). Error 4e-15 to 1e-10 against the modal solution. A start vector inside a small invariant subspace makes the
  test meaningless (exact at 3 Krylov vectors): tests use random vectors.
* LOBPCG without a preconditioner on the 1-D Laplacian stalls for hundreds of iterations without being wrong; a stall test with a 30-iteration window declared failure too early, so the default window is 200.
  The AMG-preconditioned 2-D case needs 21 iterations for 8 modes (10× fewer than unpreconditioned).
* Chebyshev filtering cost 148 matrix products per wanted vector for the 40 lowest modes of a 2-D Laplacian (degree 20); the polynomial filter is
  weakest on 1-D-like spectra with tiny relative gaps, where a shift-and-invert Krylov method is the better tool.

## 23. Mixed precision (S3)  (measured)

* A single-precision LU with double-precision refinement reaches full accuracy in 3 refinement steps for well-conditioned shifted systems (error 4e-15 against a
  double LU), but the benefit is modest: factorisation 1.4× faster (4.1 vs 5.7 s, 3-D, 64 000 unknowns, shared pool), each solve 1.5× *slower* (three f32 solves plus three residual
  products). Break-even is at a few solves per factorisation, so shift-and-invert eigensolves (hundreds of solves) do not benefit; contour methods, frequency sweeps and
  transient refactorisation (one or a few solves per factorisation) do, as does anything memory-bound.
* Near an eigenvalue the shifted matrix has `κ ≈ 1/δ`: for δ = 1e-9 the f32 factorisation is useless. The test "residual must drop by a factor 2 per step" catches it after one or two steps
  and the fallback factorises in double precision on first need; inside a 30-solve shift-and-invert run one solve needed the fallback.
* A double LU has residual `~ ε‖A‖‖x‖/‖b‖`; for κ = 1e9 that is 4e-7, so "residual below 1e-8" is not a valid acceptance test there: compare against the double-precision residual instead.
* `faer` accepts `f32` and `Complex<f32>` in the low-level sparse LU through the same generic calls, with no changes besides the element type (the `FaerScalar` bound `Real = f64` only concerns the wrapper types here).
