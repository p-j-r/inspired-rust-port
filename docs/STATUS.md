# femsolve: port status (2026-10-29)

**Scope.** Phase 1 ported the P0 set of `solver_port_list.md` section 13, step 1, and stopped where the
handoff brief says to stop and report (after V1 + interface + export, and after E1). Paul then
approved going on ("start implementing the rest"), with three decisions: build our own AMS test
generator, read only permissively licensed reference sources (papers otherwise), and follow the list
order (K1 → P2 → E3 → E2 → AMS → T1 → N2/N1 → W1 → L1 → M1/P3/P4 → O1; GPU and MPI as design notes
only). Phase 2 has done K1, P2, E3, E2 and P1 (AMS) so far; the table says what remains. Everything is in the
`femsolve` crate of this repository because the nyx repository was not available (the brief leaves
its name blank); `NYX_INTEGRATION.md` says how to move it.

**How sure I am.** "Done" below means implemented, tested, and compared with SLEPc where a
comparison exists; the evidence column says which. Timings are from one 4-vCPU VM and are noisy
(±15 %); nothing was measured on paul's machines.

## Items

| Port-list item | Status | Evidence |
|---|---|---|
| Solver interface (trait) | **done** | `linop::LinearOperator` (with fusion hooks `apply_dotc`, `apply_dotu`, `apply_block`, `apply_block_with_bx`), `linop::Preconditioner` (`apply_dots`, `scaling`), `direct::ShiftedSolver` ("factorise at σ, solve"), `problem::ShiftInvertEigenSolver` ("eigenpairs near a shift"). Real and complex scalars through one `Scalar` trait. |
| Matrix export | **done** | `mm.rs` (Matrix Market, real/complex, general/symmetric/hermitian, vectors), `bundle.rs` (A, B, v0, meta), CLIs `fs-gen`, `fs-eps`. Round trips are bit exact (unit tests). |
| **V1** SLEPc/PETSc comparison harness | **done for eigenproblems** (real and complex builds); the linear-solver (KSP) side is not built | `harness/` (C driver, `run.py`, 16 cases, `harness/results/full.md`). Needs Linux/WSL. |
| **S1** symbolic-factorisation reuse | **done** | `direct::ShiftedFactorization` keeps the union pattern, the symbolic analysis and the numeric buffers; test asserts one analysis for three shifts. Measured gain is small, see below. |
| **E1** Krylov–Schur, stage 1 (Hermitian, B inner product) | **done** | tier-A parity with SLEPc (restart and matvec counts identical, eigenvalues to 1e-14) on 5 real/complex Hermitian cases |
| E1 stage 2 (general complex, Schur reordering) | **done** | tier-A parity on complex symmetric and complex-pencil cases (including 11 restarts / 119 matvecs with complex `B`) |
| E1 block variant | **done** | finds every copy of degenerate eigenvalues (tests); not comparable with SLEPc (it has no block Krylov–Schur) |
| E1 "24 modes with a basis of about 60 to 1e-9" | **done on synthetic problems**; **not done on nyx's elastic matrices** | `tests/shift_invert.rs::twenty_four_modes_with_a_basis_of_sixty` (real generalised and complex symmetric, exact spectra). The 24.85 s baseline problem needs nyx. |
| **K1** CG / COCG / GMRES / FGMRES (real and complex) | **done** | `ksp/` (`Cg`, `Gmres`, PETSc 3.19 defaults and stopping rules), `pc::Jacobi`, `fs-ksp`; harness `run-ksp` against PETSc: GMRES/FGMRES iteration counts equal in 9 of 9 cases (real and complex, left/right PC, MGS/CGS/CGS+refinement); CG and COCG on real SPD agree to 6e-14. PETSc's COCG is broken on complex symmetric systems (stops at iteration 2, real-part sign test), so COCG is compared on real SPD only; `petsc_sign_test` reproduces PETSc if wanted. Results: `harness/results/ksp.jsonl`. |
| **P2** smoothed-aggregation AMG, shifted-Laplacian helper | **done** (symmetric real and complex-symmetric; `hermitian` option) | `amg.rs`; `tests/amg.rs` (7 tests): operator complexity 1.3–1.6, grid-independent PCG count (9–10), about PETSc GAMG's iterations with 4–6× faster setup; shifted-Laplacian AMG preconditions indefinite Helmholtz about as well as an exact shifted LU. **Not done:** classical/Ruge–Stüben coarsening, non-symmetric problems, parallel setup. |
| **E3** contour-integral eigensolver (block SS/CIRR) | **done** | `contour.rs`, `fs-contour`, harness `run-contour` vs SLEPc EPSCISS: same eigenvalue sets to 1e-15 (real FE), 4e-12 / 2e-10 (complex pencil), 4 workers 1.8× faster than serial CISS; results independent of worker count. The stochastic count estimate is too noisy on the complex pencil (reported with `count_stderr`; see `LEARNINGS.md` 12). |
| **E2** Jacobi–Davidson (interior, standard or Hermitian-generalised) | **done**, tier B (no SLEPc/PETSc counterpart in the harness) | `jd.rs`, `tests/jd.rs` (5 tests: B-orthonormal interior modes with exact and with crude preconditioners, non-Hermitian complex-symmetric standard problem with Schur deflation, degenerate pair). JD returns *some* true eigenvalues near the target, not necessarily the exact nearest set: tests check that every value is a true eigenvalue among the nearest dozen. Non-Hermitian *pencils* are rejected (use E3 or Krylov–Schur). Not compared with an external JD implementation (none is wired into the harness). |
| **P1** AMS (Hiptmair–Xu) for lowest-order H(curl) | **done on the generated cube problem**; **not run on nyx edge matrices** | `ams.rs` (symmetric `01210` cycle: Gauss–Seidel, `G` correction, `Π` correction as three additive scalar AMGs, `G` correction, Gauss–Seidel; real or complex-symmetric `A`); own test generator `maxwell.rs` (Kuhn-tet Nédélec order 1 with PEC; `K·G = 0` to rounding, cavity eigenvalues 2π², 3π² with second-order convergence). `tests/ams.rs`: PCG iterations 12/12/13 for n = 4/8/12 against 145/402/682 for Jacobi; at β = 1e-4 (almost singular) 10 vs 1078; COCG on a lossy complex-symmetric matrix 13 iterations; JD + AMS finds the 2π² modes of the cube without any factorisation (33 outer/307 inner iterations, standard extraction). **Not done:** hypre comparison (hypre is not installed here), higher-order elements, the full-matrix `Π` AMG (we use the three scalar diagonal blocks), additive cycle, anisotropic/variable-coefficient tuning, nyx's real `G`/coordinates (needs the export). |
| P3, P4, P5, E4, E5, T1, W1, N1–N3, L1, M1–M2, O1, X1 | **not started** | next in the agreed order: T1, N2/N1, W1, L1, M1/P3/P4, O1 |
| G1 (GPU), D1 (MPI) | **design notes only, by agreement** (no GPU or cluster here) | |

## Verification

Acceptance tiers (from the review in the handoff):

* **Tier A** (same algorithm): identical restart and operator-application counts, eigenvalues to
  1e-10 relative, residuals recomputed independently. **Passed in 13 of 13 non-degenerate cases**,
  up to 178 restarts / 1790 matvecs; eigenvalue agreement 1e-14 and better; error estimates per cycle equal to 4–8 digits.
* **Tier B** (different path to the same answer): distinct eigenvalues to 1e-8, residuals below
  tolerance. Used for degenerate spectra (`lap2d`, `lap3d`) and for the 100-restart non-normal
  case where rounding amplification makes counts unreproducible (128 vs 100 restarts, eigenvalues
  to 3e-10). 3 of 3 pass. Reasons and numbers: `LEARNINGS.md` items 5 and 6.
* **Tier C**: no adaptive/time-dependent solver was ported.
* Analytic tests: 1-D Laplacian, gauge-transformed (complex Hermitian) Laplacian, FE generalised
  spectrum, `L + iγI` (exact complex-symmetric spectrum), a mildly non-normal tridiagonal, degenerate
  2-D grids, forced breakdown, 24 modes.
* Determinism: `cargo test --features parallel` checks bitwise equality of eigenvalues and
  vectors for 1 vs 4 rayon threads (block variant, weighted, complex general, fused SpMV+dot).
* Unit tests of every kernel against naive loops, of the complex Schur/reordering against faer.

## Timing against SLEPc (full table: `harness/results/full.md`)

| Case | Rust (warm, incl. numeric factorisation) | SLEPc (warm, factorisation reused) | Cold (all-in) |
|---|---|---|---|
| 1-D problems, n = 2 000 – 20 000, real/complex | 0.5 – 1.2 × | = 1 | 0.7 – 1.3 × |
| direct Hermitian (no shift), 178 restarts | 0.43 × | = 1 | n/a |
| complex pencil, n = 3 000 | 1.13 × | = 1 | 1.05 × |
| complex pencil with absorbing layer, n = 40 000 (128 vs 100 restarts) | 1.85 × (1.45 × per matvec) | = 1 | 1.8 × |
| real 3-D SPD, Cholesky, n = 64 000 | 1.18 × | = 1 | **0.26 ×** (4.7 s vs 18.2 s: faer's numeric Cholesky is ~14× faster than PETSc's) |

Reading: the eigensolver itself matches SLEPc; wall time is dominated by the direct solver's
solves (62–99 %), where faer ≈ PETSc's built-in LU per solve (faer's *factorisations* are faster, its
2-D complex solves ~25 % slower per vector, probably fill from a different ordering).
**Timing caveats:** SLEPc's repeated solves reuse the factorisation (its logged warm factor time is 0), ours
re-factorise at every `solve_near`; the 1-D cases are milliseconds, inside the noise.

With the `parallel` feature the basis kernels reach 3.5–4 × the single-thread bandwidth on the 4
vCPUs (`examples/basis_bench.rs`). End to end on the 40 000-unknown complex pencil (`ncv` 40, 21
restarts, 440 matvecs, identical counts): orthogonalisation 3.84 → 1.66 s, restarts 0.23 → 0.09 s,
total 11.6 → 9.7 s (1.2 ×), because the 7.5 s of sequential solves dominate. Parallelism across
right-hand sides of a block would not help much either: solves are memory-bound and faer's
multi-RHS solve already reads the factors once per call.

## Known limitations of what was ported

* Shift-and-invert only. No harmonic extraction, no purification (SLEPc's default for generalised
  Hermitian problems; it changes returned vectors, not restart logic), no polynomial filtering.
* Real *non-symmetric* operators are rejected (Schur form with 2×2 blocks is not implemented);
  promote to complex.
* Hermitian kind assumes the shifted operator is self-adjoint in the inner product used; this is not checked.
* The general kind's residual estimate uses the eigenvector of the active triangular block and
  ignores the locked-part components (as the SLEPc estimate effectively does); the problem layer
  recomputes true residuals, the raw `krylov_schur` does not.
* Matrices are limited to 32-bit indices (`direct.rs` returns `TooLarge`).
* Real chip matrices, GPUs, cluster runs, Windows: untested.
* E2: no thick-restart across *different* targets, no block JD; the inner solver is plain GMRES
  with the (projected) preconditioner, `inner_its` 12 / `inner_tol` 0.1 are untuned defaults.
* E3: no adaptive quadrature order or automatic region splitting; count estimate unreliable on
  non-Hermitian pencils.
* P2: smoothed aggregation only; strength threshold and smoothing weight were set on 2-D/3-D Laplacian
  and Helmholtz-type problems, not on elasticity (needs near-null-space vectors, not implemented).

## Next steps (ordered; each is self-contained)

1. **Get the nyx repository name** and follow `NYX_INTEGRATION.md`. Until then the whole port is unverified against nyx's real operators.
2. **Export one optical and one elastic matrix pair** (needs paul's permission, list item 10) and run `fs-eps` and the SLEPc harness on them; this also settles the open "MKL PARDISO / threaded MUMPS" question when added as a ShiftedSolver backend.
3. **Check nyx's block solves for the faer parallel multi-RHS cliff** (`LEARNINGS.md` item 1) and for the dirty-vector-state slowdown (item 3). Both can silently cost multiples on an AVX-512 machine and both are minutes to test.
4. **Measure rayon wake-up on paul's Windows machine** (an empty broadcast, as in the review's pitfall 2) before trusting the `parallel` feature there; `PAR_MIN_ELEMS` in `par.rs` is a guess from the VM.
5. ~~KSP half of V1 and item K1~~ done in phase 2 (`harness/c/petsc_ksp.c`, `run.py run-ksp`).
6. **Block default and `ncv` rule.** Blocks helped real SPD 3-D problems (1.7 ×) and hurt a slow complex one (3 ×): either keep 1 or add a heuristic after trying it on nyx models.
7. **Optional performance:** runtime-dispatched AVX2 versions of `dots_block` (≤ 5 % of an eigensolve); fill-reducing ordering experiment S2 (faer's complex 2-D solves are ~25 % slower than PETSc's per vector); reuse solve scratch buffers across calls (allocation per solve is small but nonzero).
8. **Breadth set:** SuiteSparse matrices were unreachable (`sparse.tamu.edu` returned 403 from the sandbox); someone must upload a few or change the network allowlist.
9. **Questions still open for paul** (port list section 11, none answered in this session): GPU model, largest 3-D model and RAM, typical/largest mode counts, interior vs lowest modes, frequency-dependent materials, transient work, optimisation plans, cluster access, accuracy targets, chip-matrix export permission.
