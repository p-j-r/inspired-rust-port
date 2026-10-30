# femmesh status

Branch `claude/quirky-euler-20hyv1`. Build and test: `cargo test --release` (59 tests, ~1 s).
Machine for all numbers below: 4-core Intel Xeon @ 2.8 GHz VM, rustc 1.97.0, release profile
(`opt-level 3`, fat LTO), no `target-cpu=native`. VM timing noise is about ±15 %, so differences under
10 % are ties.

## What was and was not available

The nyx repository, the plan (`nyx_meshing_adaptivity_plan.md`) and the nyx source files named in the
handoff were **not reachable** in this session. paul's instruction was to build the port as a
standalone crate (`femmesh`) that another session moves into `femcore`. The mesh types in
`mesh.rs` / `cellmesh.rs` are therefore **stand-ins** for `SimplexMesh` / `CellMesh`, written from the
handoff's description, and the label scheme for PML regions is configurable instead of copied from
`chips.rs`. Nothing here was checked against the real nyx APIs.

## Milestones

| # | Milestone | State | Notes |
|---|---|---|---|
| M0 | fixtures, statistics, VTK, timing/determinism patterns | **partly done** | Done: box meshes (2D/3D) with boundary labels, analytic fields with exact Hessians, quality and edge-length statistics, VTK writer, `.fmsh` compact format, per-phase timers, deterministic chunked parallel reductions. **Blocked on nyx:** the four chip benchmarks and the baseline numbers of the hand-tuned chip meshes. **Not done (waiting for answer 12):** Mmg/Omega_h offline export. |
| M1 | `adapt/metric.rs` | **done** | Tier-A algebra tests pass; fused derivative-to-metric kernel; benchmark below. |
| M2 | `adapt/pml.rs` in 2D | **done except the chip comparison** | Generator, labels, split fallback, reflection-vs-layers study done on a stand-in Helmholtz solver. **Stop and report** (this document). The chip eigenvalue comparison needs nyx. |
| M7 (early) | `adapt/pml.rs` in 3D | generator done, solver-side checks not | Prisms on faces, hexahedra on edges and corners, conforming, with the simplex split. No 3D solver in this crate to run the native-vs-split eigenvalue check. |
| M3–M6, M8–M16 | | not started | |

## M1 result: fused derivative-to-metric kernel

`results/m1_metric_bench.md`, 200 000 vertices, minimum of 7 interleaved runs, 4 threads for the
parallel column:

| | unfused | fused 1 thread | fused 4 threads | complexity miss fused / unfused |
|---|---|---|---|---|
| 2D | 25.8 ms | 26.5 ms | 13.7 ms | 3e-15 / 1e-8 |
| 3D | 196 ms | 134 ms | 40.5 ms | 2e-16 / 1e-5 |

* 3D: 1.46x faster than the multi-pass version and reaches the target complexity exactly. 2D: a tie.
* "Unfused" is the straightforward multi-pass version (separate loops, `|H|`, determinants, scaling,
  complexity, clamp). It scales once and clamps once, so it **misses the target complexity whenever
  clamping is active** (1e-8 in 2D and 1e-5 in 3D on this data). The fused kernel solves for the
  global scale on the *clamped* complexity, so it hits the target to roundoff. The two are therefore not
  equal work; the speed comparison understates the accuracy gain.
* Diagnosis of the first version: it bisected the scale (about 35 evaluations over all vertices) and was
  0.64x in 2D. Replacing the bisection with a safeguarded regula falsi on the monotone log-log function
  brought the scale search to a few evaluations.
* 3D time is 66 % eigen-decomposition (Jacobi) — hypothesis: a closed-form 3x3 solver would help, at some
  cost in eigenvector accuracy. Not tried. 2D is dominated by `powf` and the scale search, not by the eigen-solve (15 %).
* Results are bit-identical for 1, 2, 3 and 8 threads (fixed 512-element chunks, sums in chunk order).
  Assembly of the generated code was not inspected.

## M2 result: structured quad PML vs the same layers split into triangles

`results/m2_pml_reflection.md`, produced by `examples/pml_reflection.rs`. Independent scalar Helmholtz
solver in `validate/` (P1 triangles, Q1 quads, complex coordinate stretching sampled at the quadrature points,
Dirichlet outer boundary, banded LU). Physical domain 3x3 wavelengths, 12 points per wavelength,
Gaussian source, layer thickness 1 wavelength. Error = relative L2 difference over the physical vertices from a
reference on the same interface mesh with a 2-wavelength layer and 48 layers. Quad layers are compared with a quad reference
and split layers with a split reference (the two references differ by 2.6e-3 on the physical vertices — comparing both with a single
reference favoured whichever kind the reference was).

Best error over the design reflection R0 in 1e-1..1e-8 and grading in {1, 4}:

| layers | dof | quad | triangles (split) |
|---|---|---|---|
| 3 | 1849 | 1.0e-1 | 9.8e-2 |
| 4 | 2025 | 4.3e-2 | 4.3e-2 |
| 6 | 2401 | 4.6e-3 | 4.5e-3 |
| 8 | 2809 | 1.2e-3 | 1.8e-3 |
| 12 | 3721 | 1.2e-3 | 1.2e-3 |

* **For scalar P1/Q1 the structured quad layers are not more accurate than the same layers split into
  triangles.** The split has the *same vertices*, hence the same dof count; differences are within the noise of
  the two references. About 6 layers give ~0.5 %, about 8 give ~0.1–0.2 %, and the error floors near 1e-3.
* So the case for quads in 2D rests on what this test cannot show: half the cell count, exact control of layer
  count and grading, no sliver triangles from an unstructured layer mesher, and (hypothesis) Nédélec on quads. I could not compare
  against nyx's current triangular PML mesh because I do not have it.
* The handoff's Tier-A line "native mixed PML mesh and its simplex split give the same eigenvalues" cannot hold
  exactly: Q1/Nédélec on quads and P1/Nédélec on the split triangles are different finite-element spaces. I expect agreement to
  discretisation error only (the table above is that agreement for the scalar problem) and propose the Tier-A test be
  stated that way.

PML generator properties tested (2D and 3D): conforming (every face shared by at most two cells,
the only boundary faces are the outer surface), positive scaled Jacobians, exact total area/volume, deterministic bits,
region boxes with the correct thickness and the stretched directions encoded in the label, materials
crossing the interface continue into PML copies, partial layers (sides with zero thickness), graded interface meshes,
rejection of non-box domains, and the simplex split being conforming and volume-exact (hex into 6 tets, prism into 3,
cone from the smallest vertex id, quad diagonal through the smallest id — this is what makes neighbouring cells agree).

## Decisions I made that you may want to change

* Crate name `femmesh`, no dependencies, BSD-2-Clause (the licence of the other branch; **nyx's licence is unknown to me**).
  It builds in its own workspace (`femmesh/Cargo.toml` has an empty `[workspace]`) so it does not touch PR #1's workspace.
* PML label = `label_base + region_index * label_stride + material`, region index `sum (code_a + 1) 3^a`; defaults 1000 / 1000.
  nyx's real labels (`chips.rs`) will need an adapter.
* Edge/corner region material rule: material of the lowest-index boundary facet on the first stretched side touching it.
* Gradation uses a linear size-growth law with slope `g` (my parameterisation; not verified equal to Alauzet's).
* `.fmsh`: lossless, CRC-32 checked, varint/delta connectivity, per-axis dictionary coding of coordinates; no external codec.
  On a 100x100 structured triangle mesh it is 4.3x smaller than plain binary arrays and 5.2x smaller than ASCII VTK. Arbitrary adapted coordinates
  are close to incompressible losslessly; a lossy quantised option was not added because PML interface vertices must stay exactly on their plane.
  Deflate/zstd would need a pure-Rust crate (`miniz_oxide` or `ruzstd` are candidates; licences not yet checked).

## Still open from the handoff (questions 3, 5, 7–13) and new ones

* Max Nédélec order and families (3), what users draw for 3D geometry (5), thin structured films outside the PML (7),
  target sizes and time per cycle (8), chip export (9), QoIs (10), Floquet/mirror importance (11), offline Mmg/Omega_h
  comparison (12), `brillouin.rs` (13).
* Where is the nyx repo, and is `p-j-r/brillouin` related? (I was told not to worry about nyx; this only matters for the adapter.)
* The plan file and the nyx files were never visible to me; the summary copy "to the project files" could not be made
  (`/mnt/project-files` does not exist here). This document is the summary.
