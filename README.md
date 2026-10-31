# inspired-rust-port

Rust ports of numerical-solver and meshing building blocks, staged here for later porting into nyx `femcore`. One folder per topic,
each self-contained (own Cargo workspace, README, licence and notices) and written by its own working session.

| Folder | Content | Start here |
|---|---|---|
| [`solvers/`](solvers/README.md) | `rsolvers` (real-only Krylov / Krylov–Schur port following PETSc/SLEPc 3.19 behaviour) and `femsolve` (generic real/complex eigen-, linear-, nonlinear- and time-stepping solvers, preconditioners, low-rank and matrix-free tools, SLEPc/PETSc comparison harness) | [`solvers/docs/STATUS.md`](solvers/docs/STATUS.md), [`solvers/docs/EXPLAINER.md`](solvers/docs/EXPLAINER.md), [`solvers/PROVENANCE.md`](solvers/PROVENANCE.md) |
| [`geoms/`](geoms/README.md) | `geom`: CSG-over-B-rep geometry kernel (feature lists, named regions and faces, selectors, label maps, layer stacks, exact projection) | [`geoms/HANDOFF.md`](geoms/HANDOFF.md), [`geoms/geom/README.md`](geoms/geom/README.md), [`geoms/PROVENANCE.md`](geoms/PROVENANCE.md) |
| [`meshers/`](meshers/README.md) | `femmesh`: metric-based mesh adaptation, structured PML meshes, mixed-cell meshes, a compact lossless mesh/results format | [`meshers/HANDOFF.md`](meshers/HANDOFF.md), [`meshers/femmesh/docs/STATUS.md`](meshers/femmesh/docs/STATUS.md), [`meshers/PROVENANCE.md`](meshers/PROVENANCE.md) |

## Solvers: `solvers/`

* Build and test: `cd solvers && cargo test --release` (add `--features parallel` in `femsolve` for the threaded kernels).
* **Provenance and notices: [`solvers/PROVENANCE.md`](solvers/PROVENANCE.md)** — for every module: was code taken (none is recorded),
  was a PETSc/SLEPc source file read and the behaviour re-implemented in new Rust (BSD-2-Clause, files named), or was the code written
  from a paper. Licence texts: [`solvers/THIRD_PARTY_NOTICES.md`](solvers/THIRD_PARTY_NOTICES.md).
* Licence: BSD-2-Clause ([`solvers/LICENSE`](solvers/LICENSE)).

## Meshing: `meshers/`

`meshers/femmesh` is a pure-Rust meshing crate for the nyx FEM solvers (metric-based mesh adaptation, structured PML meshes,
mixed-cell meshes, a compact lossless mesh/results format). It is staging code that a later session moves into nyx `femcore`.

* Takeover notes: [`meshers/HANDOFF.md`](meshers/HANDOFF.md). Detailed status and results: [`meshers/femmesh/docs/STATUS.md`](meshers/femmesh/docs/STATUS.md).
* **Provenance and notices: [`meshers/PROVENANCE.md`](meshers/PROVENANCE.md).** In short: no code was taken from any other project and no dependencies are used;
  the operator set follows the general approach of Omega_h / PRAgMaTIc / Mmg-style adaptation (inspiration only, no source read), and a few standard
  algorithms (range coder, Fx hash, Gorilla-style float coding, RCM) were re-implemented from their published descriptions. The table there says which, file by file.
* Licence: BSD-2-Clause ([`meshers/femmesh/LICENSE`](meshers/femmesh/LICENSE)), the same terms as the solver ports in this repository.

Repository-wide summary of origins and licences: [`NOTICES.md`](NOTICES.md).

## Geometry: `geoms/`

`geoms/geom` is a pure-Rust geometry kernel for the nyx FEM stack: a serialisable feature list (parameters, 2D profiles, 3D prismatic solids,
Booleans, an ordered `Overlay` of named regions) evaluated into a region complex with provenance-based names, rule-based selectors, a compiled
`LabelMap`, exact projection and layer-stack data for the meshers. Only `serde` is required. See [`geoms/HANDOFF.md`](geoms/HANDOFF.md)
for the state (2D complete; 3D prismatic only), and [`geoms/PROVENANCE.md`](geoms/PROVENANCE.md). Licence: BSD-2-Clause.
