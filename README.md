# inspired-rust-port

## Meshing: [`meshers/`](meshers/README.md)

`meshers/femmesh` is a pure-Rust meshing crate for the nyx FEM solvers (metric-based mesh adaptation, structured PML meshes,
mixed-cell meshes, a compact lossless mesh/results format). It is staging code that a later session moves into nyx `femcore`.

* Takeover notes: [`meshers/HANDOFF.md`](meshers/HANDOFF.md). Detailed status and results: [`meshers/femmesh/docs/STATUS.md`](meshers/femmesh/docs/STATUS.md).
* **Provenance and notices: [`meshers/PROVENANCE.md`](meshers/PROVENANCE.md).** In short: no code was taken from any other project and no dependencies are used;
  the operator set follows the general approach of Omega_h / PRAgMaTIc / Mmg-style adaptation (inspiration only, no source read), and a few standard
  algorithms (range coder, Fx hash, Gorilla-style float coding, RCM) were re-implemented from their published descriptions. The table there says which, file by file.
* Licence: BSD-2-Clause ([`meshers/femmesh/LICENSE`](meshers/femmesh/LICENSE)), the same terms as the solver ports in this repository.

