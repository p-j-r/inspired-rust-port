# meshers

Pure-Rust meshing for the nyx FEM solvers: metric-based mesh adaptation, structured PML meshes, mixed-cell meshes
(triangle, quad, tetrahedron, prism, hexahedron) and a compact lossless mesh/results file format.

| Path | What |
|---|---|
| [`femmesh/`](femmesh/README.md) | The crate (`cargo test --release` inside it). No dependencies. BSD-2-Clause. |
| [`HANDOFF.md`](HANDOFF.md) | **Start here if you are taking this over**: state, what to do next, traps, open questions. |
| [`PROVENANCE.md`](PROVENANCE.md) | **What was taken, what was only inspired, and what was written from scratch.** Licence rules for the next session. |
| [`docs/nyx_meshing_adaptivity_handoff.md`](docs/nyx_meshing_adaptivity_handoff.md) | The original brief (milestones M0-M16, rules, questions). |
| [`femmesh/docs/STATUS.md`](femmesh/docs/STATUS.md) | Detailed status, measured results, decisions. |
| [`femmesh/results/`](femmesh/results) | Result tables (M1 kernel, M2 and M7 PML studies, M4 adaptation, `.fmsh` sizes). |

State in one line: milestones M0 (partly), M1, M2, M3, M4 and M7 are done; the work was written without access to the nyx
repository, so it runs on stand-in mesh types and has never been run against the real solvers or chips.
