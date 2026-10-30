# femmesh

Pure-Rust metric-based mesh adaptation, structured PML meshes and mixed-cell meshes (triangle, quad,
tetrahedron, prism, hexahedron) for finite-element solvers. No dependencies.

```bash
cargo test --release
cargo run --release --example pml_reflection      # PML truncation error vs layers, quad vs split
cargo run --release --example bench_metric        # fused vs unfused derivative-to-metric kernel
cargo run --release --example adapt_convergence   # 2D adaptation: error vs elements, adapted vs uniform
```

Modules: `mesh`, `cellmesh` (stand-ins for nyx's mesh types), `adapt::{metric, field, topo, geom, ops, drive, pml}`,
`quality`, `fixtures`, `io::{vtk, fmsh}`, `par`, `timer`, `validate` (small Helmholtz PML solver used to test meshes).
Status, results and open questions: `docs/STATUS.md`. References and licence provenance:
`docs/REFERENCES.md`, `THIRD_PARTY_NOTICES.md`.
