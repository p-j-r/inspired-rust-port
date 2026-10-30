# femmesh

Part of [`meshers/`](../README.md) (takeover notes: [`../HANDOFF.md`](../HANDOFF.md), provenance: [`../PROVENANCE.md`](../PROVENANCE.md)).

Pure-Rust metric-based mesh adaptation, structured PML meshes and mixed-cell meshes (triangle, quad,
tetrahedron, prism, hexahedron) for finite-element solvers. No dependencies.

```bash
cargo test --release
cargo run --release --example pml_reflection      # PML truncation error vs layers, quad vs split
cargo run --release --example bench_metric        # fused vs unfused derivative-to-metric kernel
cargo run --release --example adapt_convergence   # 2D adaptation: error vs elements, adapted vs uniform
cargo run --release --example pml_3d              # 3D PML: native hex/prism layers vs split tets
cargo run --release --example fmsh_size           # .fmsh size vs plain binary / gzip / xz
```

Modules: `mesh`, `cellmesh` (stand-ins for nyx's mesh types), `adapt::{metric, field, topo, geom, ops, drive, pml}`,
`quality`, `fixtures`, `io::{fmsh, entropy}`, `par`, `timer`, `validate` (small 2D/3D Helmholtz PML solver used to test meshes).
Status, results and open questions: `docs/STATUS.md`. References and licence provenance:
`docs/REFERENCES.md`, `THIRD_PARTY_NOTICES.md`.

Storage is a single `.fmsh` file (mesh plus any number of result steps, lossless, CRC-checked); there is no VTK or text writer.
