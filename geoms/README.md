# geoms

Geometry kernel staging for the nyx FEM stack, next to [`meshers/`](../meshers/README.md) and [`solvers/`](../solvers/README.md).

| Path | What |
|---|---|
| [`geom/`](geom/README.md) | The crate `geom` (`cargo test --release` inside it). Depends only on `serde`. BSD-2-Clause. |
| [`docs/GEOM_HANDOFF.md`](docs/GEOM_HANDOFF.md) | The brief (scope, API contract, tests, steps A to D). |
| [`docs/GEOMETRY_ARCHITECTURE.md`](docs/GEOMETRY_ARCHITECTURE.md) | The design and the reasons. Where it differs from the brief, the brief wins. |
| [`docs/GEOM_HANDOFF_ADDENDUM_1.md`](docs/GEOM_HANDOFF_ADDENDUM_1.md) | Parameter-sweep support (signature, stable ids, fractions, `diff`). |
| [`PROVENANCE.md`](PROVENANCE.md) | What was taken from where (nothing), and the licence. |
| [`geom/CHANGELOG.md`](geom/CHANGELOG.md) | One entry per delivered step, including conflicts found between the brief and the reference code. |

The crate is staging code: it was written without the nyx repository (the handoff package carried read-only snapshots of the
relevant nyx files), and a later session integrates it as `crates/geom`.
