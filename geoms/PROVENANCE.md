# Provenance and licence notice for `geoms/`

* **No source code was copied, translated or adapted from any other project.** The crate has one required dependency, `serde`
  (MIT / Apache-2.0), plus `rayon` behind the off-by-default `parallel` feature and `serde_json` / `proptest` as dev-dependencies only.
* The handoff package (`geom_handoff.zip`) contained **read-only snapshots of nyx source files** (`femcore` `mesher.rs`, `mesher3d.rs`,
  `expr.rs`, `xmlio.rs`, `usecases/*`, `chips.rs`, ...). They were *read* to learn names, numbering rules, reference numbers and
  conventions. They are **not** included in this repository and no code was copied from them, with one deliberate exception in spirit:
  `geom/src/expr.rs` follows the grammar and function list of nyx `femcore::expr` (a parser of the same shape: precedence climbing, the
  same function set) because the brief asks for the same grammar; it is re-written, with parameter names instead of `x y z`.
  nyx's own licence is not known to the author; check it before merging `geom/src/expr.rs` into the nyx tree (it will be the same
  owner's code in that case).
* Algorithms are standard: Green's theorem for areas, Gauss-Legendre quadrature, closed-form line/circle/circle intersections, a
  half-edge rotation system for loops, sort-and-sweep pair finding. Written from understanding; no source read.
* Licence of this folder: **BSD-2-Clause** (`geom/LICENSE`, copied from `meshers/femmesh/LICENSE`, same holder line).
