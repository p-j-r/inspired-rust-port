# Handoff: `geom` (geoms/)

For the integration session. Read [`geom/README.md`](geom/README.md) (conventions, limits, performance), [`geom/CHANGELOG.md`](geom/CHANGELOG.md)
(what each step delivered and what is missing), then the code. The brief is [`docs/GEOM_HANDOFF.md`](docs/GEOM_HANDOFF.md) with
[`docs/GEOM_HANDOFF_ADDENDUM_1.md`](docs/GEOM_HANDOFF_ADDENDUM_1.md).

## Build and test

```bash
cd geoms/geom
cargo test --release                 # about 40 tests (the property tests run 200-300 random cases each), a few seconds
cargo test --release --features parallel
cargo clippy --release --all-targets -- -D warnings
cargo run --release --example geom_bench
GEOM_WRITE_GOLDEN=1 cargo test --test goldens write_goldens   # regenerate golden/*.json from tests/common
```

## State against the brief

| Step | State |
|---|---|
| A (2D kernel, names, selectors, LabelMap, serde, builder, goldens) | done; 2D goldens match closed forms to 1e-10 |
| B (params, tiers, cache, normalisation, to_rust, tessellation, oracle, deviation and curvature queries, addendum 1) | done |
| C (3D) | polyhedral solids at any orientation (`planar.rs`: rotations, tapers, slanted cuts, Booleans), prismatic models (Box, z-Cylinder, Extrude, z-rotations, Cut along axes, Pml, Periodic, Size), 3D edges and vertices, and `Sphere` / `Torus` bodies clear of everything else; **no** intersection curves between curved surfaces (cone, revolve, a sphere crossing a wall, skew or tilted cylinders) |
| D (layer stack, `surfaces_for_stuffing`, bench) | done for what C produces |

## What to check first when integrating

1. **Edge and face names**: auto-names are `feature.role` and `A|B`; boundaries take their name from the shape of the *inside* region
   (`box.bottom`, not `clad.bottom`, for the bottom of the BOX region). Confirm this reads right for the GUI (brief section 14, fragment naming).
2. **Tolerance is relative to the model**: adding a feature far away changes `tol = 1e-9 L`; input within a few tolerances of degenerate is refused
   or snapped. Everything undecidable (tangent contact, near-coincident curves, nearly touching corners, primitives thinner than 64 tol, z levels
   closer than 64 tol) is an error that names the features.
3. **Edge ids are parameter independent** within a topology (ordered by first origin and the fraction along the origin segment). A line rotated
   through the vertical flips its canonical direction, which swaps `left`/`right` (a signature change): sweeping rotation angles through 90 degrees is not morph-safe.
4. **Elliptic arcs** are intersected numerically (720 samples); two roots closer than one sample can be missed.
5. **Layer stack**: faces are fragments of the overlay; a lateral face spanning layers is merged while sides and name agree. Caps are one fragment per overlay cell.
6. The curved-element refactor needs, from geom: exact projection (`oracle`), `point_at_mid`, `project_midpoint`, `chord_deviation`,
   `facet_deviation`, `min_radius_of_curvature`, `discretize_edge` (2D edges today; for 3D use the overlay edges of the stack), and orientation
   (normal from the first region to the second). 3D surface curvature queries other than the cylinder radius (`facet_radius`) are not implemented.

## Open items and risks (brief section 14)

* The general 3D arrangement (quadrics, `SurfSurf`) is the remaining large piece; the prismatic route covers the photonic cases in the brief.
* Source size is about 8.1k lines after `rustfmt`, just over the brief's 8k line; the largest modules are `arrange.rs`, `scene.rs`, `stack.rs`, `select.rs`. The growth since step B is the 3D work (`stack.rs`, `topo3.rs`) and the curved bodies.
* The pair cache saves little at the model sizes measured; see the README performance table.
