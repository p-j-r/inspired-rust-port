# geom

A lean CSG-over-B-rep geometry kernel for FEM: a serialisable feature list (parameters, profiles, Booleans, an ordered `Overlay` of
named regions) is evaluated into a **region complex** with shared interface edges, provenance-based names, rule-based selectors and a
compiled `LabelMap`. Code and GUI edit the same data. No meshing happens here.

Status: **step A (2D) is delivered**; see [`CHANGELOG.md`](CHANGELOG.md) and the brief in [`../docs/GEOM_HANDOFF.md`](../docs/GEOM_HANDOFF.md).

```bash
cd geoms/geom
cargo test --release
cargo clippy --all-targets -- -D warnings
GEOM_WRITE_GOLDEN=1 cargo test --test goldens   # regenerate golden/*.json from tests/common
```

```rust
use geom::{eval::Tier, model::*, select::Selector};
let mut m = Model::new_2d();
m.param("w", 500e-9).param("t", 300e-9);
let clad = m.rect("clad", ("-w/2-1e-6", "-2e-6"), ("w/2+1e-6", "t+1e-6"));
let bx   = m.rect("box",  ("-w/2-1e-6", "-2e-6"), ("w/2+1e-6", 0.0));
let core = m.rect("core", ("-w/2", 0.0), ("w/2", "t"));
m.overlay(&[("clad", clad), ("BOX", bx), ("Si", core)]);       // painter order: later wins
let ev = m.eval(Tier::Complex)?;
let lm = ev.label_map()?;                                       // names <-> ints
let si_box = lm.boundary("BOX|Si");                             // Vec<i32>
let walls = ev.select_labels(&Selector::parse("outer")?)?;
```

## Decisions and conventions

* **Tolerance**: `tol = 1e-9 * L`, `L` the diagonal of the model; undecidable band `64 tol` is refused, not guessed (see `arrange.rs`).
* **Edges** run along the canonical direction of their curve (lines: angle in (-90 deg, 90 deg]; circles and ellipses counter-clockwise from
  the major axis, i.e. from angle 0 for circles). `left` / `right` are the regions on either side; the edge normal points from `left` to `right`.
  A `None` side is the void (outside, or a removed hole).
* **Names**: boundary edges (one side void) `<feature instance>.<role>` taken from the origin that belongs to the inside region;
  interfaces `A|B` with the region names sorted. Roles are fixed per primitive (see `prim.rs`). Pattern instances append `[i]`.
* **Ids** are dense and deterministic; edges are ordered by (first origin feature id, instance, role, fraction along the origin segment),
  which does not depend on parameter values.
* `A & extreme(+y)` evaluates `extreme` inside the set `A`.
* `Pml { window, sides, thickness, layers, grading }` splits every material region into the physical part and one
  `<region>.pml_<sides>` region per side combination (kind `Pml`); `Evaluated::pml_regions()` carries the data. Pieces that end up empty
  (painted over) are dropped.

## Limits worth knowing

* A line whose direction passes through the vertical flips its canonical direction (so `left` and `right` swap); a model swept through
  that angle changes its topology signature.
* Ellipse intersections are found numerically (720 samples per curve); two roots closer than a sample interval can be missed.
* `robust` predicates are not used: all decisions are tolerance based on interned geometry.
