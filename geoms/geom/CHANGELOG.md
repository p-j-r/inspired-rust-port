# Changelog

## Steps C and D (prismatic 3D, layer stacks) - delivered in part, see "Not delivered"

* 3D feature lists: `Box`, `Cylinder` (axis z), `Extrude` of any 2D profile, Transform (translate, rotate about z, mirror, uniform
  scale), Pattern, Booleans, `Overlay`, `Pml` (26 side combinations in 3D), `Cut` (axis-aligned planes; faces `<feature>.cut`, tag
  `symmetry`), `Periodic` (verified congruent by sampling both sides), `Size`, `Tag`. Roles: Box `xmin xmax ymin ymax zmin zmax`, Cylinder
  `side bottom top`, Extrude `side:<role> bottom top`.
* Evaluation as a **layer stack** (`stack.rs`): z levels, one 2D arrangement per layer, and the *overlay*, the common refinement of all layers
  (the 2D arrangement run with composite per-layer classes). `Evaluated::layer_stack()` returns the layers (z range, partition, region of every
  overlay cell) and the overlay. Detected for every 3D model this build can represent; non-prismatic content is refused (below).
* 3D faces derived from the stack: lateral faces = overlay edges swept over runs of layers (merged while the sides and name agree), caps =
  overlay cells at levels where the region below differs from the region above. Names follow the 2D rules (`A|B`, `<feature>.<role>`).
  Volumes, face areas, projection (`oracle().project(.., Face(f))`), fractions (two per face), `classify`, selectors and `LabelMap` all work on faces.
* Self-check for 3D: every face separates two sides, volumes are positive, and every region is closed (the outward area vectors sum to
  zero in x, y and z, to a few tolerances).
* `surfaces_for_stuffing()` (planes and cylinders, coincident faces share one surface, with the extent of its faces), 3D display tessellation,
  `diff` kind `ZOnly` (same topology, same xy overlay geometry, different z).
* Golden models: three Bragg periods (symmetric, asymmetric, uniform) with a Floquet pair, cylinder cavity, concentric cylinders.
* Property test: random 3D box / cylinder models never panic or fail the closure check.

### Not delivered (needs the general 3D arrangement)

* Sphere, Cone, Torus, Revolve, tapered sweeps, rotations about x or y, cylinders along x or y, slanted cut planes, skew cylinders and the
  `SurfSurf` intersection curves. A model that needs them is refused with a message (`Cut ... slanted`, `Cylinder ... axis`, `must be the input
  of an Extrude`). The goldens "sphere in a box", "torus", "concentric spheres", "box with a slanted cut" and "two skew cylinders" are therefore
  not provided; the slanted cut and skew cases exist only as refusals.
* 3D edges and vertices of the B-rep (the 2D edges and vertices of the overlay are available through `layer_stack().overlay`), 3D curve
  queries and 3D `discretize_edge`; `adjacent(...)` between 3D faces; the Euler relation (replaced by the closure check).
* Because everything prismatic is also a layer stack, `layer_stack()` never returns `None` for a model that evaluates.

## Step B (tiers, cache, queries, sweep support) - delivered

* Normalisation of region expressions (flatten, drop empties, `A cap A = A`, bounding-box-disjoint intersections and difference
  operands) and pruning of primitives nothing refers to; the primitive and pair-intersection caches are keyed by content hashes
  (`Evaluator::stats()` counts builds and hits; a one-parameter change rebuilds one primitive and the pairs that involve it).
* `FeatureList::to_rust` (builder code; round trip checked as pure data by a small interpreter in `tests/common/interp.rs`).
* Queries: `length`, `volume`, `interface_area`, curve evaluation / tangent / curvature, `min_radius_of_curvature`, `chord_deviation`,
  `facet_deviation`, `point_at_mid`, `project_midpoint` (seam safe), `discretize_edge` (deviation and size, shared by both neighbours).
* Display tessellation with region / edge ids (`Evaluated::tessellate`).
* Addendum 1: `topology_signature`, stable ids within a signature (edge order is by first origin and the fraction along the origin segment, not by
  position), `fractions_of` / `position_at` (round trip to 1e-12 L), `diff -> ChangeReport`, documented continuous parametrisation (see `query.rs`).
* Property tests: random feature lists never panic or fail a self-check (6000 cases run), union area associativity / commutativity,
  `A cap A = A`, unrelated features keep existing names.
* Bug found by the property tests and fixed: the collinear re-merge removed the wrong edge when the second fragment had the lower index.
* New refusal: a primitive thinner than `64 tol` (about `6e-8 L`) is refused by name; before it produced an inconsistent loop.
* Conflict with the brief: the tolerance is relative to the model size, so adding a distant feature changes `tol`; input within a few
  `1e-9 L` of degenerate can therefore change meaning (the name-stability property is tested on well-separated input).
* Measured (release, 4-core VM): 49-feature model 0.77 ms cold, 4-term selector over 250 edges 6 us, 200-hole plate 0.94 ms,
  100-point width sweep 70 ms. The pair cache saves little at this size because the arrangement dominates; see the README.

## Step A (2D kernel) - delivered

* Added: curves (line, circle, ellipse; canonical parametrisation, exact Green-theorem areas, closest point, curve-curve intersection
  with tangency and near-coincidence reporting), 2D profiles (Rect, RoundedRect, Circle, Ellipse, Polygon) with fixed role names,
  Transform / Pattern / Union / Difference / Intersect, `Overlay` (painter order, Hole/Void/Pml kinds, pinned ids), `Pml` (window +
  sides -> physical region and one `<region>.pml_<sides>` piece per side combination, empty pieces dropped), `Cut` (`<feature>.cut`,
  tag `symmetry`), `Periodic` (verified congruent), `Size`, `Tag`.
* n-ary planar arrangement with interned curves and vertices, owner-oriented side decisions, tolerance `1e-9 L`, refusal band
  `64 tol` (tangent contact, near-coincident curves, corners that nearly touch, self-crossing profiles, thin slivers) with errors that name the features.
* Region complex: regions with loops (holes assigned to outer loops), edges (left to right normal convention), vertices, auto-names
  (`feature.role`, `A|B`, `a/b`), provenance, always-on self checks (`Complex::check`).
* Selectors: text form with deterministic printing and serde form, `resolve` with typed sets and warnings, `suggest`.
* `LabelMap`: deterministic domain and boundary ints, pins, name <-> int in both directions, groups with their rule.
* Oracle (`project` on edges / vertices / regions), serde round trip (JSON bit-exact; XML value encoding checked), builder.
* Golden models in `golden/`: SOI strip, SOI rib with unequal sloped walls, the four ring chips with PML, Bragg cross-section,
  plate with hole, disc, annulus, concentric discs; closed-form areas to 1e-10.
* Conflict with the brief / design choices (see README "Decisions"): the brief's `Pml{target,...}` is implemented as a window plus
  region splitting; edge ids are ordered by (first origin, fraction along the origin segment) rather than by centroid so they do not
  depend on parameter values (addendum item 2).
