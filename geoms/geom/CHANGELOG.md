# Changelog

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
