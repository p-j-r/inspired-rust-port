# Changelog

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
