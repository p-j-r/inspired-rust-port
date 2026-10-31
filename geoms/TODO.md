# geom: remaining work

Everything below is **not implemented**; each item is refused at evaluation with a message that names the features involved (nothing is
approximated). Ordered by value for typical FEM models. Sizes are rough (S: a day, M: several days, L: a week or more).

## 1. Curved solids touching, crossing or cut by other faces (L, the core missing piece)

What is refused today: a sphere / torus / cone / cylinder / revolved solid that touches a wall, is cut by a plane (a half or quarter model of a
sphere), or meets another curved solid; a cone standing on a floor; two skew or tilted cylinders; curved solids mixed with rotated, tapered or
slanted polyhedra; cylinders / cones / extrusions with different sweep axes in one model.

What it needs: splitting a curved face along an intersection curve in the surface's own parameter space, and the loops that result.
* plane x sphere = circle, plane x cylinder / cone / torus = conic or quartic, cylinder x cylinder = general curve (march with Newton, the
  brief's `SurfSurf`). Imprint these curves on both faces, build loops per face (the 2D arrangement in `arrange.rs` can be reused when the
  parameter space is flat: plane, and (angle, height) for cylinders and cones, with a periodic seam).
* Region sides for coincident curved faces by orientation (as `arrange.rs` already does for lines and arcs).
* Cheapest useful first step: **planes through the axis of a body** (half / quarter models for symmetry): the cut curve is a circle or a pair
  of lines, faces are patches of the original surface bounded by it. Then plane x sphere at arbitrary offset, then cylinder x plane.
* A cone or cylinder standing on a flat face: coincident circle edge, no tangency issue; the arrangement of the flat face gets a circular hole.
  This is likely the most requested case after symmetry cuts (a driver flush in a baffle).

## 2. Partial revolves and revolves of curved profiles (M)

`Revolve` is a full turn of a `Rect` / `Polygon` profile about the sweep axis (segments become cone sides, cylinder walls and flat rings).
Missing: an angle (`start` / `end` faces, wedge models), profiles with arcs (then spheres, tori and ellipsoid caps appear as faces), and
profiles touching the axis tangentially.

## 3. Mixed sweep axes (M, comes with item 1)

All prismatic solids (Cylinder, Extrude, Cone, Revolve) of a model must share one sweep axis because the model is evaluated as a stack along
it. A cylinder along x through a wall that is a z-prism needs the general arrangement of item 1.

## 4. Smaller items

* `Tier::Bounds` still builds the whole scene; make it parameters-only (S).
* Arrangement cache: the pair cache saves little at current sizes; cache the arrangement per connected group if a profile of a large model asks (M).
* Edge direction of a line is canonical (angle in (-90, 90] degrees): sweeping a rotation through the vertical swaps `left` / `right` and changes the
  topology signature. Make the edge follow the direction of its first origin (M, touches loops and the signature).
* Elliptic arcs are intersected by sampling (720 points); replace by a closed-form quartic or a subdivision with a guaranteed bracket (M).
* The torus clearance test is a bounding-box test; use the exact distance from a face to the torus (S).
* Surface-level curvature queries for 3D (only `facet_radius` and the body radii exist) and 3D `chord_deviation` / `facet_deviation` / `discretize_edge`
  for 3D edges (M, needed by the curved-element refactor on the femcore side).
* Tapered extrusions of concave profiles (offset curves with self-overlap) and of profiles with arcs (S / M).
* Boolean `Cut` with a curved cutter, and fillets / chamfers / shells / offsets in 3D are out of scope of the brief.
* A GUI-facing layer (descriptor tables per `Op`, pick-to-selector) is on the integration side.
