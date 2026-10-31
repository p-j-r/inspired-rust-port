//! 2D primitives in the world frame: a local shape (rectangle, rounded rectangle, circle, ellipse, polygon), the similarity
//! transform that placed it, its directed boundary pieces with role names (interior always on the left), an exact signed
//! distance in the local frame, and a bounding box. Curves are *not* interned here; the arrangement does that once the
//! model tolerance is known.

use crate::curve::Curve;
use crate::error::GeomResult;
use crate::math::*;

/// Local-frame shape. Polygons are counter-clockwise with `names[i]` the role of the edge `pts[i] -> pts[i+1]`.
#[derive(Clone, Debug, PartialEq)]
pub enum LShape {
    Rect { lo: V2, hi: V2 },
    Rounded { lo: V2, hi: V2, r: f64 },
    Circle { c: V2, r: f64 },
    Ellipse { c: V2, a: f64, b: f64 },
    Poly { pts: Vec<V2>, names: Vec<String> },
}

/// A directed boundary piece in world coordinates; the primitive's interior is on the left of the travel direction.
#[derive(Clone, Debug, PartialEq)]
pub enum DGeom {
    Line {
        p0: V2,
        p1: V2,
    },
    /// counter-clockwise arc from angle `a0` over `sweep` (`TAU`: the full circle)
    Arc {
        c: V2,
        r: f64,
        a0: f64,
        sweep: f64,
    },
    /// full ellipse
    Ell {
        c: V2,
        a: f64,
        b: f64,
        phi: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DSeg {
    pub geom: DGeom,
    pub role: String,
}

#[derive(Clone, Debug)]
pub struct Prim {
    pub feature: u32,
    /// instance name: feature name plus pattern indices (`hole[3]`)
    pub name: String,
    pub shape: LShape,
    pub xf: Xf2,
    pub segs: Vec<DSeg>,
    pub bbox: Aabb,
    /// helper primitives (cut half-planes) do not count towards the model size
    pub aux: bool,
    pub key: u64,
}

fn poly_area(p: &[V2]) -> f64 {
    0.5 * (0..p.len()).map(|i| cross2(p[i], p[(i + 1) % p.len()])).sum::<f64>()
}

impl LShape {
    /// Local segments (counter-clockwise) as `(geometry, role)`.
    fn local_segs(&self) -> Vec<(DGeom, String)> {
        let line = |p0: V2, p1: V2, r: &str| (DGeom::Line { p0, p1 }, r.to_string());
        let arc = |c: V2, r: f64, a0: f64, name: &str| {
            (DGeom::Arc { c, r, a0, sweep: std::f64::consts::FRAC_PI_2 }, name.to_string())
        };
        match self {
            LShape::Rect { lo, hi } => vec![
                line([lo[0], lo[1]], [hi[0], lo[1]], "bottom"),
                line([hi[0], lo[1]], [hi[0], hi[1]], "right"),
                line([hi[0], hi[1]], [lo[0], hi[1]], "top"),
                line([lo[0], hi[1]], [lo[0], lo[1]], "left"),
            ],
            LShape::Rounded { lo, hi, r } => {
                let r = *r;
                let (x0, y0, x1, y1) = (lo[0], lo[1], hi[0], hi[1]);
                let mut v = vec![];
                let h = std::f64::consts::FRAC_PI_2;
                let push_line = |v: &mut Vec<(DGeom, String)>, p0: V2, p1: V2, role: &str| {
                    if dist2(p0, p1) > 0.0 {
                        v.push(line(p0, p1, role));
                    }
                };
                push_line(&mut v, [x0 + r, y0], [x1 - r, y0], "bottom");
                v.push(arc([x1 - r, y0 + r], r, -h, "c_br"));
                push_line(&mut v, [x1, y0 + r], [x1, y1 - r], "right");
                v.push(arc([x1 - r, y1 - r], r, 0.0, "c_tr"));
                push_line(&mut v, [x1 - r, y1], [x0 + r, y1], "top");
                v.push(arc([x0 + r, y1 - r], r, h, "c_tl"));
                push_line(&mut v, [x0, y1 - r], [x0, y0 + r], "left");
                v.push(arc([x0 + r, y0 + r], r, 2.0 * h, "c_bl"));
                v
            }
            LShape::Circle { c, r } => vec![(DGeom::Arc { c: *c, r: *r, a0: 0.0, sweep: TAU }, "edge".into())],
            LShape::Ellipse { c, a, b } => vec![(DGeom::Ell { c: *c, a: *a, b: *b, phi: 0.0 }, "edge".into())],
            LShape::Poly { pts, names } => (0..pts.len())
                .map(|i| (DGeom::Line { p0: pts[i], p1: pts[(i + 1) % pts.len()] }, names[i].clone()))
                .collect(),
        }
    }

    /// Signed distance in the local frame (negative inside).
    fn sdf(&self, p: V2) -> f64 {
        let boxd = |lo: V2, hi: V2| {
            let (c, h) =
                ([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])], [0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1])]);
            let q = [(p[0] - c[0]).abs() - h[0], (p[1] - c[1]).abs() - h[1]];
            q[0].max(0.0).hypot(q[1].max(0.0)) + q[0].max(q[1]).min(0.0)
        };
        match self {
            LShape::Rect { lo, hi } => boxd(*lo, *hi),
            LShape::Rounded { lo, hi, r } => boxd([lo[0] + r, lo[1] + r], [hi[0] - r, hi[1] - r]) - r,
            LShape::Circle { c, r } => dist2(p, *c) - r,
            LShape::Ellipse { c, a, b } => Curve::Ellipse { c: *c, a: *a, b: *b, phi: 0.0 }.sdf(p).unwrap_or(0.0),
            LShape::Poly { pts, .. } => {
                let n = pts.len();
                let (mut d2, mut inside) = (f64::INFINITY, false);
                for i in 0..n {
                    let (a, b) = (pts[i], pts[(i + 1) % n]);
                    let ab = sub2(b, a);
                    let t = (dot2(sub2(p, a), ab) / dot2(ab, ab)).clamp(0.0, 1.0);
                    let q = add2(a, mul2(ab, t));
                    d2 = d2.min(dot2(sub2(p, q), sub2(p, q)));
                    if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
                        inside = !inside;
                    }
                }
                if inside {
                    -d2.sqrt()
                } else {
                    d2.sqrt()
                }
            }
        }
    }

    fn hash(&self, h: &mut Hasher64) {
        match self {
            LShape::Rect { lo, hi } => {
                h.u64(1).f64(lo[0]).f64(lo[1]).f64(hi[0]).f64(hi[1]);
            }
            LShape::Rounded { lo, hi, r } => {
                h.u64(2).f64(lo[0]).f64(lo[1]).f64(hi[0]).f64(hi[1]).f64(*r);
            }
            LShape::Circle { c, r } => {
                h.u64(3).f64(c[0]).f64(c[1]).f64(*r);
            }
            LShape::Ellipse { c, a, b } => {
                h.u64(4).f64(c[0]).f64(c[1]).f64(*a).f64(*b);
            }
            LShape::Poly { pts, names } => {
                h.u64(5);
                for (p, n) in pts.iter().zip(names) {
                    h.f64(p[0]).f64(p[1]).str(n);
                }
            }
        }
    }
}

impl DGeom {
    pub fn curve_range(&self) -> (Curve, f64, f64) {
        match self {
            DGeom::Line { p0, p1 } => {
                let (c, rev) = crate::curve::canonical_line(*p0, sub2(*p1, *p0));
                let (a, b) = (c.closest(*p0).0, c.closest(*p1).0);
                let _ = rev;
                (c, a.min(b), a.max(b))
            }
            DGeom::Arc { c, r, a0, sweep } => {
                let a = a0.rem_euclid(TAU);
                (Curve::Circle { c: *c, r: *r }, a, a + sweep)
            }
            DGeom::Ell { c, a, b, phi } => (crate::curve::canonical_ellipse(*c, *a, *b, *phi, 0.0), 0.0, TAU),
        }
    }
    fn map(&self, xf: &Xf2) -> DGeom {
        match self {
            DGeom::Line { p0, p1 } => DGeom::Line { p0: xf.apply(*p0), p1: xf.apply(*p1) },
            DGeom::Arc { c, r, a0, sweep } => {
                let c2 = xf.apply(*c);
                let at = |a: f64| xf.apply([c[0] + r * a.cos(), c[1] + r * a.sin()]);
                let start = if xf.mirrored() { at(a0 + sweep) } else { at(*a0) };
                let v = sub2(start, c2);
                DGeom::Arc { c: c2, r: r * xf.scale(), a0: v[1].atan2(v[0]), sweep: *sweep }
            }
            DGeom::Ell { c, a, b, .. } => {
                let e1 = xf.apply_dir([1.0, 0.0]);
                DGeom::Ell { c: xf.apply(*c), a: a * xf.scale(), b: b * xf.scale(), phi: e1[1].atan2(e1[0]) }
            }
        }
    }
}

impl Prim {
    pub fn new(feature: u32, name: String, shape: LShape, xf: Xf2, aux: bool) -> GeomResult<Prim> {
        let mut segs: Vec<DSeg> =
            shape.local_segs().into_iter().map(|(g, role)| DSeg { geom: g.map(&xf), role }).collect();
        if xf.mirrored() {
            // reflection reverses the travel direction of lines (arcs were re-anchored above); restore counter-clockwise order
            for s in &mut segs {
                if let DGeom::Line { p0, p1 } = &mut s.geom {
                    std::mem::swap(p0, p1);
                }
            }
            segs.reverse();
        }
        let mut bbox = Aabb::EMPTY;
        for s in &segs {
            let (c, a, b) = s.geom.curve_range();
            bbox = bbox.union(&c.bbox(a, b));
        }
        let mut h = Hasher64::new();
        shape.hash(&mut h);
        h.f64(xf.m[0][0])
            .f64(xf.m[0][1])
            .f64(xf.m[1][0])
            .f64(xf.m[1][1])
            .f64(xf.t[0])
            .f64(xf.t[1])
            .str(&name)
            .u64(aux as u64)
            .u64(feature as u64);
        Ok(Prim { feature, name, shape, xf, segs, bbox, aux, key: h.finish() })
    }

    /// Signed distance estimate in world units (negative inside); exact for polygons, rectangles and circles.
    pub fn sdf(&self, p: V2) -> f64 {
        self.xf.scale() * self.shape.sdf(self.xf.inverse().apply(p))
    }

    /// Signed distance, skipping the evaluation when `p` is farther than `reach` from the bounding box.
    pub fn sdf_near(&self, p: V2, reach: f64) -> f64 {
        let b = &self.bbox;
        let d = (b.lo[0] - p[0]).max(p[0] - b.hi[0]).max(b.lo[1] - p[1]).max(p[1] - b.hi[1]);
        if d > reach {
            d
        } else {
            self.sdf(p)
        }
    }
}

/// Counter-clockwise copy of a polygon with the edge names reordered to match.
pub fn ccw_polygon(pts: &[V2], names: &[String]) -> (Vec<V2>, Vec<String>) {
    if poly_area(pts) >= 0.0 {
        return (pts.to_vec(), names.to_vec());
    }
    let n = pts.len();
    let q: Vec<V2> = pts.iter().rev().copied().collect();
    // edge j of the reversed polygon joins pts[n-1-j] and pts[n-2-j], which is the original edge n-2-j
    let nm = (0..n).map(|j| names[(2 * n - 2 - j) % n].clone()).collect();
    (q, nm)
}

pub fn polygon_area(p: &[V2]) -> f64 {
    poly_area(p)
}
