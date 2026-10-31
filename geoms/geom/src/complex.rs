//! The evaluated region complex of a 2D model: regions (planar faces, possibly with holes), the edges that separate them
//! (each used by exactly two region sides, one of which may be the void), vertices, and the closed boundary loops of every
//! region. Areas, lengths and the self-checks are analytic (Green's theorem on the exact curves).
//!
//! Orientation convention: an edge runs along the canonical direction of its curve; `left` is the region on the left of that
//! direction, `right` the region on the right, and the edge normal points from `left` to `right` (the first listed region to
//! the second). `None` is the void (outside, or a removed hole).

use crate::curve::Curve;
use crate::math::*;
use crate::model::{FeatureId, RegionKind};

/// One origin of an edge: the feature (and pattern instance) whose boundary piece `role` produced it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Prov {
    pub feature: FeatureId,
    pub instance: String,
    pub role: String,
}

impl Prov {
    pub fn name(&self) -> String {
        format!("{}.{}", self.instance, self.role)
    }
}

#[derive(Clone, Debug)]
pub struct Region {
    pub name: String,
    pub kind: RegionKind,
    pub feature: FeatureId,
    pub pinned_id: Option<i32>,
    /// index of the region in the scene (before empty generated regions were dropped)
    pub scene_index: usize,
    /// area
    pub measure: f64,
    pub bbox: Aabb,
    /// loops bounding this region (outer boundaries have positive area, holes negative)
    pub loops: Vec<u32>,
    pub parent: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub name: String,
    pub curve: u32,
    pub t0: f64,
    pub t1: f64,
    pub v0: u32,
    pub v1: u32,
    pub left: Option<u32>,
    pub right: Option<u32>,
    pub prov: Vec<Prov>,
    pub length: f64,
    pub bbox: Aabb,
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pub p: V3,
    pub name: String,
    pub edges: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Loop {
    pub region: u32,
    /// `(edge, forward)`; the region is on the left of the travel direction
    pub edges: Vec<(u32, bool)>,
    /// signed area enclosed (negative for holes)
    pub area: f64,
    /// for a hole: index of the enclosing outer loop
    pub outer: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Complex {
    pub dim: u8,
    pub tol: f64,
    /// point about which area integrals are taken (centre of the model)
    pub origin: V2,
    pub curves: Vec<Curve>,
    pub regions: Vec<Region>,
    pub edges: Vec<Edge>,
    pub vertices: Vec<Vertex>,
    pub loops: Vec<Loop>,
    /// scene region index -> final region id (`None` for holes and for dropped empty generated regions)
    pub scene_to_region: Vec<Option<u32>>,
}

pub fn curve_kind(c: &Curve) -> &'static str {
    match c {
        Curve::Line { .. } => "line",
        Curve::Circle { .. } => "circle",
        Curve::Ellipse { .. } => "ellipse",
    }
}

impl Complex {
    pub fn curve_of(&self, e: u32) -> (&Curve, f64, f64) {
        let ed = &self.edges[e as usize];
        (&self.curves[ed.curve as usize], ed.t0, ed.t1)
    }

    pub fn edge_point(&self, e: u32, t: f64) -> V3 {
        to3(self.curve_of(e).0.eval(t))
    }

    /// Exact closest point of edge `e` (the arc between its end points, not the whole curve) to `p`, with its parameter
    /// (unwrapped into `[t0, t1]`). Stable at periodic seams: parameters are taken relative to `t0`.
    pub fn edge_closest(&self, e: u32, p: V2) -> (f64, V2) {
        let (c, t0, t1) = self.curve_of(e);
        let (t, q) = c.closest(p);
        if c.in_range(t, t0, t1, 1e-12) {
            let u = c.param_in(t, t0).min(t1);
            return (u, if u == t { q } else { c.eval(u) });
        }
        let (a, b) = (c.eval(t0), c.eval(t1));
        if dist2(p, a) <= dist2(p, b) {
            (t0, a)
        } else {
            (t1, b)
        }
    }

    pub fn project_edge(&self, e: u32, p: V2) -> V2 {
        self.edge_closest(e, p).1
    }

    /// Unit normal at parameter `t` pointing from the left region to the right region.
    pub fn edge_normal(&self, e: u32, t: f64) -> V2 {
        let tg = self.curve_of(e).0.tangent(t);
        [tg[1], -tg[0]]
    }

    /// Outward normal for boundary edges (towards the void), the left-to-right normal for interfaces.
    pub fn outward_normal(&self, e: u32, t: f64) -> V2 {
        let n = self.edge_normal(e, t);
        match self.edges[e as usize].left {
            None => mul2(n, -1.0),
            Some(_) => n,
        }
    }

    // ---- uniform "facet" access (the codimension-1 entities: edges of a 2D complex) used by selectors ----

    pub fn facet_count(&self) -> usize {
        self.edges.len()
    }
    pub fn facet_name(&self, i: u32) -> &str {
        &self.edges[i as usize].name
    }
    pub fn facet_prov(&self, i: u32) -> &[Prov] {
        &self.edges[i as usize].prov
    }
    pub fn facet_sides(&self, i: u32) -> (Option<u32>, Option<u32>) {
        let e = &self.edges[i as usize];
        (e.left, e.right)
    }
    pub fn facet_bbox(&self, i: u32) -> Aabb {
        self.edges[i as usize].bbox
    }
    pub fn facet_measure(&self, i: u32) -> f64 {
        self.edges[i as usize].length
    }
    pub fn facet_kind(&self, i: u32) -> &'static str {
        curve_kind(&self.curves[self.edges[i as usize].curve as usize])
    }
    pub fn facet_radius(&self, i: u32) -> Option<f64> {
        match &self.curves[self.edges[i as usize].curve as usize] {
            Curve::Circle { r, .. } => Some(*r),
            _ => None,
        }
    }
    /// Outward normals sampled at the end points and the middle of the facet.
    pub fn facet_normals(&self, i: u32) -> Vec<V3> {
        let e = &self.edges[i as usize];
        [e.t0, 0.5 * (e.t0 + e.t1), e.t1].iter().map(|&t| to3(self.outward_normal(i, t))).collect()
    }
    pub fn facet_vertices(&self, i: u32) -> Vec<u32> {
        let e = &self.edges[i as usize];
        if e.v0 == e.v1 {
            vec![e.v0]
        } else {
            vec![e.v0, e.v1]
        }
    }

    /// Region containing `p` by testing the boundary loops (chord-flattened); agrees with `Evaluated::classify` away from the boundary.
    pub fn locate(&self, p: V2) -> Option<u32> {
        (0..self.regions.len() as u32).find(|&r| {
            self.regions[r as usize]
                .loops
                .iter()
                .filter(|&&l| crate::arrange::point_in_loop(&self.loops[l as usize], &self.edges, &self.curves, p))
                .count()
                % 2
                == 1
        })
    }

    pub fn region_area(&self, r: u32) -> f64 {
        self.regions[r as usize].measure
    }

    /// Total length of the interface between two regions.
    pub fn interface_length(&self, a: u32, b: u32) -> f64 {
        self.edges
            .iter()
            .filter(|e| (e.left == Some(a) && e.right == Some(b)) || (e.left == Some(b) && e.right == Some(a)))
            .map(|e| e.length)
            .sum()
    }

    /// Twice-signed-area integral of one edge in its canonical direction.
    pub fn edge_green(&self, e: u32) -> f64 {
        let (c, t0, t1) = self.curve_of(e);
        c.green(t0, t1, self.origin)
    }

    /// Always-on self-checks (cheap): every edge separates two different sides, loops are closed and consistently oriented,
    /// region areas are positive, and the regions tile their union (the sum of region areas equals the area enclosed by the
    /// edges that touch the void).
    pub fn check(&self) -> Result<(), String> {
        let scale = self.regions.iter().map(|r| r.bbox.diagonal()).fold(0.0, f64::max).max(1e-300);
        for (i, e) in self.edges.iter().enumerate() {
            if e.left == e.right {
                return Err(format!("edge {i} ('{}') has the same region on both sides", e.name));
            }
            if e.length.is_nan() || e.length <= 0.0 {
                return Err(format!("edge {i} ('{}') has no length", e.name));
            }
        }
        for (i, l) in self.loops.iter().enumerate() {
            for (k, &(e, fwd)) in l.edges.iter().enumerate() {
                let (n_e, n_f) = l.edges[(k + 1) % l.edges.len()];
                let ed = &self.edges[e as usize];
                let end = if fwd { ed.v1 } else { ed.v0 };
                let nd = &self.edges[n_e as usize];
                let start = if n_f { nd.v0 } else { nd.v1 };
                if end != start {
                    return Err(format!(
                        "loop {i} of region '{}' is not closed at edge '{}'",
                        self.regions[l.region as usize].name, ed.name
                    ));
                }
            }
        }
        let mut total = 0.0;
        for r in &self.regions {
            let a: f64 = r.loops.iter().map(|&l| self.loops[l as usize].area).sum();
            if (a - r.measure).abs() > 1e-12 * r.measure.abs().max(1e-300) + 1e-14 * scale * scale {
                return Err(format!("region '{}' area {} disagrees with its loops {a}", r.name, r.measure));
            }
            if r.measure < 0.0 {
                return Err(format!("region '{}' has negative area {}", r.name, r.measure));
            }
            total += r.measure;
        }
        // area of the union from the edges that border the void
        let mut outer = 0.0;
        for (i, e) in self.edges.iter().enumerate() {
            let g = 0.5 * self.edge_green(i as u32);
            if e.left.is_some() && e.right.is_none() {
                outer += g;
            } else if e.left.is_none() && e.right.is_some() {
                outer -= g;
            }
        }
        if (outer - total).abs() > 1e-10 * total.abs().max(1e-300) + 1e-13 * scale * scale {
            return Err(format!("region areas sum to {total} but the outer boundary encloses {outer}"));
        }
        Ok(())
    }
}
