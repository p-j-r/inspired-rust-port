//! From a feature list to a *scene*: parameters evaluated, every profile instantiated (with transforms and pattern copies)
//! into world-frame primitives, and one Boolean expression over primitives per overlay region (Cut and Pml already folded
//! in). This is the Classify tier: it needs no arrangement. `Scene::classify` answers inside/outside with the painter rule.

use crate::error::{GeomError, GeomResult};
use crate::expr::Env;
use crate::math::*;
use crate::model::*;
use crate::prim::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Boolean expression over primitive indices.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Empty,
    Prim(usize),
    Union(Vec<Node>),
    Inter(Vec<Node>),
    Diff(Box<Node>, Vec<Node>),
}

impl Node {
    pub fn eval(&self, bit: &dyn Fn(usize) -> bool) -> bool {
        match self {
            Node::Empty => false,
            Node::Prim(i) => bit(*i),
            Node::Union(v) => v.iter().any(|n| n.eval(bit)),
            Node::Inter(v) => !v.is_empty() && v.iter().all(|n| n.eval(bit)),
            Node::Diff(a, b) => a.eval(bit) && !b.iter().any(|n| n.eval(bit)),
        }
    }
    /// Conservative bounding box.
    pub fn bbox(&self, prims: &[Arc<Prim>]) -> Aabb {
        match self {
            Node::Empty => Aabb::EMPTY,
            Node::Prim(i) => prims[*i].bbox3(),
            Node::Union(v) => v.iter().fold(Aabb::EMPTY, |a, n| a.union(&n.bbox(prims))),
            Node::Inter(v) => {
                let mut r: Option<Aabb> = None;
                for n in v {
                    let b = n.bbox(prims);
                    r = Some(match r {
                        None => b,
                        Some(a) => Aabb {
                            lo: [0, 1, 2].map(|k| a.lo[k].max(b.lo[k])),
                            hi: [0, 1, 2].map(|k| a.hi[k].min(b.hi[k])),
                        },
                    });
                }
                r.unwrap_or(Aabb::EMPTY)
            }
            Node::Diff(a, _) => a.bbox(prims),
        }
    }
    /// Normalisation: flatten nested Unions / Intersections, drop empty operands, remove duplicates (`A cap A = A`), turn
    /// bounding-box-disjoint intersections into `Empty` and drop difference operands that cannot touch the minuend.
    pub fn simplify(self, prims: &[Arc<Prim>]) -> Node {
        fn flat(v: Vec<Node>, union: bool, prims: &[Arc<Prim>]) -> Vec<Node> {
            let mut out: Vec<Node> = vec![];
            for n in v {
                match n.simplify(prims) {
                    Node::Union(c) if union => out.extend(c),
                    Node::Inter(c) if !union => out.extend(c),
                    Node::Empty if union => {}
                    n => out.push(n),
                }
            }
            let mut uniq: Vec<Node> = vec![];
            for n in out {
                if !uniq.contains(&n) {
                    uniq.push(n);
                }
            }
            uniq
        }
        match self {
            Node::Union(v) => {
                let mut u = flat(v, true, prims);
                match u.len() {
                    0 => Node::Empty,
                    1 => u.remove(0),
                    _ => Node::Union(u),
                }
            }
            Node::Inter(v) => {
                let mut u = flat(v, false, prims);
                if u.contains(&Node::Empty) || u.is_empty() {
                    return Node::Empty;
                }
                let boxes: Vec<Aabb> = u.iter().map(|n| n.bbox(prims)).collect();
                for i in 0..boxes.len() {
                    for j in i + 1..boxes.len() {
                        if boxes[i].is_empty() || boxes[j].is_empty() || !boxes[i].overlaps(&boxes[j]) {
                            return Node::Empty;
                        }
                    }
                }
                if u.len() == 1 {
                    u.remove(0)
                } else {
                    Node::Inter(u)
                }
            }
            Node::Diff(a, b) => {
                let a = a.simplify(prims);
                if a == Node::Empty {
                    return a;
                }
                let ab = a.bbox(prims);
                let b: Vec<Node> = flat(b, true, prims)
                    .into_iter()
                    .filter(|n| *n != Node::Empty && n.bbox(prims).overlaps(&ab))
                    .collect();
                if b.contains(&a) {
                    return Node::Empty;
                }
                if b.is_empty() {
                    a
                } else {
                    Node::Diff(Box::new(a), b)
                }
            }
            n => n,
        }
    }

    pub fn remap(&mut self, map: &[usize]) {
        match self {
            Node::Prim(i) => *i = map[*i],
            Node::Union(v) | Node::Inter(v) => v.iter_mut().for_each(|n| n.remap(map)),
            Node::Diff(a, b) => {
                a.remap(map);
                b.iter_mut().for_each(|n| n.remap(map));
            }
            Node::Empty => {}
        }
    }

    pub fn leaves(&self, out: &mut Vec<usize>) {
        match self {
            Node::Empty => {}
            Node::Prim(i) => out.push(*i),
            Node::Union(v) | Node::Inter(v) => v.iter().for_each(|n| n.leaves(out)),
            Node::Diff(a, b) => {
                a.leaves(out);
                b.iter().for_each(|n| n.leaves(out));
            }
        }
    }
}

/// Data attached to a region generated by `Pml`.
#[derive(Clone, Debug, PartialEq)]
pub struct PmlInfo {
    pub feature: FeatureId,
    /// per axis -1, 0 or +1: the side(s) of the window the layer sits on
    pub sides: [i8; 3],
    pub window: Aabb,
    pub thickness: f64,
    pub layers: f64,
    pub grading: f64,
}

#[derive(Clone, Debug)]
pub struct RegionX {
    pub name: String,
    pub kind: RegionKind,
    pub feature: FeatureId,
    pub pinned_id: Option<i32>,
    pub expr: Node,
    /// name of the region this one was cut from (Pml pieces)
    pub parent: Option<String>,
    pub pml: Option<PmlInfo>,
}

#[derive(Debug)]
pub struct Scene {
    pub dim: u8,
    pub env: Env,
    pub prims: Vec<Arc<Prim>>,
    pub regions: Vec<RegionX>,
    pub pins: Vec<LabelPin>,
    /// union of the non-auxiliary primitives
    pub bounds: Aabb,
    pub warnings: Vec<String>,
    pub naming: NameScheme,
    /// (feature, role, tag) added by the operations (Cut: `symmetry`)
    pub auto_tags: Vec<(FeatureId, String, String)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub prims_built: usize,
    pub prims_cached: usize,
    pub pairs_computed: usize,
    pub pairs_cached: usize,
}

/// Evaluation cache shared across evaluations of related feature lists (parameter sweeps, GUI edits).
#[derive(Default)]
pub struct Cache {
    pub(crate) prims: HashMap<u64, Arc<Prim>>,
    pub(crate) pairs: HashMap<(u64, u64), Arc<crate::arrange::PairRes>>,
    pub stats: Stats,
}

impl Cache {
    pub fn new() -> Cache {
        Cache::default()
    }
    pub fn clear(&mut self) {
        self.prims.clear();
        self.pairs.clear();
    }
}

impl Scene {
    /// Region index of the point `p` (painter rule: the last region that contains it wins; holes and voids give `None`).
    pub fn classify(&self, p: V3) -> Option<usize> {
        let bits: Vec<bool> = self.prims.iter().map(|q| q.sdf3(p, 0.0) < 0.0).collect();
        self.region_of(&|i| bits[i])
    }
    pub fn region_of(&self, bit: &dyn Fn(usize) -> bool) -> Option<usize> {
        for (i, r) in self.regions.iter().enumerate().rev() {
            if r.expr.eval(bit) {
                return if matches!(r.kind, RegionKind::Hole | RegionKind::Void) { None } else { Some(i) };
            }
        }
        None
    }
    /// Smallest distance from `p` to any primitive boundary, capped at `margin`.
    pub fn sdf_bound(&self, p: V3, margin: f64) -> f64 {
        self.prims.iter().filter(|q| !q.aux).map(|q| q.sdf3(p, margin).abs()).fold(margin, f64::min)
    }
}

struct Builder<'a> {
    list: &'a FeatureList,
    env: &'a Env,
    cache: &'a mut Cache,
    prims: Vec<Arc<Prim>>,
    memo: HashMap<(FeatureId, [u64; 8], String), usize>,
    index: HashMap<FeatureId, usize>,
    depth: usize,
    /// z interval and naming style while instantiating the profile of an `Extrude`
    extrude: Option<(f64, f64)>,
}

const MAX_INSTANCES: f64 = 100_000.0;

impl Builder<'_> {
    fn ev(&self, e: &crate::expr::Expr, feat: &Feature, what: &str) -> GeomResult<f64> {
        self.env.eval(e, &format!("{what} of feature '{}'", feat.name))
    }
    fn ev2(&self, e: &[crate::expr::Expr; 2], f: &Feature, what: &str) -> GeomResult<V2> {
        Ok([self.ev(&e[0], f, what)?, self.ev(&e[1], f, what)?])
    }
    fn ev3(&self, e: &[crate::expr::Expr; 3], f: &Feature, what: &str) -> GeomResult<V3> {
        Ok([self.ev(&e[0], f, what)?, self.ev(&e[1], f, what)?, self.ev(&e[2], f, what)?])
    }
    fn dim(&self) -> u8 {
        self.list.dim
    }

    fn step_xf(&self, s: &TStep, f: &Feature) -> GeomResult<Xf3> {
        let three = self.dim() == 3;
        let xy = |x: Xf2| Xf3 { xy: x, zs: 1.0, zt: 0.0 };
        Ok(match s {
            TStep::Translate(v) => {
                let v = self.ev3(v, f, "translation")?;
                if v[2] != 0.0 && !three {
                    return Err(GeomError(format!("Transform '{}' moves a 2D profile along z", f.name)));
                }
                Xf3 { xy: Xf2::translate([v[0], v[1]]), zs: 1.0, zt: v[2] }
            }
            TStep::Rotate { axis, deg, about } => {
                if *axis != Axis::Z {
                    return Err(GeomError(format!(
                        "Transform '{}' rotates about the {} axis; only rotations about z are supported (prismatic solids stay prismatic)",
                        f.name,
                        axis.name()
                    )));
                }
                let (a, c) = (self.ev(deg, f, "rotation angle")?.to_radians(), self.ev3(about, f, "rotation centre")?);
                let c = [c[0], c[1]];
                xy(Xf2::translate(c).after(&Xf2::rotate(a)).after(&Xf2::translate(mul2(c, -1.0))))
            }
            TStep::Mirror { axis, at } => {
                let at = self.ev(at, f, "mirror position")?;
                match axis {
                    Axis::Z if three => Xf3 { xy: Xf2::ID, zs: -1.0, zt: 2.0 * at },
                    Axis::Z => {
                        return Err(GeomError(format!("Transform '{}' mirrors a 2D profile across a z plane", f.name)))
                    }
                    a => xy(Xf2::mirror(a.index(), at)),
                }
            }
            TStep::Scale { factor, about } => {
                let (s, c) = (self.ev(factor, f, "scale factor")?, self.ev3(about, f, "scale centre")?);
                if s <= 0.0 {
                    return Err(GeomError(format!("Transform '{}' has scale factor {s}; it must be positive", f.name)));
                }
                let cxy = [c[0], c[1]];
                Xf3 {
                    xy: Xf2::translate(cxy).after(&Xf2::scaling(s)).after(&Xf2::translate(mul2(cxy, -1.0))),
                    zs: s,
                    zt: c[2] * (1.0 - s),
                }
            }
        })
    }

    fn leaf(
        &mut self,
        f: &Feature,
        shape: LShape,
        xf: Xf3,
        suffix: &str,
        style: Style,
        zl: (f64, f64),
    ) -> GeomResult<Node> {
        let key = (f.id, xf.key(), suffix.to_string());
        if let Some(&i) = self.memo.get(&key) {
            return Ok(Node::Prim(i));
        }
        let name = format!("{}{}", f.name, suffix);
        let z = if self.dim() == 3 { xf.interval(zl) } else { (0.0, 0.0) };
        let prim = Prim::new_styled(f.id, name, shape, xf.xy, false, style, z)?;
        let prim = match self.cache.prims.get(&prim.key) {
            Some(p) => {
                self.cache.stats.prims_cached += 1;
                p.clone()
            }
            None => {
                self.cache.stats.prims_built += 1;
                let p = Arc::new(prim);
                self.cache.prims.insert(p.key, p.clone());
                p
            }
        };
        self.prims.push(prim);
        self.memo.insert(key, self.prims.len() - 1);
        Ok(Node::Prim(self.prims.len() - 1))
    }

    fn inst(&mut self, id: FeatureId, xf: Xf3, suffix: &str) -> GeomResult<Node> {
        let Some(&fi) = self.index.get(&id) else { return Err(GeomError(format!("unknown feature id {id}"))) };
        let list = self.list;
        let f = &list.features[fi];
        if f.suppressed {
            return Ok(Node::Empty);
        }
        self.depth += 1;
        if self.depth > 256 {
            return Err(GeomError(format!("feature '{}' is nested more than 256 levels deep", f.name)));
        }
        let r = self.inst_op(f, xf, suffix);
        self.depth -= 1;
        r
    }

    /// A 2D profile leaf: plain in a 2D model, extruded (with a z interval) in a 3D model.
    fn profile(&mut self, f: &Feature, shape: LShape, xf: Xf3, suffix: &str) -> GeomResult<Node> {
        match (self.dim(), self.extrude) {
            (2, _) => self.leaf(f, shape, xf, suffix, Style::Plain, (0.0, 0.0)),
            (_, Some(z)) => self.leaf(f, shape, xf, suffix, Style::Extrude, z),
            _ => Err(GeomError(format!(
                "{} '{}' is a 2D profile: in a 3D model it must be the input of an Extrude",
                f.op.kind(),
                f.name
            ))),
        }
    }

    fn inst_op(&mut self, f: &Feature, xf: Xf3, suffix: &str) -> GeomResult<Node> {
        let pos = |what: &str, v: f64| -> GeomResult<f64> {
            if v > 0.0 {
                Ok(v)
            } else {
                Err(GeomError(format!("{} '{}' has {what} {v}; it must be positive", f.op.kind(), f.name)))
            }
        };
        let solid_only = |this: &Self| -> GeomResult<()> {
            if this.dim() == 3 {
                Ok(())
            } else {
                Err(GeomError(format!("{} '{}' is a 3D solid; this is a 2D model", f.op.kind(), f.name)))
            }
        };
        match &f.op {
            Op::Rect { min, max } | Op::RoundedRect { min, max, .. } => {
                let (a, b) = (self.ev2(min, f, "min corner")?, self.ev2(max, f, "max corner")?);
                let (lo, hi) = ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])]);
                pos("width", hi[0] - lo[0])?;
                pos("height", hi[1] - lo[1])?;
                let shape = if let Op::RoundedRect { r, .. } = &f.op {
                    let r = self.ev(r, f, "corner radius")?;
                    if r < 0.0 || r > 0.5 * (hi[0] - lo[0]).min(hi[1] - lo[1]) {
                        return Err(GeomError(format!(
                            "RoundedRect '{}' has corner radius {r}; it must lie between 0 and half the smaller side",
                            f.name
                        )));
                    }
                    if r == 0.0 {
                        LShape::Rect { lo, hi }
                    } else {
                        LShape::Rounded { lo, hi, r }
                    }
                } else {
                    LShape::Rect { lo, hi }
                };
                self.profile(f, shape, xf, suffix)
            }
            Op::Circle { center, r } => {
                let (c, r) = (self.ev2(center, f, "centre")?, self.ev(r, f, "radius")?);
                pos("radius", r)?;
                self.profile(f, LShape::Circle { c, r }, xf, suffix)
            }
            Op::Ellipse { center, radii, angle_deg } => {
                let (c, ab, ang) =
                    (self.ev2(center, f, "centre")?, self.ev2(radii, f, "radii")?, self.ev(angle_deg, f, "angle")?);
                pos("semi-axis", ab[0])?;
                pos("semi-axis", ab[1])?;
                let rot = Xf2::translate(c).after(&Xf2::rotate(ang.to_radians()));
                self.profile(
                    f,
                    LShape::Ellipse { c: [0.0, 0.0], a: ab[0], b: ab[1] },
                    Xf3 { xy: xf.xy.after(&rot), ..xf },
                    suffix,
                )
            }
            Op::Polygon { pts, names } => {
                if pts.len() < 3 {
                    return Err(GeomError(format!("Polygon '{}' needs at least 3 points", f.name)));
                }
                if !names.is_empty() && names.len() != pts.len() {
                    return Err(GeomError(format!(
                        "Polygon '{}' has {} points but {} edge names",
                        f.name,
                        pts.len(),
                        names.len()
                    )));
                }
                let p: Vec<V2> = pts.iter().map(|q| self.ev2(q, f, "point")).collect::<GeomResult<_>>()?;
                let nm: Vec<String> =
                    if names.is_empty() { (0..p.len()).map(|i| format!("e{i}")).collect() } else { names.clone() };
                if (0..nm.len()).any(|i| nm[..i].contains(&nm[i])) {
                    return Err(GeomError(format!("Polygon '{}' has repeated edge names", f.name)));
                }
                if polygon_area(&p).abs()
                    <= 1e-12 * p.iter().map(|q| q[0].abs().max(q[1].abs())).fold(0.0, f64::max).powi(2)
                {
                    return Err(GeomError(format!("Polygon '{}' has zero area", f.name)));
                }
                let (pts, names) = ccw_polygon(&p, &nm);
                self.profile(f, LShape::Poly { pts, names }, xf, suffix)
            }
            Op::Box { min, max } => {
                solid_only(self)?;
                let (a, b) = (self.ev3(min, f, "min corner")?, self.ev3(max, f, "max corner")?);
                let (lo, hi) = ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])]);
                pos("x extent", hi[0] - lo[0])?;
                pos("y extent", hi[1] - lo[1])?;
                pos("z extent", (a[2] - b[2]).abs())?;
                self.leaf(f, LShape::Rect { lo, hi }, xf, suffix, Style::Box, (a[2].min(b[2]), a[2].max(b[2])))
            }
            Op::Cylinder { base, axis, r, h } => {
                solid_only(self)?;
                if *axis != Axis::Z {
                    return Err(GeomError(format!(
                        "Cylinder '{}' has axis {}; this build supports cylinders along z (the stack direction) only",
                        f.name,
                        axis.name()
                    )));
                }
                let (c, r, h) = (self.ev3(base, f, "base")?, self.ev(r, f, "radius")?, self.ev(h, f, "height")?);
                pos("radius", r)?;
                pos("height", h.abs())?;
                self.leaf(
                    f,
                    LShape::Circle { c: [c[0], c[1]], r },
                    xf,
                    suffix,
                    Style::Cylinder,
                    (c[2].min(c[2] + h), c[2].max(c[2] + h)),
                )
            }
            Op::Extrude { profile, z0, z1 } => {
                solid_only(self)?;
                let (a, b) = (self.ev(z0, f, "z0")?, self.ev(z1, f, "z1")?);
                pos("height", (b - a).abs())?;
                let saved = self.extrude.replace((a.min(b), a.max(b)));
                let r = self.inst(*profile, xf, suffix);
                self.extrude = saved;
                r
            }
            Op::Transform { input, steps } => {
                let mut t = Xf3::ID;
                for s in steps {
                    t = self.step_xf(s, f)?.after(&t);
                }
                self.inst(*input, xf.after(&t), suffix)
            }
            Op::Pattern { input, kind, n } => {
                let nn = self.ev(n, f, "count")?;
                if nn.fract() != 0.0 || !(1.0..=MAX_INSTANCES).contains(&nn) {
                    return Err(GeomError(format!(
                        "Pattern '{}' count {nn} must be a whole number between 1 and {MAX_INSTANCES}",
                        f.name
                    )));
                }
                let nn = nn as usize;
                let mut out = Vec::with_capacity(nn);
                for i in 0..nn {
                    let t = match kind {
                        PatternKind::Linear { step } => {
                            let s = self.ev3(step, f, "step")?;
                            Xf3 { xy: Xf2::translate([s[0] * i as f64, s[1] * i as f64]), zs: 1.0, zt: s[2] * i as f64 }
                        }
                        PatternKind::Polar { axis, center, total_deg } => {
                            if *axis != Axis::Z {
                                return Err(GeomError(format!(
                                    "Pattern '{}' rotates about {}; use z",
                                    f.name,
                                    axis.name()
                                )));
                            }
                            let (c, tot) = (self.ev3(center, f, "centre")?, self.ev(total_deg, f, "angle")?);
                            let c = [c[0], c[1]];
                            Xf3 {
                                xy: Xf2::translate(c)
                                    .after(&Xf2::rotate((tot * i as f64 / nn as f64).to_radians()))
                                    .after(&Xf2::translate(mul2(c, -1.0))),
                                zs: 1.0,
                                zt: 0.0,
                            }
                        }
                    };
                    out.push(self.inst(*input, xf.after(&t), &format!("{suffix}[{i}]"))?);
                }
                Ok(Node::Union(out))
            }
            Op::Union { ids } => {
                let v = ids.iter().map(|i| self.inst(*i, xf, suffix)).collect::<GeomResult<Vec<_>>>()?;
                Ok(Node::Union(v.into_iter().filter(|n| *n != Node::Empty).collect()))
            }
            Op::Intersect { ids } => {
                let v = ids.iter().map(|i| self.inst(*i, xf, suffix)).collect::<GeomResult<Vec<_>>>()?;
                let v: Vec<Node> = v.into_iter().filter(|n| *n != Node::Empty).collect();
                Ok(if v.is_empty() { Node::Empty } else { Node::Inter(v) })
            }
            Op::Difference { a, b } => {
                let a = self.inst(*a, xf, suffix)?;
                if a == Node::Empty {
                    return Ok(a);
                }
                let b = b.iter().map(|i| self.inst(*i, xf, suffix)).collect::<GeomResult<Vec<_>>>()?;
                Ok(Node::Diff(Box::new(a), b.into_iter().filter(|n| *n != Node::Empty).collect()))
            }
            op => Err(GeomError(format!("feature '{}' ({}) cannot be used as a shape", f.name, op.kind()))),
        }
    }
}

/// Builds the scene of a feature list (validates it first).
pub fn build_scene(list: &FeatureList, cache: &mut Cache) -> GeomResult<Scene> {
    list.validate()?;
    let dim = list.dim as usize;
    let mut env = Env::default();
    for p in &list.params {
        let v = env.eval(&p.value, &format!("parameter '{}'", p.name))?;
        env.vars.push((p.name.clone(), v));
    }
    let overlay = list
        .features
        .iter()
        .find(|f| matches!(f.op, Op::Overlay { .. }) && !f.suppressed)
        .ok_or_else(|| GeomError("the model has no Overlay feature; add one that lists the regions".into()))?;
    let Op::Overlay { regions, pins, naming } = &overlay.op else { unreachable!() };
    let index = list.features.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
    let mut b = Builder { list, env: &env, cache, prims: vec![], memo: HashMap::new(), index, depth: 0, extrude: None };
    let mut rx: Vec<RegionX> = vec![];
    let mut warnings = vec![];
    for r in regions {
        let expr = b.inst(r.input, Xf3::ID, "")?;
        if expr == Node::Empty {
            warnings.push(format!("region '{}' is built from a suppressed or empty feature", r.name));
        }
        rx.push(RegionX {
            name: r.name.clone(),
            kind: r.kind,
            feature: r.input,
            pinned_id: r.pinned_id,
            expr,
            parent: None,
            pml: None,
        });
    }
    let model_box = |prims: &[Arc<Prim>]| prims.iter().filter(|p| !p.aux).fold(Aabb::EMPTY, |a, p| a.union(&p.bbox3()));
    let diag = model_box(&b.prims).diagonal();
    let mut auto_tags = vec![];
    let after = list.features.iter().position(|f| f.id == overlay.id).unwrap_or(0);
    let big_z = 4.0 * diag + 1.0;
    // Cut: intersect every region with a half-space solid whose face `cut` lies on the plane (axis-aligned planes in 3D)
    for f in list.features[after..].iter().filter(|f| !f.suppressed) {
        let Op::Cut { normal, offset, keep } = &f.op else { continue };
        let n = b.ev3(normal, f, "normal")?;
        let off = b.ev(offset, f, "offset")?;
        let nl = norm3(n);
        if nl < 1e-12 {
            return Err(GeomError(format!("Cut '{}' needs a nonzero normal", f.name)));
        }
        let (n, off) = (mul3(n, 1.0 / nl), off / nl);
        let s = if *keep == Keep::Above { 1.0 } else { -1.0 };
        let (prim, zcut) = if n[2].abs() > 1e-12 {
            if n[0].abs() > 1e-12 || n[1].abs() > 1e-12 || dim == 2 {
                return Err(GeomError(format!(
                    "Cut '{}' is a slanted plane; this build cuts only along axes (a slanted cut needs the general 3D arrangement)",
                    f.name
                )));
            }
            // z plane: a slab above or below `z = off / n_z`
            let zc = off / n[2];
            let keep_up = s * n[2] > 0.0;
            let (z0, z1) = if keep_up { (zc, zc + big_z) } else { (zc - big_z, zc) };
            let lo = [scene_lo(&b.prims, 0) - big_z, scene_lo(&b.prims, 1) - big_z];
            let hi = [lo[0] + 3.0 * big_z + diag, lo[1] + 3.0 * big_z + diag];
            let mut p =
                Prim::new_styled(f.id, f.name.clone(), LShape::Rect { lo, hi }, Xf2::ID, true, Style::Box, (z0, z1))?;
            p.caps = if keep_up {
                ["cut".to_string(), "top".to_string()]
            } else {
                ["bottom".to_string(), "cut".to_string()]
            };
            (p, true)
        } else {
            let (n2, off2) = ([n[0], n[1]], off);
            let (q, d) = (mul2(n2, off2), perp(n2));
            let big = 4.0 * diag + 4.0 * off2.abs() + 1.0;
            let sv = mul2(n2, s);
            let (p0, p1) = (sub2(q, mul2(d, big)), add2(q, mul2(d, big)));
            let pts = vec![p0, p1, add2(p1, mul2(sv, big)), add2(p0, mul2(sv, big))];
            let names = vec!["cut".to_string(), "far1".into(), "far2".into(), "far3".into()];
            let (pts, names) = ccw_polygon(&pts, &names);
            let z = if dim == 3 {
                (-big_z + scene_lo(&b.prims, 2), big_z + scene_lo(&b.prims, 2) + diag)
            } else {
                (0.0, 0.0)
            };
            let style = if dim == 3 { Style::Extrude } else { Style::Plain };
            let mut p = Prim::new_styled(f.id, f.name.clone(), LShape::Poly { pts, names }, Xf2::ID, true, style, z)?;
            for sg in &mut p.segs {
                if sg.role == "side:cut" {
                    sg.role = "cut".to_string();
                }
            }
            (p, false)
        };
        let _ = zcut;
        let role = "cut";
        b.prims.push(Arc::new(prim));
        let h = Node::Prim(b.prims.len() - 1);
        for r in &mut rx {
            if r.expr != Node::Empty {
                r.expr = Node::Inter(vec![std::mem::replace(&mut r.expr, Node::Empty), h.clone()]);
            }
        }
        auto_tags.push((f.id, role.to_string(), "symmetry".to_string()));
    }
    // Pml: split matching regions into the physical part and one piece per side combination
    for f in list.features[after..].iter().filter(|f| !f.suppressed) {
        let Op::Pml { window, sides, thickness, layers, grading, regions: only } = &f.op else { continue };
        let (t, nl, gr) = (b.ev(thickness, f, "thickness")?, b.ev(layers, f, "layers")?, b.ev(grading, f, "grading")?);
        if t <= 0.0 || nl < 1.0 || gr <= 0.0 {
            return Err(GeomError(format!("Pml '{}' needs thickness > 0, layers >= 1 and grading > 0", f.name)));
        }
        let mark = b.prims.len();
        let wnode = b.inst(*window, Xf3::ID, "")?;
        let win = wnode.bbox(&b.prims);
        b.prims.truncate(mark);
        b.memo.retain(|_, v| *v < mark);
        if win.is_empty() {
            return Err(GeomError(format!("Pml '{}' has an empty window", f.name)));
        }
        let all: Vec<Side> = [Side::XMin, Side::XMax, Side::YMin, Side::YMax, Side::ZMin, Side::ZMax]
            .into_iter()
            .take(2 * dim)
            .collect();
        let sel: Vec<Side> = if sides.is_empty() { all.clone() } else { sides.clone() };
        if sel.iter().any(|s| s.axis() >= dim) {
            return Err(GeomError(format!("Pml '{}' lists a z side in a 2D model", f.name)));
        }
        // side combinations: fewer sides first (faces, then edges, then corners); deterministic order
        let mut combos: Vec<[i8; 3]> = vec![];
        let total = 3usize.pow(dim as u32);
        for code in 0..total {
            let mut c = [0i8; 3];
            let mut k = code;
            for ci in c.iter_mut().take(dim) {
                *ci = (k % 3) as i8 - 1;
                k /= 3;
            }
            let ok = c != [0, 0, 0]
                && (0..dim).all(|a| c[a] == 0 || sel.contains(&if c[a] < 0 { all[2 * a] } else { all[2 * a + 1] }));
            if ok {
                combos.push(c);
            }
        }
        combos.sort_by_key(|c| {
            let n = c.iter().filter(|&&x| x != 0).count();
            let single = (0..dim).find(|&a| c[a] != 0).map_or(0, |a| 2 * a + (c[a] > 0) as usize);
            (n, if n == 1 { single } else { 0 }, (0..dim).rev().map(|a| c[a] + 1).collect::<Vec<_>>())
        });
        let mut pieces: Vec<(Node, [i8; 3], String)> = vec![];
        for c in &combos {
            let rng = |k: usize| match c[k] {
                -1 => (win.lo[k] - t, win.lo[k]),
                0 => (win.lo[k], win.hi[k]),
                _ => (win.hi[k], win.hi[k] + t),
            };
            let piece = (0..dim)
                .filter(|&a| c[a] != 0)
                .map(|a| format!("{}{}", ["x", "y", "z"][a], if c[a] < 0 { "min" } else { "max" }))
                .collect::<Vec<_>>()
                .join("_");
            let (x, y) = (rng(0), rng(1));
            let (style, z) = if dim == 3 { (Style::Box, rng(2)) } else { (Style::Plain, (0.0, 0.0)) };
            let prim = Prim::new_styled(
                f.id,
                format!("{}[{piece}]", f.name),
                LShape::Rect { lo: [x.0, y.0], hi: [x.1, y.1] },
                Xf2::ID,
                true,
                style,
                z,
            )?;
            b.prims.push(Arc::new(prim));
            pieces.push((Node::Prim(b.prims.len() - 1), *c, piece));
        }
        let mut out: Vec<RegionX> = vec![];
        let pad_bb: Vec<Aabb> = pieces.iter().map(|(n, _, _)| n.bbox(&b.prims)).collect();
        for r in rx.iter() {
            let candidate = r.pml.is_none()
                && r.parent.is_none()
                && matches!(r.kind, RegionKind::Material)
                && (only.is_empty() || only.contains(&r.name));
            if !candidate || r.expr == Node::Empty {
                out.push(r.clone());
                continue;
            }
            let eb = r.expr.bbox(&b.prims);
            let all_pieces: Vec<Node> = pieces.iter().map(|p| p.0.clone()).collect();
            let mut phys = r.clone();
            phys.expr = Node::Diff(Box::new(r.expr.clone()), all_pieces);
            out.push(phys);
            for ((node, c, piece), bb) in pieces.iter().zip(&pad_bb) {
                if !eb.overlaps(bb) {
                    continue;
                }
                out.push(RegionX {
                    name: format!("{}.pml_{piece}", r.name),
                    kind: RegionKind::Pml,
                    feature: r.feature,
                    pinned_id: None,
                    expr: Node::Inter(vec![r.expr.clone(), node.clone()]),
                    parent: Some(r.name.clone()),
                    pml: Some(PmlInfo { feature: f.id, sides: *c, window: win, thickness: t, layers: nl, grading: gr }),
                });
            }
        }
        rx = out;
    }
    // normalise, then drop primitives that no region refers to any more
    let mut prims = b.prims;
    for r in &mut rx {
        r.expr = std::mem::replace(&mut r.expr, Node::Empty).simplify(&prims);
    }
    let mut used = vec![false; prims.len()];
    let mut leaves = vec![];
    for r in &rx {
        r.expr.leaves(&mut leaves);
    }
    for l in leaves {
        used[l] = true;
    }
    let mut map = vec![usize::MAX; prims.len()];
    let mut kept = vec![];
    for (i, p) in prims.drain(..).enumerate() {
        if used[i] {
            map[i] = kept.len();
            kept.push(p);
        }
    }
    for r in &mut rx {
        r.expr.remap(&map);
    }
    let prims = kept;
    let bounds = model_box(&prims);
    Ok(Scene {
        dim: list.dim,
        env,
        prims,
        regions: rx,
        pins: pins.clone(),
        bounds,
        warnings,
        naming: *naming,
        auto_tags,
    })
}

/// Lower corner of the non-auxiliary primitives along an axis.
fn scene_lo(prims: &[Arc<Prim>], axis: usize) -> f64 {
    let m = prims.iter().filter(|p| !p.aux).fold(Aabb::EMPTY, |a, p| a.union(&p.bbox3()));
    if m.is_empty() {
        0.0
    } else {
        m.lo[axis]
    }
}
