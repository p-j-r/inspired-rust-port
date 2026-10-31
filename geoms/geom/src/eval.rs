//! Evaluation: `Evaluator` owns the cache and turns a `FeatureList` into an `Evaluated` model at a chosen tier. Cheaper tiers
//! never build costlier data: `Bounds` needs parameters only, `Classify` builds the scene (primitives and region
//! expressions) but no arrangement, `Complex` adds the arrangement, names and selectors, `Display` adds tessellation.

use crate::arrange::arrange;
use crate::complex::Complex;
use crate::error::GeomResult;
use crate::math::*;
use crate::model::*;
use crate::scene::{build_scene, Cache, Scene, Stats};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Bounds,
    Classify,
    Complex,
    Display,
}

pub struct Evaluated {
    pub list: FeatureList,
    pub tier: Tier,
    pub scene: Option<Scene>,
    pub complex: Option<Complex>,
    pub warnings: Vec<String>,
    pub(crate) sim: crate::sim::SimData,
    pub(crate) feature_hashes: Vec<(FeatureId, u64)>,
    bounds: Aabb,
}

#[derive(Default)]
pub struct Evaluator {
    pub cache: Cache,
}

impl Evaluator {
    pub fn new() -> Evaluator {
        Evaluator::default()
    }

    pub fn stats(&self) -> Stats {
        self.cache.stats
    }

    pub fn eval(&mut self, list: &FeatureList, tier: Tier) -> GeomResult<Evaluated> {
        let scene = build_scene(list, &mut self.cache)?;
        let bounds = scene.bounds;
        let mut warnings = scene.warnings.clone();
        let mut ev = Evaluated {
            list: list.clone(),
            tier,
            scene: None,
            complex: None,
            warnings: vec![],
            sim: Default::default(),
            feature_hashes: vec![],
            bounds,
        };
        if tier >= Tier::Complex {
            let cx = arrange(&scene, &mut self.cache)?;
            for r in &cx.regions {
                if r.loops.is_empty() {
                    warnings.push(format!(
                        "region '{}' is empty: nothing of it is left after the other regions are painted over it",
                        r.name
                    ));
                }
            }
            #[cfg(debug_assertions)]
            cx.check().map_err(|m| crate::error::GeomError(format!("self-check failed: {m}")))?;
            ev.complex = Some(cx);
        }
        ev.feature_hashes = crate::diff::feature_hashes(list, &scene.env);
        ev.scene = Some(scene);
        ev.warnings = warnings;
        if tier >= Tier::Complex {
            ev.compute_sim()?;
        }
        Ok(ev)
    }
}

impl Model {
    /// Evaluates with a throw-away cache; use an `Evaluator` for sweeps.
    pub fn eval(&self, tier: Tier) -> GeomResult<Evaluated> {
        Evaluator::new().eval(&self.list, tier)
    }
}

impl Evaluated {
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }
    pub fn regions(&self) -> &[crate::complex::Region] {
        &self.complex().regions
    }
    pub fn edges(&self) -> &[crate::complex::Edge] {
        &self.complex().edges
    }
    pub fn vertices(&self) -> &[crate::complex::Vertex] {
        &self.complex().vertices
    }
    pub fn complex(&self) -> &Complex {
        self.complex.as_ref().expect("this evaluation did not reach the Complex tier")
    }
    pub fn scene(&self) -> &Scene {
        self.scene.as_ref().expect("scene")
    }
    /// Region id containing `p` (painter rule), without building the arrangement.
    pub fn classify(&self, p: V3) -> Option<u32> {
        let s = self.scene();
        let i = s.classify([p[0], p[1]])?;
        match &self.complex {
            Some(c) => c.scene_to_region[i],
            None => Some(i as u32),
        }
    }
    /// Distance from `p` to the nearest primitive boundary, at most `margin`.
    pub fn sdf_bound(&self, p: V3, margin: f64) -> f64 {
        self.scene().sdf_bound([p[0], p[1]], margin)
    }
    pub fn tol(&self) -> f64 {
        1e-9 * self.bounds.diagonal()
    }
}
