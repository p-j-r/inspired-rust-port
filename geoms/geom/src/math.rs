//! Small vector, box and similarity-transform helpers (2D and 3D, `f64`, plain arrays).

pub type V2 = [f64; 2];
pub type V3 = [f64; 3];

pub const TAU: f64 = std::f64::consts::TAU;

#[inline]
pub fn add2(a: V2, b: V2) -> V2 {
    [a[0] + b[0], a[1] + b[1]]
}
#[inline]
pub fn sub2(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}
#[inline]
pub fn mul2(a: V2, s: f64) -> V2 {
    [a[0] * s, a[1] * s]
}
#[inline]
pub fn dot2(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
#[inline]
pub fn cross2(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
#[inline]
pub fn norm2(a: V2) -> f64 {
    a[0].hypot(a[1])
}
#[inline]
pub fn dist2(a: V2, b: V2) -> f64 {
    norm2(sub2(a, b))
}
/// Rotation by +90 degrees (the left normal of a direction).
#[inline]
pub fn perp(a: V2) -> V2 {
    [-a[1], a[0]]
}

#[inline]
pub fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn mul3(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
pub fn norm3(a: V3) -> f64 {
    dot3(a, a).sqrt()
}
#[inline]
pub fn unit3(a: V3) -> V3 {
    let n = norm3(a);
    if n > 0.0 {
        mul3(a, 1.0 / n)
    } else {
        a
    }
}
#[inline]
pub fn to3(p: V2) -> V3 {
    [p[0], p[1], 0.0]
}

/// Angle of `a` relative to `base`, wrapped into `[base, base + TAU)`.
#[inline]
pub fn wrap_from(t: f64, base: f64) -> f64 {
    base + (t - base).rem_euclid(TAU)
}

/// Axis-aligned box (2D or 3D; 2D boxes keep z = 0).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Aabb {
    pub lo: V3,
    pub hi: V3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { lo: [f64::INFINITY; 3], hi: [f64::NEG_INFINITY; 3] };

    pub fn new(lo: V3, hi: V3) -> Aabb {
        Aabb { lo, hi }
    }
    pub fn is_empty(&self) -> bool {
        (0..3).any(|k| self.lo[k] > self.hi[k])
    }
    pub fn grow(&mut self, p: V3) {
        for (k, &x) in p.iter().enumerate() {
            self.lo[k] = self.lo[k].min(x);
            self.hi[k] = self.hi[k].max(x);
        }
    }
    pub fn union(&self, o: &Aabb) -> Aabb {
        let mut r = *self;
        if !o.is_empty() {
            r.grow(o.lo);
            r.grow(o.hi);
        }
        r
    }
    pub fn pad(&self, d: f64) -> Aabb {
        Aabb { lo: sub3(self.lo, [d; 3]), hi: add3(self.hi, [d; 3]) }
    }
    pub fn overlaps(&self, o: &Aabb) -> bool {
        (0..3).all(|k| self.lo[k] <= o.hi[k] && o.lo[k] <= self.hi[k])
    }
    pub fn contains(&self, p: V3) -> bool {
        (0..3).all(|k| p[k] >= self.lo[k] && p[k] <= self.hi[k])
    }
    pub fn diagonal(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            norm3(sub3(self.hi, self.lo))
        }
    }
    pub fn center(&self) -> V3 {
        mul3(add3(self.lo, self.hi), 0.5)
    }
}

/// Similarity transform of the plane: `p -> m p + t` with `m = s * R` or `s * R * mirror`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xf2 {
    pub m: [[f64; 2]; 2],
    pub t: V2,
}

impl Xf2 {
    pub const ID: Xf2 = Xf2 { m: [[1.0, 0.0], [0.0, 1.0]], t: [0.0; 2] };

    pub fn apply(&self, p: V2) -> V2 {
        [self.m[0][0] * p[0] + self.m[0][1] * p[1] + self.t[0], self.m[1][0] * p[0] + self.m[1][1] * p[1] + self.t[1]]
    }
    /// Linear part only (directions).
    pub fn apply_dir(&self, p: V2) -> V2 {
        [self.m[0][0] * p[0] + self.m[0][1] * p[1], self.m[1][0] * p[0] + self.m[1][1] * p[1]]
    }
    pub fn det(&self) -> f64 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }
    /// Uniform scale factor.
    pub fn scale(&self) -> f64 {
        self.det().abs().sqrt()
    }
    pub fn mirrored(&self) -> bool {
        self.det() < 0.0
    }
    /// `self` applied after `o`.
    pub fn after(&self, o: &Xf2) -> Xf2 {
        let (a, b) = (self.m, o.m);
        let m = [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ];
        Xf2 { m, t: self.apply(o.t) }
    }
    pub fn inverse(&self) -> Xf2 {
        let d = self.det();
        let m = [[self.m[1][1] / d, -self.m[0][1] / d], [-self.m[1][0] / d, self.m[0][0] / d]];
        let inv = Xf2 { m, t: [0.0; 2] };
        Xf2 { m, t: mul2(inv.apply(self.t), -1.0) }
    }
    pub fn translate(t: V2) -> Xf2 {
        Xf2 { m: Xf2::ID.m, t }
    }
    pub fn rotate(rad: f64) -> Xf2 {
        let (s, c) = rad.sin_cos();
        Xf2 { m: [[c, -s], [s, c]], t: [0.0; 2] }
    }
    pub fn scaling(s: f64) -> Xf2 {
        Xf2 { m: [[s, 0.0], [0.0, s]], t: [0.0; 2] }
    }
    /// Reflection across the line `p[axis] = at`.
    pub fn mirror(axis: usize, at: f64) -> Xf2 {
        let mut m = Xf2::ID.m;
        m[axis][axis] = -1.0;
        let mut t = [0.0; 2];
        t[axis] = 2.0 * at;
        Xf2 { m, t }
    }
}

/// 64-bit content hasher (splitmix-style mixing); deterministic across runs and platforms.
#[derive(Clone, Copy, Debug)]
pub struct Hasher64(pub u64);

impl Hasher64 {
    pub fn new() -> Self {
        Hasher64(0x9E37_79B9_7F4A_7C15)
    }
    pub fn u64(&mut self, v: u64) -> &mut Self {
        let mut z = (self.0 ^ v).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        self.0 = z ^ (z >> 31);
        self
    }
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.u64(if v == 0.0 { 0 } else { v.to_bits() })
    }
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        for b in s.bytes() {
            self.u64(b as u64);
        }
        self
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
}

impl Default for Hasher64 {
    fn default() -> Self {
        Self::new()
    }
}
