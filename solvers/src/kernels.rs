//! Chunk-level arithmetic kernels with run-time ISA dispatch.
//!
//! Every kernel body is written once (in [`define_kernels!`]) and instantiated three times:
//! a portable baseline, an AVX2 version and an AVX-512 version, the latter two compiled with
//! `#[target_feature]` and selected at run time. The *arithmetic is identical* in all
//! versions (same operations in the same order; FMA is never used), so results are bit-for-bit
//! the same on any CPU. Only the width of the vector instructions the compiler can use differs.
//!
//! Set `RSOLVERS_ISA=generic|avx2|avx512` to force a version (benchmarking, debugging).

use std::sync::OnceLock;

/// Independent partial sums in a dot product. 32 doubles hide the FP-add latency with
/// 4 AVX-512 / 8 AVX2 accumulators and keep the summation order ISA-independent.
const LANES: usize = 32;

macro_rules! define_kernels {
    ($(#[$attr:meta])*) => {
        /// `aᵀb` over one chunk.
        $(#[$attr])*
        pub unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
            let mut acc = [0.0f64; LANES];
            let mut ca = a.chunks_exact(LANES);
            let mut cb = b.chunks_exact(LANES);
            for (x, y) in (&mut ca).zip(&mut cb) {
                for i in 0..LANES {
                    acc[i] += x[i] * y[i];
                }
            }
            let mut tail = 0.0;
            for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
                tail += x * y;
            }
            let mut w = LANES;
            while w > 1 {
                w /= 2;
                for i in 0..w {
                    acc[i] += acc[i + w];
                }
            }
            acc[0] + tail
        }

        /// `y += α x`.
        $(#[$attr])*
        pub unsafe fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi += alpha * *xi;
            }
        }

        /// `y = x + β y`.
        $(#[$attr])*
        pub unsafe fn aypx(beta: f64, x: &[f64], y: &mut [f64]) {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = *xi + beta * *yi;
            }
        }

        /// `x += α p; r -= α w`.
        $(#[$attr])*
        pub unsafe fn axpy2(alpha: f64, p: &[f64], x: &mut [f64], w: &[f64], r: &mut [f64]) {
            for i in 0..x.len() {
                x[i] += alpha * p[i];
                r[i] -= alpha * w[i];
            }
        }

        /// Fused CG update of one chunk: `x += α p; r -= α w; z = d ⊙ r` (`d = None`: `z`
        /// aliases `r` and is untouched). Returns `(zᵀr, zᵀz, rᵀr)`.
        $(#[$attr])*
        pub unsafe fn cg_update(
            alpha: f64, p: &[f64], x: &mut [f64], w: &[f64], r: &mut [f64],
            z: &mut [f64], d: Option<&[f64]>,
        ) -> [f64; 3] {
            for i in 0..x.len() {
                x[i] += alpha * p[i];
                r[i] -= alpha * w[i];
            }
            match d {
                None => {
                    // SAFETY: same target features as the caller.
                    let rr = unsafe { dot(r, r) };
                    [rr, rr, rr]
                }
                Some(d) => {
                    for i in 0..r.len() {
                        z[i] = d[i] * r[i];
                    }
                    // SAFETY: same target features as the caller.
                    unsafe { [dot(z, r), dot(z, z), dot(r, r)] }
                }
            }
        }

        /// `wc -= Σ_j h[j] V[s.., j]` over `k` columns (a chunk of `wc.len()` rows starting at `s`).
        $(#[$attr])*
        pub unsafe fn sub_cols(data: &[f64], n: usize, s: usize, k: usize, h: &[f64], wc: &mut [f64]) {
            let len = wc.len();
            // Four columns at a time: `wc` is read and written once per group.
            let mut j = 0;
            while j + 4 <= k {
                let c0 = &data[j * n + s..j * n + s + len];
                let c1 = &data[(j + 1) * n + s..(j + 1) * n + s + len];
                let c2 = &data[(j + 2) * n + s..(j + 2) * n + s + len];
                let c3 = &data[(j + 3) * n + s..(j + 3) * n + s + len];
                let (h0, h1, h2, h3) = (h[j], h[j + 1], h[j + 2], h[j + 3]);
                for i in 0..len {
                    wc[i] -= h0 * c0[i] + h1 * c1[i] + h2 * c2[i] + h3 * c3[i];
                }
                j += 4;
            }
            while j < k {
                let c = &data[j * n + s..j * n + s + len];
                let hj = h[j];
                for i in 0..len {
                    wc[i] -= hj * c[i];
                }
                j += 1;
            }
        }

        /// `yc += Σ_j c[j] V[s.., j]` over `k` columns.
        $(#[$attr])*
        pub unsafe fn add_cols(data: &[f64], n: usize, s: usize, k: usize, c: &[f64], yc: &mut [f64]) {
            let len = yc.len();
            let mut j = 0;
            while j + 4 <= k {
                let c0 = &data[j * n + s..j * n + s + len];
                let c1 = &data[(j + 1) * n + s..(j + 1) * n + s + len];
                let c2 = &data[(j + 2) * n + s..(j + 2) * n + s + len];
                let c3 = &data[(j + 3) * n + s..(j + 3) * n + s + len];
                let (a0, a1, a2, a3) = (c[j], c[j + 1], c[j + 2], c[j + 3]);
                for i in 0..len {
                    yc[i] += a0 * c0[i] + a1 * c1[i] + a2 * c2[i] + a3 * c3[i];
                }
                j += 4;
            }
            while j < k {
                let cc = &data[j * n + s..j * n + s + len];
                let cj = c[j];
                for i in 0..len {
                    yc[i] += cj * cc[i];
                }
                j += 1;
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! dots_cols_simd {
    ($(#[$attr:meta])* $name:ident, $vt:ty, $w:expr, $zero:ident, $load:ident, $mul:ident, $add:ident, $store:ident) => {
        /// `part[j] = V[s..e, j]ᵀ w` (`w` holds rows `s..e` only) for `j < part.len()`; `data` holds
        /// columns of length `n`. Four columns stream concurrently so `w` is loaded once per
        /// four columns. Column `j` keeps eight partial sums (lane `l` takes rows `≡ l mod 8`),
        /// combined by a fixed pairwise tree, then the scalar tail is added.
        $(#[$attr])*
        pub unsafe fn $name(data: &[f64], n: usize, s: usize, e: usize, w: &[f64], part: &mut [f64]) {
            unsafe {
                const NV: usize = 8 / $w;
                let len = e - s;
                let body = len - len % 8;
                let k = part.len();
                debug_assert!(w.len() == len && data.len() >= k * n);
                let wp = w.as_ptr();
                macro_rules! reduce {
                    ($acc:expr) => {{
                        let mut lanes = [0.0f64; 8];
                        for v in 0..NV {
                            $store(lanes.as_mut_ptr().add(v * $w), $acc[v]);
                        }
                        ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
                    }};
                }
                let mut j = 0;
                while j + 4 <= k {
                    let p0 = data.as_ptr().add(j * n + s);
                    let p1 = data.as_ptr().add((j + 1) * n + s);
                    let p2 = data.as_ptr().add((j + 2) * n + s);
                    let p3 = data.as_ptr().add((j + 3) * n + s);
                    let z: $vt = $zero();
                    let (mut a0, mut a1, mut a2, mut a3) = ([z; NV], [z; NV], [z; NV], [z; NV]);
                    let mut i = 0;
                    while i < body {
                        for v in 0..NV {
                            let o = i + v * $w;
                            let wl = $load(wp.add(o));
                            a0[v] = $add(a0[v], $mul($load(p0.add(o)), wl));
                            a1[v] = $add(a1[v], $mul($load(p1.add(o)), wl));
                            a2[v] = $add(a2[v], $mul($load(p2.add(o)), wl));
                            a3[v] = $add(a3[v], $mul($load(p3.add(o)), wl));
                        }
                        i += 8;
                    }
                    let (mut t0, mut t1, mut t2, mut t3) = (0.0, 0.0, 0.0, 0.0);
                    for i in body..len {
                        let wi = *wp.add(i);
                        t0 += *p0.add(i) * wi;
                        t1 += *p1.add(i) * wi;
                        t2 += *p2.add(i) * wi;
                        t3 += *p3.add(i) * wi;
                    }
                    part[j] = reduce!(a0) + t0;
                    part[j + 1] = reduce!(a1) + t1;
                    part[j + 2] = reduce!(a2) + t2;
                    part[j + 3] = reduce!(a3) + t3;
                    j += 4;
                }
                while j < k {
                    let p = data.as_ptr().add(j * n + s);
                    let mut a = [$zero(); NV];
                    let mut i = 0;
                    while i < body {
                        for v in 0..NV {
                            let o = i + v * $w;
                            a[v] = $add(a[v], $mul($load(p.add(o)), $load(wp.add(o))));
                        }
                        i += 8;
                    }
                    let mut t = 0.0;
                    for i in body..len {
                        t += *p.add(i) * *wp.add(i);
                    }
                    part[j] = reduce!(a) + t;
                    j += 1;
                }
            }
        }
    };
}

/// Scalar reference for [`dots_cols`] (non-x86 targets). Written with explicit SIMD per tier on
/// x86_64 because LLVM does not vectorise the eight-lane accumulators reliably (it spills them
/// to the stack on some target-cpu settings); the summation order here is the contract that
/// every tier reproduces bit for bit.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
/// `part[j] = V[s..e, j]ᵀ w` (`w` holds rows `s..e` only) for `j < part.len()`; `data` holds columns of length `n`.
///
/// Four columns are streamed concurrently so `w` is loaded once per four columns and the
/// memory system sees four long sequential streams instead of one short one per column.
/// Each column keeps 8 partial sums combined by a fixed pairwise tree.
pub(crate) unsafe fn dots_cols_portable(data: &[f64], n: usize, s: usize, e: usize, w: &[f64], part: &mut [f64]) {
    const L: usize = 8;
    let len = e - s;
    let wv = w;
    let body = len - len % L;
    let k = part.len();
    let tree = |a: [f64; L]| -> f64 {
        ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
    };
    let mut j = 0;
    while j + 4 <= k {
        let c0 = &data[j * n + s..j * n + e];
        let c1 = &data[(j + 1) * n + s..(j + 1) * n + e];
        let c2 = &data[(j + 2) * n + s..(j + 2) * n + e];
        let c3 = &data[(j + 3) * n + s..(j + 3) * n + e];
        let (mut a0, mut a1, mut a2, mut a3) = ([0.0f64; L], [0.0f64; L], [0.0f64; L], [0.0f64; L]);
        // Fixed-size chunk references carry no bounds checks, which keeps the eight
        // accumulators in vector registers on every target-cpu.
        let it = wv[..body].chunks_exact(L)
            .zip(c0[..body].chunks_exact(L))
            .zip(c1[..body].chunks_exact(L))
            .zip(c2[..body].chunks_exact(L))
            .zip(c3[..body].chunks_exact(L));
        for ((((w8, x0), x1), x2), x3) in it {
            let w8: &[f64; L] = w8.try_into().unwrap();
            let x0: &[f64; L] = x0.try_into().unwrap();
            let x1: &[f64; L] = x1.try_into().unwrap();
            let x2: &[f64; L] = x2.try_into().unwrap();
            let x3: &[f64; L] = x3.try_into().unwrap();
            for l in 0..L {
                a0[l] += x0[l] * w8[l];
                a1[l] += x1[l] * w8[l];
                a2[l] += x2[l] * w8[l];
                a3[l] += x3[l] * w8[l];
            }
        }
        let (mut t0, mut t1, mut t2, mut t3) = (0.0, 0.0, 0.0, 0.0);
        for i in body..len {
            t0 += c0[i] * wv[i];
            t1 += c1[i] * wv[i];
            t2 += c2[i] * wv[i];
            t3 += c3[i] * wv[i];
        }
        part[j] = tree(a0) + t0;
        part[j + 1] = tree(a1) + t1;
        part[j + 2] = tree(a2) + t2;
        part[j + 3] = tree(a3) + t3;
        j += 4;
    }
    while j < k {
        let c = &data[j * n + s..j * n + e];
        let mut a = [0.0f64; L];
        let mut i = 0;
        while i < body {
            for l in 0..L {
                a[l] += c[i + l] * wv[i + l];
            }
            i += L;
        }
        let mut t = 0.0;
        for i in body..len {
            t += c[i] * wv[i];
        }
        part[j] = tree(a) + t;
        j += 1;
    }
}


mod generic {
    use super::LANES;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    define_kernels!();

    #[cfg(not(target_arch = "x86_64"))]
    pub use super::dots_cols_portable as dots_cols;
    #[cfg(target_arch = "x86_64")]
    dots_cols_simd!(dots_cols, __m128d, 2, _mm_setzero_pd, _mm_loadu_pd, _mm_mul_pd, _mm_add_pd, _mm_storeu_pd);
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::LANES;
    use std::arch::x86_64::*;
    define_kernels!(#[target_feature(enable = "avx2")]);
    dots_cols_simd!(#[target_feature(enable = "avx2")] dots_cols, __m256d, 4, _mm256_setzero_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_add_pd, _mm256_storeu_pd);
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::LANES;
    use std::arch::x86_64::*;
    define_kernels!(#[target_feature(enable = "avx512f")]);
    dots_cols_simd!(#[target_feature(enable = "avx512f")] dots_cols, __m512d, 8, _mm512_setzero_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_add_pd, _mm512_storeu_pd);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Generic,
    Avx2,
    Avx512,
}

fn detect() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if let Ok(v) = std::env::var("RSOLVERS_ISA") {
            match v.as_str() {
                "generic" => return Isa::Generic,
                "avx2" if is_x86_feature_detected!("avx2") => return Isa::Avx2,
                "avx512" if is_x86_feature_detected!("avx512f") => return Isa::Avx512,
                _ => {}
            }
        }
        if is_x86_feature_detected!("avx512f") {
            return Isa::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
    }
    Isa::Generic
}

/// The instruction-set tier selected at run time.
pub fn isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(detect)
}

/// Route a kernel call to the selected ISA tier.
macro_rules! dispatch {
    ($name:ident($($arg:expr),*)) => {{
        #[cfg(target_arch = "x86_64")]
        {
            match isa() {
                // SAFETY: the tier was selected only after the CPU reported the feature.
                Isa::Avx512 => unsafe { avx512::$name($($arg),*) },
                Isa::Avx2 => unsafe { avx2::$name($($arg),*) },
                Isa::Generic => unsafe { generic::$name($($arg),*) },
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unsafe { generic::$name($($arg),*) }
        }
    }};
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    dispatch!(dot(a, b))
}

pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    dispatch!(axpy(alpha, x, y))
}

pub fn aypx(beta: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    dispatch!(aypx(beta, x, y))
}

pub fn axpy2(alpha: f64, p: &[f64], x: &mut [f64], w: &[f64], r: &mut [f64]) {
    assert!(p.len() == x.len() && w.len() == x.len() && r.len() == x.len());
    dispatch!(axpy2(alpha, p, x, w, r))
}

pub fn cg_update(
    alpha: f64,
    p: &[f64],
    x: &mut [f64],
    w: &[f64],
    r: &mut [f64],
    z: &mut [f64],
    d: Option<&[f64]>,
) -> [f64; 3] {
    let n = x.len();
    assert!(p.len() == n && w.len() == n && r.len() == n);
    if let Some(d) = d {
        assert!(d.len() == n && z.len() == n);
    }
    dispatch!(cg_update(alpha, p, x, w, r, z, d))
}

pub fn dots_cols(data: &[f64], n: usize, s: usize, e: usize, w: &[f64], part: &mut [f64]) {
    assert!(s <= e && e <= n && w.len() == e - s && data.len() >= part.len() * n);
    dispatch!(dots_cols(data, n, s, e, w, part))
}

pub fn sub_cols(data: &[f64], n: usize, s: usize, k: usize, h: &[f64], wc: &mut [f64]) {
    assert!(s + wc.len() <= n && h.len() >= k && data.len() >= k * n);
    dispatch!(sub_cols(data, n, s, k, h, wc))
}

pub fn add_cols(data: &[f64], n: usize, s: usize, k: usize, c: &[f64], yc: &mut [f64]) {
    assert!(s + yc.len() <= n && c.len() >= k && data.len() >= k * n);
    dispatch!(add_cols(data, n, s, k, c, yc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::random_vector;

    /// Every ISA tier must produce bit-identical results.
    #[test]
    fn all_tiers_bit_identical() {
        let n = 1000 + 37;
        let (a, b) = (random_vector(n, 1), random_vector(n, 2));
        let d = random_vector(n, 3);
        let cols = random_vector(n * 9, 4);
        let h = random_vector(9, 5);

        let run = |f: &dyn Fn() -> Vec<f64>| f();
        macro_rules! tier {
            ($m:ident) => {
                run(&|| unsafe {
                    let mut out = vec![$m::dot(&a, &b)];
                    let (mut x, mut r, mut z) = (a.clone(), b.clone(), vec![0.0; n]);
                    out.extend($m::cg_update(0.37, &a, &mut x, &b, &mut r, &mut z, Some(&d)));
                    out.extend(&x);
                    out.extend(&r);
                    out.extend(&z);
                    let mut y = a.clone();
                    $m::axpy(0.5, &b, &mut y);
                    $m::aypx(0.25, &a, &mut y);
                    out.extend(&y);
                    let mut w = b.clone();
                    $m::sub_cols(&cols, n, 7, 9, &h, &mut w[7..n]);
                    out.extend(&w);
                    let mut w = b.clone();
                    $m::add_cols(&cols, n, 0, 9, &h, &mut w[0..n]);
                    out.extend(&w);
                    let mut part = vec![0.0; 9];
                    $m::dots_cols(&cols, n, 3, n - 3, &a[3..n - 3], &mut part);
                    out.extend(&part);
                    out
                })
            };
        }
        let g = tier!(generic);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                assert_eq!(g, tier!(avx2), "avx2 differs from generic");
            }
            if is_x86_feature_detected!("avx512f") {
                assert_eq!(g, tier!(avx512), "avx512 differs from generic");
            }
        }
    }

    #[test]
    fn dot_matches_reference() {
        let n = 12_345;
        let (a, b) = (random_vector(n, 1), random_vector(n, 2));
        let r: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - r).abs() < 1e-11);
        assert_eq!(dot(&[], &[]), 0.0);
    }
}
