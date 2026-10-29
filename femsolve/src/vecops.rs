//! Level-1 operations on vectors of any [`Scalar`], with the fixed-chunk reduction order.

use crate::par;
use crate::scalar::Scalar;

/// Rows per reduction chunk. Part of the numerical contract: changing it changes the last bits
/// of every dot product.
pub const CHUNK: usize = 2048;

/// `Σ conj(xᵢ)·yᵢ`.
pub fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len());
    let nch = x.len().div_ceil(CHUNK);
    let parts = par::chunk_map(nch, par::worth_parallel(x.len()), |c| {
        let (s, e) = (c * CHUNK, ((c + 1) * CHUNK).min(x.len()));
        T::k_dotc(&x[s..e], &y[s..e])
    });
    parts.into_iter().fold(T::ZERO, |a, b| a + b)
}

/// `Σ xᵢ·yᵢ` without conjugation.
pub fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    assert_eq!(x.len(), y.len());
    let nch = x.len().div_ceil(CHUNK);
    let parts = par::chunk_map(nch, par::worth_parallel(x.len()), |c| {
        let (s, e) = (c * CHUNK, ((c + 1) * CHUNK).min(x.len()));
        T::k_dotu(&x[s..e], &y[s..e])
    });
    parts.into_iter().fold(T::ZERO, |a, b| a + b)
}

/// Squared 2-norm `Σ |xᵢ|²`.
pub fn norm2_sq<T: Scalar>(x: &[T]) -> f64 {
    dotc(x, x).re()
}

/// 2-norm (no over/underflow protection beyond `f64` range).
pub fn norm2<T: Scalar>(x: &[T]) -> f64 {
    norm2_sq(x).sqrt()
}

/// `y ← y + a·x`.
pub fn axpy<T: Scalar>(a: T, x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len());
    let par = par::worth_parallel(x.len());
    par::for_chunks_mut(y, CHUNK, par, |c, yc| {
        let s = c * CHUNK;
        T::k_axpy(a, &x[s..s + yc.len()], yc);
    });
}

/// `y ← a·y`.
pub fn scale<T: Scalar>(a: T, y: &mut [T]) {
    let par = par::worth_parallel(y.len());
    par::for_chunks_mut(y, CHUNK, par, |_, yc| {
        for v in yc {
            *v *= a;
        }
    });
}

/// `y ← r·y` for a real `r`.
pub fn scale_real<T: Scalar>(r: f64, y: &mut [T]) {
    let par = par::worth_parallel(y.len());
    par::for_chunks_mut(y, CHUNK, par, |_, yc| {
        for v in yc {
            *v = v.scale(r);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::random_vector;
    use crate::scalar::C64;

    fn check<T: Scalar>() {
        let n = 3 * CHUNK + 17;
        let (x, y) = (random_vector::<T>(n, 1), random_vector::<T>(n, 2));
        let want = x.iter().zip(&y).fold(T::ZERO, |a, (p, q)| a + p.conj() * *q);
        assert!((dotc(&x, &y) - want).modulus() < 1e-10);
        let nrm = norm2(&x);
        assert!((nrm * nrm - x.iter().map(|v| v.modulus_sq()).sum::<f64>()).abs() < 1e-8);
        let mut z = y.clone();
        axpy(T::from_parts(0.5, -0.25), &x, &mut z);
        for i in (0..n).step_by(509) {
            assert!((z[i] - (y[i] + T::from_parts(0.5, -0.25) * x[i])).modulus() < 1e-14);
        }
    }

    #[test]
    fn real_and_complex() {
        check::<f64>();
        check::<C64>();
    }
}
