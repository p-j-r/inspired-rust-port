use super::{PcError, Preconditioner};
use crate::par::{self, CHUNK};
use crate::sparse::CsrMatrix;
use crate::vecops::dot_chunk;

/// Point Jacobi, `M = diag(A)` (PETSc `PCJACOBI`).
#[derive(Debug, Clone)]
pub struct Jacobi {
    inv_diag: Vec<f64>,
}

impl Jacobi {
    pub fn new(a: &CsrMatrix) -> Result<Self, PcError> {
        if a.nrows() != a.ncols() {
            return Err(PcError::NotSquare);
        }
        let d = a.diagonal();
        if let Some(i) = d.iter().position(|&v| v == 0.0) {
            return Err(PcError::ZeroPivot(i));
        }
        Ok(Self { inv_diag: d.into_iter().map(|v| 1.0 / v).collect() })
    }
}

impl Preconditioner for Jacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        assert!(r.len() == self.inv_diag.len() && z.len() == r.len());
        par::for_chunks_mut(z, CHUNK, |s, zc| {
            for (k, zi) in zc.iter_mut().enumerate() {
                *zi = self.inv_diag[s + k] * r[s + k];
            }
        });
    }

    fn apply_dots(&self, r: &[f64], z: &mut [f64]) -> (f64, f64) {
        assert!(r.len() == self.inv_diag.len() && z.len() == r.len());
        let n = z.len();
        if par::parallel_enabled(n) {
            // Rare path; keep the bit-for-bit contract by reusing the unfused code.
            self.apply(r, z);
            return crate::vecops::dot_and_sq(z, r);
        }
        let (mut zr, mut zz) = (0.0, 0.0);
        let mut s = 0;
        while s < n {
            let e = (s + CHUNK).min(n);
            for k in s..e {
                z[k] = self.inv_diag[k] * r[k];
            }
            zr += dot_chunk(&z[s..e], &r[s..e]);
            zz += dot_chunk(&z[s..e], &z[s..e]);
            s = e;
        }
        (zr, zz)
    }
}
