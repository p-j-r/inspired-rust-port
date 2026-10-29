use super::{PcError, Preconditioner, check_square_sorted};
use crate::sparse::CsrMatrix;

/// Symmetric SOR / SSOR with a zero initial guess (PETSc `PCSOR`, default
/// `SOR_LOCAL_SYMMETRIC_SWEEP`, `its = lits = 1`).
///
/// `M⁻¹ = ω(2−ω) (D+ωU)⁻¹ D (D+ωL)⁻¹`; with `ω = 1` this is symmetric Gauss–Seidel.
#[derive(Debug, Clone)]
pub struct Sor<'a> {
    a: &'a CsrMatrix,
    diag_pos: Vec<usize>,
    omega: f64,
}

impl<'a> Sor<'a> {
    pub fn new(a: &'a CsrMatrix, omega: f64) -> Result<Self, PcError> {
        check_square_sorted(a)?;
        assert!(omega > 0.0 && omega < 2.0, "omega must be in (0, 2)");
        let mut diag_pos = Vec::with_capacity(a.nrows());
        for i in 0..a.nrows() {
            let (cols, vals) = a.row(i);
            match cols.binary_search(&(i as u32)) {
                Ok(k) if vals[k] != 0.0 => diag_pos.push(a.indptr()[i] + k),
                Ok(_) => return Err(PcError::ZeroPivot(i)),
                Err(_) => return Err(PcError::MissingDiagonal(i)),
            }
        }
        Ok(Self { a, diag_pos, omega })
    }
}

impl Preconditioner for Sor<'_> {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.a.nrows();
        assert!(r.len() == n && z.len() == n);
        let (ip, ix, va) = (self.a.indptr(), self.a.indices(), self.a.values());
        let w = self.omega;
        // Forward: (D/ω + L) y = r, stored in z.
        for i in 0..n {
            let mut s = r[i];
            for k in ip[i]..self.diag_pos[i] {
                s -= va[k] * z[ix[k] as usize];
            }
            z[i] = w * s / va[self.diag_pos[i]];
        }
        // Scale by (2−ω)/ω · D, then backward: (D/ω + U) z = t.
        let scale = (2.0 - w) / w;
        for v in 0..n {
            z[v] *= scale * va[self.diag_pos[v]];
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in self.diag_pos[i] + 1..ip[i + 1] {
                s -= va[k] * z[ix[k] as usize];
            }
            z[i] = w * s / va[self.diag_pos[i]];
        }
    }
}
