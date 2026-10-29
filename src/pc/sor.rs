use super::tri::TriSolver;
use super::{PcError, Preconditioner, check_square_sorted};
use crate::sparse::CsrMatrix;

/// Symmetric SOR / SSOR with a zero initial guess (PETSc `PCSOR`, default
/// `SOR_LOCAL_SYMMETRIC_SWEEP`, `its = lits = 1`).
///
/// `M⁻¹ = ω(2−ω) (D+ωU)⁻¹ D (D+ωL)⁻¹`; with `ω = 1` this is symmetric Gauss–Seidel.
///
/// With `winvᵢ = ω/aᵢᵢ` the two sweeps are `yᵢ = winvᵢ rᵢ − Σ_{j<i} winvᵢ aᵢⱼ yⱼ` and
/// `xᵢ = (2−ω) yᵢ − Σ_{j>i} winvᵢ aᵢⱼ xⱼ`; the scaled triangles are stored so no multiply
/// sits on the dependency chain.
#[derive(Debug, Clone)]
pub struct Sor {
    tri: TriSolver,
}

impl Sor {
    pub fn new(a: &CsrMatrix, omega: f64) -> Result<Self, PcError> {
        check_square_sorted(a)?;
        assert!(omega > 0.0 && omega < 2.0, "omega must be in (0, 2)");
        let n = a.nrows();
        let (mut lp, mut li, mut lv) = (vec![0usize], Vec::new(), Vec::new());
        let (mut up, mut ui, mut uv) = (vec![0usize], Vec::new(), Vec::new());
        let mut winv = Vec::with_capacity(n);
        for i in 0..n {
            let (cols, vals) = a.row(i);
            let k = match cols.binary_search(&(i as u32)) {
                Ok(k) if vals[k] != 0.0 => k,
                Ok(_) => return Err(PcError::ZeroPivot(i)),
                Err(_) => return Err(PcError::MissingDiagonal(i)),
            };
            let wi = omega / vals[k];
            winv.push(wi);
            for (p, (&c, &v)) in cols.iter().zip(vals).enumerate() {
                match p.cmp(&k) {
                    std::cmp::Ordering::Less => {
                        li.push(c);
                        lv.push(wi * v);
                    }
                    std::cmp::Ordering::Greater => {
                        ui.push(c);
                        uv.push(wi * v);
                    }
                    std::cmp::Ordering::Equal => {}
                }
            }
            lp.push(li.len());
            up.push(ui.len());
        }
        let tri = TriSolver::new(n, (lp, li, lv), (up, ui, uv), Some(winv), vec![2.0 - omega; n]);
        Ok(Self { tri })
    }
}

impl Preconditioner for Sor {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        self.tri.solve(r, z);
    }
}
