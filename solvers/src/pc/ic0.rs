use super::{PcError, Preconditioner, check_square_sorted};
use super::tri::TriSolver;
use crate::sparse::CsrMatrix;

/// Incomplete Cholesky with zero fill (PETSc `PCICC`, `levels = 0`): `A ≈ L Lᵀ`, with
/// `L` restricted to the lower-triangular pattern of `A`. Only the lower triangle of
/// `A` is read, so `A` is assumed symmetric.
#[derive(Debug, Clone)]
pub struct Ic0 {
    tri: TriSolver,
}

impl Ic0 {
    pub fn new(a: &CsrMatrix) -> Result<Self, PcError> {
        check_square_sorted(a)?;
        let n = a.nrows();
        let mut ptr = vec![0usize];
        let mut idx: Vec<u32> = Vec::new();
        let mut val: Vec<f64> = Vec::new();
        for i in 0..n {
            let (cols, vals) = a.row(i);
            let mut have_diag = false;
            for (&c, &v) in cols.iter().zip(vals) {
                if (c as usize) <= i {
                    have_diag |= c as usize == i;
                    idx.push(c);
                    val.push(v);
                }
            }
            if !have_diag {
                return Err(PcError::MissingDiagonal(i));
            }
            ptr.push(idx.len());
        }
        let mut inv_diag = vec![0.0; n]; // 1 / lᵢᵢ
        for i in 0..n {
            let (lo, hi) = (ptr[i], ptr[i + 1]);
            for p in lo..hi - 1 {
                let j = idx[p] as usize;
                // Sparse dot of row i and row j over columns < j (both sorted).
                let (jlo, jhi) = (ptr[j], ptr[j + 1] - 1);
                let (mut x, mut y) = (lo, jlo);
                let mut s = 0.0;
                while x < p && y < jhi {
                    match idx[x].cmp(&idx[y]) {
                        std::cmp::Ordering::Less => x += 1,
                        std::cmp::Ordering::Greater => y += 1,
                        std::cmp::Ordering::Equal => {
                            s += val[x] * val[y];
                            x += 1;
                            y += 1;
                        }
                    }
                }
                val[p] = (val[p] - s) * inv_diag[j];
            }
            let mut d = val[hi - 1];
            for p in lo..hi - 1 {
                d -= val[p] * val[p];
            }
            if !(d > 0.0) || !d.is_finite() {
                return Err(PcError::ZeroPivot(i));
            }
            let l_ii = d.sqrt();
            val[hi - 1] = l_ii;
            inv_diag[i] = 1.0 / l_ii;
        }
        // Strict lower triangle scaled to unit diagonal, `L₁ = L diag(1/lⱼⱼ)`, and its transpose.
        // Then `(L Lᵀ)⁻¹ r` is a unit-lower solve, a diagonal scaling by `1/lᵢᵢ²`, and a
        // unit-upper solve, with no multiply on either dependency chain.
        let (mut l_ptr, mut l_idx, mut l_val) = (vec![0usize], Vec::new(), Vec::new());
        let mut count = vec![0usize; n + 1];
        for i in 0..n {
            let (lo, hi) = (ptr[i], ptr[i + 1] - 1);
            for p in lo..hi {
                let c = idx[p] as usize;
                l_idx.push(idx[p]);
                l_val.push(val[p] * inv_diag[c]);
                count[c + 1] += 1;
            }
            l_ptr.push(l_idx.len());
        }
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let u_ptr = count.clone();
        let mut next = count;
        let mut u_idx = vec![0u32; l_idx.len()];
        let mut u_val = vec![0.0; l_idx.len()];
        for i in 0..n {
            for p in l_ptr[i]..l_ptr[i + 1] {
                let c = l_idx[p] as usize;
                u_idx[next[c]] = i as u32;
                u_val[next[c]] = l_val[p];
                next[c] += 1;
            }
        }
        let inv_diag_sq: Vec<f64> = inv_diag.iter().map(|d| d * d).collect();
        let tri = TriSolver::new(n, (l_ptr, l_idx, l_val), (u_ptr, u_idx, u_val), None, inv_diag_sq);
        Ok(Self { tri })
    }
}

impl Preconditioner for Ic0 {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        self.tri.solve(r, z);
    }
}
