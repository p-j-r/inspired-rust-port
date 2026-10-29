use super::{PcError, Preconditioner, check_square_sorted};
use crate::sparse::CsrMatrix;

/// Incomplete Cholesky with zero fill (PETSc `PCICC`, `levels = 0`): `A ≈ L Lᵀ`, with
/// `L` restricted to the lower-triangular pattern of `A`. Only the lower triangle of
/// `A` is read, so `A` is assumed symmetric.
#[derive(Debug, Clone)]
pub struct Ic0 {
    n: usize,
    // Lower triangle by rows, diagonal stored *last* in each row.
    ptr: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<f64>,
    inv_diag: Vec<f64>,
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
        let mut inv_diag = vec![0.0; n];
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
        Ok(Self { n, ptr, idx, val, inv_diag })
    }
}

impl Preconditioner for Ic0 {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.n;
        assert!(r.len() == n && z.len() == n);
        // L y = r
        for i in 0..n {
            let (lo, hi) = (self.ptr[i], self.ptr[i + 1] - 1);
            let mut s = r[i];
            for (&c, &v) in self.idx[lo..hi].iter().zip(&self.val[lo..hi]) {
                s -= v * z[c as usize];
            }
            z[i] = s * self.inv_diag[i];
        }
        // Lᵀ z = y, scattering each finished z_i into the rows above it.
        for i in (0..n).rev() {
            let zi = z[i] * self.inv_diag[i];
            z[i] = zi;
            let (lo, hi) = (self.ptr[i], self.ptr[i + 1] - 1);
            for (&c, &v) in self.idx[lo..hi].iter().zip(&self.val[lo..hi]) {
                z[c as usize] -= v * zi;
            }
        }
    }
}
