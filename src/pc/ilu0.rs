use super::{PcError, Preconditioner, check_square_sorted};
use crate::sparse::CsrMatrix;

/// Incomplete LU with zero fill, natural ordering (PETSc `PCILU`, `levels = 0`).
///
/// `A ≈ L U` with `L` unit lower triangular and `L + U` restricted to the sparsity
/// pattern of `A`. For tridiagonal matrices this is the exact LU factorisation.
#[derive(Debug, Clone)]
pub struct Ilu0 {
    n: usize,
    // Strictly lower part (unit diagonal implied), CSR.
    l_ptr: Vec<usize>,
    l_idx: Vec<u32>,
    l_val: Vec<f64>,
    // Strictly upper part, CSR.
    u_ptr: Vec<usize>,
    u_idx: Vec<u32>,
    u_val: Vec<f64>,
    inv_diag: Vec<f64>,
}

impl Ilu0 {
    pub fn new(a: &CsrMatrix) -> Result<Self, PcError> {
        check_square_sorted(a)?;
        let n = a.nrows();
        let (ip, ix) = (a.indptr(), a.indices());
        let mut val = a.values().to_vec();
        let mut diag = vec![0usize; n];
        for i in 0..n {
            match ix[ip[i]..ip[i + 1]].binary_search(&(i as u32)) {
                Ok(k) => diag[i] = ip[i] + k,
                Err(_) => return Err(PcError::MissingDiagonal(i)),
            }
        }
        // IKJ variant: row i is eliminated against the already-factored rows k < i.
        let mut pos = vec![usize::MAX; n];
        for i in 0..n {
            for p in ip[i]..ip[i + 1] {
                pos[ix[p] as usize] = p;
            }
            for p in ip[i]..diag[i] {
                let k = ix[p] as usize;
                let lik = val[p] / val[diag[k]];
                val[p] = lik;
                for q in diag[k] + 1..ip[k + 1] {
                    let t = pos[ix[q] as usize];
                    if t != usize::MAX {
                        val[t] -= lik * val[q];
                    }
                }
            }
            for p in ip[i]..ip[i + 1] {
                pos[ix[p] as usize] = usize::MAX;
            }
            if val[diag[i]] == 0.0 || !val[diag[i]].is_finite() {
                return Err(PcError::ZeroPivot(i));
            }
        }
        let (mut l_ptr, mut l_idx, mut l_val) = (vec![0], Vec::new(), Vec::new());
        let (mut u_ptr, mut u_idx, mut u_val) = (vec![0], Vec::new(), Vec::new());
        let mut inv_diag = Vec::with_capacity(n);
        for i in 0..n {
            for p in ip[i]..diag[i] {
                l_idx.push(ix[p]);
                l_val.push(val[p]);
            }
            for p in diag[i] + 1..ip[i + 1] {
                u_idx.push(ix[p]);
                u_val.push(val[p]);
            }
            l_ptr.push(l_idx.len());
            u_ptr.push(u_idx.len());
            inv_diag.push(1.0 / val[diag[i]]);
        }
        Ok(Self { n, l_ptr, l_idx, l_val, u_ptr, u_idx, u_val, inv_diag })
    }
}

impl Preconditioner for Ilu0 {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.n;
        assert!(r.len() == n && z.len() == n);
        for i in 0..n {
            let mut s = r[i];
            let (lo, hi) = (self.l_ptr[i], self.l_ptr[i + 1]);
            for (&c, &v) in self.l_idx[lo..hi].iter().zip(&self.l_val[lo..hi]) {
                s -= v * z[c as usize];
            }
            z[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            let (lo, hi) = (self.u_ptr[i], self.u_ptr[i + 1]);
            for (&c, &v) in self.u_idx[lo..hi].iter().zip(&self.u_val[lo..hi]) {
                s -= v * z[c as usize];
            }
            z[i] = s * self.inv_diag[i];
        }
    }
}
