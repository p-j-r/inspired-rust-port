#![allow(dead_code)]
//! Dense reference implementations used as ground truth in the integration tests.

use rsolvers::CsrMatrix;

/// Solve `A x = b` for a dense row-major `A` by LU with partial pivoting.
pub fn dense_solve(a: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| m[i * n + k].abs().total_cmp(&m[j * n + k].abs())).unwrap();
        if p != k {
            for c in 0..n {
                m.swap(k * n + c, p * n + c);
            }
            x.swap(k, p);
        }
        let piv = m[k * n + k];
        assert!(piv != 0.0, "singular reference matrix");
        for i in k + 1..n {
            let f = m[i * n + k] / piv;
            if f != 0.0 {
                for c in k..n {
                    m[i * n + c] -= f * m[k * n + c];
                }
                x[i] -= f * x[k];
            }
        }
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for c in i + 1..n {
            s -= m[i * n + c] * x[c];
        }
        x[i] = s / m[i * n + i];
    }
    x
}

pub fn residual_norm(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.spmv(x, &mut ax);
    ax.iter().zip(b).map(|(p, q)| (q - p) * (q - p)).sum::<f64>().sqrt()
}

pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn rel_diff(x: &[f64], y: &[f64]) -> f64 {
    let d: f64 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
    d / norm(y).max(f64::MIN_POSITIVE)
}

/// Eigenvalues of the 1-D Dirichlet Laplacian `tridiag(-1, 2, -1)` of order `n`, ascending.
pub fn laplacian_1d_eigenvalues(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos())
        .collect()
}

/// Eigenvalues of the 2-D 5-point Laplacian on an `nx × ny` grid, ascending.
pub fn laplacian_2d_eigenvalues(nx: usize, ny: usize) -> Vec<f64> {
    let (ex, ey) = (laplacian_1d_eigenvalues(nx), laplacian_1d_eigenvalues(ny));
    let mut v: Vec<f64> = ex.iter().flat_map(|a| ey.iter().map(move |b| a + b)).collect();
    v.sort_by(f64::total_cmp);
    v
}
