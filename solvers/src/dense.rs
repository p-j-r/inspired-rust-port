//! Small dense kernels for the projected problems (the role of SLEPc's `DS`).

/// Eigendecomposition `A = Q diag(w) Qᵀ` of a symmetric `m × m` matrix by cyclic Jacobi.
///
/// `a` is column-major with leading dimension `lda`; only symmetric input is meaningful
/// (the upper triangle is mirrored). Returns eigenvalues in ascending order and the
/// orthonormal eigenvectors as the columns of a column-major `m × m` matrix.
///
/// Jacobi is chosen for its high relative accuracy and simplicity; the projected
/// problems here are small (`m ≈ ncv`), far below the cost of one operator application.
pub fn sym_eig(a: &[f64], lda: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut s = vec![0.0; m * m];
    for j in 0..m {
        for i in 0..=j {
            let v = a[j * lda + i];
            s[j * m + i] = v;
            s[i * m + j] = v;
        }
    }
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    let fro: f64 = s.iter().map(|v| v * v).sum::<f64>().sqrt();
    if fro > 0.0 {
        for _sweep in 0..60 {
            let mut off = 0.0;
            for j in 0..m {
                for i in 0..j {
                    off += s[j * m + i] * s[j * m + i];
                }
            }
            if off.sqrt() <= f64::EPSILON * 1e-2 * fro {
                break;
            }
            for p in 0..m {
                for r in p + 1..m {
                    let apr = s[r * m + p];
                    if apr == 0.0 {
                        continue;
                    }
                    let (app, arr) = (s[p * m + p], s[r * m + r]);
                    let theta = (arr - app) / (2.0 * apr);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let sn = t * c;
                    // Rotate columns p and r (S ← S J), then rows (S ← Jᵀ S).
                    for k in 0..m {
                        let (x, y) = (s[p * m + k], s[r * m + k]);
                        s[p * m + k] = c * x - sn * y;
                        s[r * m + k] = sn * x + c * y;
                    }
                    for k in 0..m {
                        let (x, y) = (s[k * m + p], s[k * m + r]);
                        s[k * m + p] = c * x - sn * y;
                        s[k * m + r] = sn * x + c * y;
                    }
                    for k in 0..m {
                        let (x, y) = (q[p * m + k], q[r * m + k]);
                        q[p * m + k] = c * x - sn * y;
                        q[r * m + k] = sn * x + c * y;
                    }
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| s[i * m + i].total_cmp(&s[j * m + j]));
    let w: Vec<f64> = order.iter().map(|&i| s[i * m + i]).collect();
    let mut qs = vec![0.0; m * m];
    for (dst, &src) in order.iter().enumerate() {
        qs[dst * m..(dst + 1) * m].copy_from_slice(&q[src * m..(src + 1) * m]);
    }
    (w, qs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn random_sym(m: usize, seed: u64) -> Vec<f64> {
        let mut r = SplitMix64::new(seed);
        let mut a = vec![0.0; m * m];
        for j in 0..m {
            for i in 0..=j {
                let v = r.next_signed();
                a[j * m + i] = v;
                a[i * m + j] = v;
            }
        }
        a
    }

    #[test]
    fn reconstructs_matrix_and_is_orthonormal() {
        for &m in &[1usize, 2, 5, 17, 40] {
            let a = random_sym(m, m as u64);
            let (w, q) = sym_eig(&a, m, m);
            assert!(w.windows(2).all(|p| p[0] <= p[1]));
            for i in 0..m {
                for j in 0..m {
                    let qq: f64 = (0..m).map(|k| q[i * m + k] * q[j * m + k]).sum();
                    assert!((qq - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13, "orthonormality");
                    let r: f64 = (0..m).map(|k| q[k * m + i] * w[k] * q[k * m + j]).sum();
                    assert!((r - a[j * m + i]).abs() < 1e-12, "reconstruction m={m}");
                }
            }
        }
    }

    #[test]
    fn known_spectrum() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let (w, _) = sym_eig(&[2.0, 1.0, 1.0, 2.0], 2, 2);
        assert!((w[0] - 1.0).abs() < 1e-14 && (w[1] - 3.0).abs() < 1e-14);
        // Repeated eigenvalues and zero matrix.
        let (w, _) = sym_eig(&[0.0; 9], 3, 3);
        assert_eq!(w, vec![0.0; 3]);
        let (w, _) = sym_eig(&[5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0], 3, 3);
        assert_eq!(w, vec![5.0; 3]);
    }
}
