//! Scalar Helmholtz equation with PML on triangles (P1) and quadrilaterals (Q1).
//!
//! Time convention `exp(-i omega t)`, wavenumber `k = omega / c`. In a PML region with stretched
//! axes the coordinate is complexified, `s_a(x) = 1 + i sigma_a(x) / k`, with the polynomial profile
//! `sigma_a = sigma_max (d / t)^m` in the distance `d` from the interface plane and
//! `sigma_max = -(m + 1) ln(R0) / (2 t)`, where `R0` is the design reflection at normal incidence
//! of the continuous layer. The weak form is
//!
//! ```text
//! sum  (s_y/s_x) u_x v_x + (s_x/s_y) u_y v_y - k^2 s_x s_y u v  =  sum s_x s_y f v
//! ```
//!
//! evaluated at the quadrature points of every element (so the profile is sampled, not averaged),
//! with homogeneous Dirichlet conditions on the outer boundary of the mesh.

use super::band::{BandMatrix, rcm};
use super::cplx::{C64, I, ONE, ZERO};
use crate::adapt::pml::PmlMesh;
use crate::cellmesh::{CellMesh, CellType};

/// PML profile parameters.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    /// Polynomial order `m` of the conductivity profile.
    pub order: f64,
    /// Design reflection `R0` of the continuous layer (e.g. `1e-8`).
    pub r0: f64,
    /// Switch the stretching off (the layers then act as plain material): for sanity checks.
    pub enabled: bool,
}

impl Default for Profile {
    fn default() -> Self {
        Profile { order: 2.0, r0: 1e-8, enabled: true }
    }
}

/// Result of a solve.
pub struct Solution {
    pub u: Vec<C64>,
    pub n_dof: usize,
    pub bandwidth: usize,
    pub residual: f64,
}

struct Qp {
    w: f64,
    x: [f64; 2],
    n: [f64; 4],
    dn: [[f64; 2]; 4],
}

fn quadrature(m: &CellMesh<2>, c: usize) -> Result<Vec<Qp>, String> {
    let vs = m.cell(c);
    let p = |i: usize| m.points[vs[i] as usize];
    match m.types[c] {
        CellType::Tri => {
            let (a, b, cc) = (p(0), p(1), p(2));
            let det = (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0]);
            let area = 0.5 * det;
            // gradients of the barycentric coordinates
            let g = [
                [(b[1] - cc[1]) / det, (cc[0] - b[0]) / det],
                [(cc[1] - a[1]) / det, (a[0] - cc[0]) / det],
                [(a[1] - b[1]) / det, (b[0] - a[0]) / det],
            ];
            let mut out = Vec::new();
            for (l0, l1, l2) in [(0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)] {
                let x = [l0 * a[0] + l1 * b[0] + l2 * cc[0], l0 * a[1] + l1 * b[1] + l2 * cc[1]];
                out.push(Qp { w: area / 3.0, x, n: [l0, l1, l2, 0.0], dn: [g[0], g[1], g[2], [0.0; 2]] });
            }
            Ok(out)
        }
        CellType::Quad => {
            let q = 1.0 / 3f64.sqrt();
            let sgn = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            let mut out = Vec::new();
            for &(xi, eta) in &sgn {
                let (xi, eta) = (xi * q, eta * q);
                let n: [f64; 4] = std::array::from_fn(|i| 0.25 * (1.0 + sgn[i].0 * xi) * (1.0 + sgn[i].1 * eta));
                let dxi: [f64; 4] = std::array::from_fn(|i| 0.25 * sgn[i].0 * (1.0 + sgn[i].1 * eta));
                let deta: [f64; 4] = std::array::from_fn(|i| 0.25 * sgn[i].1 * (1.0 + sgn[i].0 * xi));
                let (mut x, mut y, mut xx, mut xe, mut yx, mut ye) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                for i in 0..4 {
                    let pi = p(i);
                    x += n[i] * pi[0];
                    y += n[i] * pi[1];
                    xx += dxi[i] * pi[0];
                    xe += deta[i] * pi[0];
                    yx += dxi[i] * pi[1];
                    ye += deta[i] * pi[1];
                }
                let det = xx * ye - xe * yx;
                if det <= 0.0 {
                    return Err(format!("quad {c}: non-positive Jacobian"));
                }
                // physical gradient = J^{-T} * reference gradient
                let dn: [[f64; 2]; 4] =
                    std::array::from_fn(|i| [(ye * dxi[i] - yx * deta[i]) / det, (-xe * dxi[i] + xx * deta[i]) / det]);
                out.push(Qp { w: det, x: [x, y], n, dn });
            }
            Ok(out)
        }
        t => Err(format!("unsupported cell type {t:?} in the 2D harness")),
    }
}

/// Solve `-Laplace u - k^2 u = f` with the PML described by `pm` (pass `None` for a plain
/// mesh without stretching). `f` is the real source density.
pub fn solve(
    mesh: &CellMesh<2>,
    pml: Option<(&PmlMesh<2>, Profile)>,
    k: f64,
    f: &dyn Fn(&[f64; 2]) -> f64,
) -> Result<Solution, String> {
    let n = mesh.n_points();
    // stretching factors at a point of a cell with region code `code`
    let stretch = |label: i32, x: &[f64; 2]| -> [C64; 2] {
        let mut s = [ONE, ONE];
        if let Some((pm, prof)) = pml {
            if !prof.enabled {
                return s;
            }
            if let Some((code, _)) = pm.spec.decode(label) {
                for a in 0..2 {
                    if code[a] != 0 {
                        let side = if code[a] < 0 { 0 } else { 1 };
                        let t = pm.spec.thickness[a][side];
                        let d = if side == 1 { x[a] - pm.interface[a][1] } else { pm.interface[a][0] - x[a] };
                        let d = d.max(0.0);
                        let sig_max = -(prof.order + 1.0) * prof.r0.ln() / (2.0 * t);
                        let sigma = sig_max * (d / t).powf(prof.order);
                        s[a] = C64::new(1.0, sigma / k);
                    }
                }
            }
        }
        s
    };

    // adjacency, ordering
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for c in 0..mesh.n_cells() {
        let vs = mesh.cell(c);
        for &a in vs {
            for &b in vs {
                if a != b {
                    adj[a as usize].push(b);
                }
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let perm = rcm(&adj);
    let mut pos = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        pos[old as usize] = new;
    }
    let bw = (0..n)
        .flat_map(|u| adj[u].iter().map(move |&v| (u, v as usize)))
        .map(|(u, v)| pos[u].abs_diff(pos[v]))
        .max()
        .unwrap_or(0);
    let mut a = BandMatrix::new(n, bw);
    let mut rhs = vec![ZERO; n];

    let bb = crate::mesh::bbox_of(&mesh.points);
    let size = (bb[0][1] - bb[0][0]).max(bb[1][1] - bb[1][0]);
    let on_outer = |p: &[f64; 2]| {
        (0..2).any(|ax| (p[ax] - bb[ax][0]).abs() < 1e-9 * size || (p[ax] - bb[ax][1]).abs() < 1e-9 * size)
    };
    let fixed: Vec<bool> = mesh.points.iter().map(on_outer).collect();
    let k2 = k * k;
    for c in 0..mesh.n_cells() {
        let vs = mesh.cell(c);
        let nn = vs.len();
        let label = mesh.label[c];
        for q in quadrature(mesh, c)? {
            let s = stretch(label, &q.x);
            let (sx, sy) = (s[0], s[1]);
            let (cxx, cyy, cm) = (sy / sx * q.w, sx / sy * q.w, sx * sy * q.w);
            let fv = f(&q.x);
            for i in 0..nn {
                let gi = pos[vs[i] as usize];
                rhs[gi] += cm * (fv * q.n[i]);
                for j in 0..nn {
                    let gj = pos[vs[j] as usize];
                    let kij =
                        cxx * (q.dn[i][0] * q.dn[j][0]) + cyy * (q.dn[i][1] * q.dn[j][1]) - cm * (k2 * q.n[i] * q.n[j]);
                    a.add(gi, gj, kij);
                }
            }
        }
    }
    // Dirichlet: replace rows and columns of fixed nodes by identity
    for v in 0..n {
        if fixed[v] {
            let g = pos[v];
            let (lo, hi) = (g.saturating_sub(bw), (g + bw).min(n - 1));
            for j in lo..=hi {
                let cur = a.get(g, j);
                a.add(g, j, -cur);
                let cur = a.get(j, g);
                a.add(j, g, -cur);
            }
            a.add(g, g, ONE);
            rhs[g] = ZERO;
        }
    }
    let (x, residual) = a.solve(&rhs)?;
    let mut u = vec![ZERO; n];
    for v in 0..n {
        u[v] = x[pos[v]];
    }
    let _ = I;
    Ok(Solution { u, n_dof: n, bandwidth: bw, residual })
}

/// Relative L2 difference `||u - v|| / ||v||` over the vertex ids `0..n`.
pub fn rel_l2(u: &[C64], v: &[C64], n: usize) -> f64 {
    let num: f64 = (0..n).map(|i| (u[i] - v[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = (0..n).map(|i| v[i].norm_sqr()).sum::<f64>().sqrt();
    num / den
}

/// Gaussian source of width `w` centred at `x0`.
pub fn gaussian_source(x0: [f64; 2], w: f64) -> impl Fn(&[f64; 2]) -> f64 {
    move |x| {
        let r2 = (x[0] - x0[0]).powi(2) + (x[1] - x0[1]).powi(2);
        (-r2 / (2.0 * w * w)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapt::pml::{PmlSpec, build_pml};
    use crate::fixtures::box_tri;
    use std::f64::consts::PI;

    fn quad_grid(n: usize) -> CellMesh<2> {
        let mut m = CellMesh::<2>::new();
        for j in 0..=n {
            for i in 0..=n {
                m.points.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        let id = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        for j in 0..n {
            for i in 0..n {
                m.push_cell(CellType::Quad, &[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], 1);
            }
        }
        m
    }

    fn manufactured_error(mesh: &CellMesh<2>, k: f64) -> f64 {
        let f = move |x: &[f64; 2]| (2.0 * PI * PI - k * k) * (PI * x[0]).sin() * (PI * x[1]).sin();
        let s = solve(mesh, None, k, &f).unwrap();
        assert!(s.residual < 1e-10);
        let mut e: f64 = 0.0;
        for (i, p) in mesh.points.iter().enumerate() {
            let ex = (PI * p[0]).sin() * (PI * p[1]).sin();
            e = e.max((s.u[i] - C64::real(ex)).abs());
        }
        e
    }

    #[test]
    fn p1_and_q1_converge_quadratically() {
        let k = 3.0;
        let p1 = |n| CellMesh::from_simplex(&box_tri([[0.0, 1.0], [0.0, 1.0]], n, n));
        let (a, b) = (manufactured_error(&p1(8), k), manufactured_error(&p1(16), k));
        assert!(a / b > 3.4 && a / b < 4.6, "P1 ratio {}", a / b);
        let (a, b) = (manufactured_error(&quad_grid(8), k), manufactured_error(&quad_grid(16), k));
        assert!(a / b > 3.4 && a / b < 4.6, "Q1 ratio {}", a / b);
    }

    /// Physical-domain error against a thick-PML reference on the same interface mesh.
    fn pml_error(n_layers: usize, thickness: f64, enabled: bool, split: bool) -> f64 {
        let k = 2.0 * PI;
        let phys = box_tri([[0.0, 1.5], [0.0, 1.5]], 15, 15);
        let src = gaussian_source([0.75, 0.75], 0.12);
        let prof = Profile { enabled, ..Default::default() };
        let run = |spec: &PmlSpec<2>, split: bool, prof: Profile| {
            let pm = build_pml(&phys, spec).unwrap();
            let mesh =
                if split { CellMesh::from_simplex(&pm.mesh.split_to_simplices().unwrap()) } else { pm.mesh.clone() };
            let s = solve(&mesh, Some((&pm, prof)), k, &src).unwrap();
            assert!(s.residual < 1e-8, "residual {}", s.residual);
            s.u
        };
        let reference = run(&PmlSpec::<2>::uniform(1.5, 24), false, Profile::default());
        let u = run(&PmlSpec::<2>::uniform(thickness, n_layers), split, prof);
        rel_l2(&u, &reference, phys.n_points())
    }

    #[test]
    fn pml_absorbs_and_stretching_matters() {
        let with = pml_error(6, 0.75, true, false);
        let without = pml_error(6, 0.75, false, false);
        assert!(with < 0.05, "PML error {with}");
        assert!(without > 5.0 * with, "no-PML error {without} vs {with}");
    }

    #[test]
    fn more_layers_help_and_split_agrees_to_discretisation_error() {
        let coarse = pml_error(2, 0.75, true, false);
        let fine = pml_error(8, 0.75, true, false);
        assert!(fine < coarse, "{fine} vs {coarse}");
        let native = pml_error(6, 0.75, true, false);
        let split = pml_error(6, 0.75, true, true);
        assert!(split < 0.1 && native < 0.1, "{native} {split}");
    }
}
