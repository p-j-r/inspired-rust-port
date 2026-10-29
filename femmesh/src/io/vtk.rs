//! Legacy ASCII VTK writer (`UNSTRUCTURED_GRID`), readable by ParaView and VisIt.
//!
//! Cell labels are always written as the cell scalar `label`. Extra named scalar fields can be
//! attached to points or cells.

use crate::cellmesh::CellMesh;
use crate::mesh::SimplexMesh;
use std::io::{self, Write};

/// Extra scalar fields to write, as `(name, values)`.
#[derive(Default)]
pub struct Fields<'a> {
    pub point: Vec<(&'a str, &'a [f64])>,
    pub cell: Vec<(&'a str, &'a [f64])>,
}

/// Write a mixed-cell mesh.
pub fn write_cells<const D: usize, W: Write>(w: &mut W, m: &CellMesh<D>, f: &Fields) -> io::Result<()> {
    writeln!(w, "# vtk DataFile Version 3.0\nfemmesh\nASCII\nDATASET UNSTRUCTURED_GRID")?;
    writeln!(w, "POINTS {} double", m.n_points())?;
    for p in &m.points {
        let z = if D == 3 { p[2] } else { 0.0 };
        writeln!(w, "{:e} {:e} {:e}", p[0], p[1], z)?;
    }
    let total: usize = m.conn.len() + m.n_cells();
    writeln!(w, "CELLS {} {}", m.n_cells(), total)?;
    for c in 0..m.n_cells() {
        let vs = m.cell(c);
        write!(w, "{}", vs.len())?;
        for &l in m.types[c].vtk_order() {
            write!(w, " {}", vs[l as usize])?;
        }
        writeln!(w)?;
    }
    writeln!(w, "CELL_TYPES {}", m.n_cells())?;
    for t in &m.types {
        writeln!(w, "{}", t.vtk_id())?;
    }
    writeln!(w, "CELL_DATA {}", m.n_cells())?;
    writeln!(w, "SCALARS label int 1\nLOOKUP_TABLE default")?;
    for l in &m.label {
        writeln!(w, "{l}")?;
    }
    for (name, v) in &f.cell {
        assert_eq!(v.len(), m.n_cells());
        writeln!(w, "SCALARS {name} double 1\nLOOKUP_TABLE default")?;
        for x in *v {
            writeln!(w, "{x:e}")?;
        }
    }
    if !f.point.is_empty() {
        writeln!(w, "POINT_DATA {}", m.n_points())?;
        for (name, v) in &f.point {
            assert_eq!(v.len(), m.n_points());
            writeln!(w, "SCALARS {name} double 1\nLOOKUP_TABLE default")?;
            for x in *v {
                writeln!(w, "{x:e}")?;
            }
        }
    }
    Ok(())
}

/// Write a simplex mesh.
pub fn write_simplex<const D: usize, W: Write>(w: &mut W, m: &SimplexMesh<D>, f: &Fields) -> io::Result<()> {
    write_cells(w, &CellMesh::from_simplex(m), f)
}

/// Write a simplex mesh to a file path.
pub fn save_simplex<const D: usize>(path: &std::path::Path, m: &SimplexMesh<D>, f: &Fields) -> io::Result<()> {
    let mut w = io::BufWriter::new(std::fs::File::create(path)?);
    write_simplex(&mut w, m, f)
}

/// Write a mixed-cell mesh to a file path.
pub fn save_cells<const D: usize>(path: &std::path::Path, m: &CellMesh<D>, f: &Fields) -> io::Result<()> {
    let mut w = io::BufWriter::new(std::fs::File::create(path)?);
    write_cells(&mut w, m, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_expected_sections() {
        let m = crate::fixtures::box_tri([[0.0, 1.0], [0.0, 1.0]], 2, 2);
        let q: Vec<f64> = (0..m.n_cells()).map(|i| i as f64).collect();
        let mut buf = Vec::new();
        write_simplex(&mut buf, &m, &Fields { point: vec![], cell: vec![("q", &q)] }).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("POINTS 9 double"));
        assert!(s.contains("CELLS 8 32"));
        assert!(s.contains("CELL_TYPES 8"));
        assert!(s.contains("SCALARS q double 1"));
    }
}
