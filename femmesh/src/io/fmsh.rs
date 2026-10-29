//! `.fmsh`: a compact, lossless, self-checking binary mesh format.
//!
//! The point of the format is portability: a mixed-cell mesh with labels round-trips bit for bit
//! in a fraction of the size of a text mesh, and a CRC-32 catches truncation and corruption.
//!
//! Layout (little endian; `varint` = LEB128, `zigzag` = signed to unsigned):
//!
//! ```text
//! "FMSH" version:u8(=1) D:u8 | n_points n_cells n_facets (varint)
//! points   : per axis one of three codings, whichever is smallest (flag byte first):
//!            0 raw (8 bytes per value); 1 Gorilla-style XOR with the previous value (one control
//!            byte per value, 0xFF for "same", else lead_zero_bytes << 3 | (len - 1), followed by
//!            the `len` significant bytes); 2 dictionary (count, the distinct values in order of
//!            first appearance, then one varint index per value). The dictionary wins on
//!            tensor-product and structured PML layers, where an axis has few distinct values;
//!            arbitrary unstructured coordinates are close to incompressible and stay raw.
//! cells    : run-length list of (cell type code, count); then per cell the first node as
//!            zigzag delta against the previous cell's first node and the other nodes as zigzag
//!            deltas against the cell's first node
//! labels   : run-length list of (zigzag label, count)
//! facets   : run-length list of node counts; connectivity coded like cells; run-length labels
//! crc32    : u32 over everything before it
//! ```
//!
//! Deflate/zstd are deliberately not used so that the crate stays dependency-free; the output is
//! friendly to an external compressor (small integers, regular byte patterns).

use crate::cellmesh::{CellMesh, CellType};

const MAGIC: &[u8; 4] = b"FMSH";
const VERSION: u8 = 1;

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// CRC-32 (IEEE, reflected, polynomial 0xEDB88320), bitwise implementation.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Result<u8, String> {
        let v = *self.b.get(self.i).ok_or("unexpected end of data")?;
        self.i += 1;
        Ok(v)
    }
    fn varint(&mut self) -> Result<u64, String> {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let b = self.byte()?;
            v |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err("varint too long".into());
            }
        }
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.i..self.i + n).ok_or("unexpected end of data")?;
        self.i += n;
        Ok(s)
    }
}

fn encode_axis(vals: impl Iterator<Item = f64> + Clone) -> Vec<u8> {
    let raw: Vec<u8> = vals.clone().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    let mut g = Vec::new();
    let mut prev = 0u64;
    for v in vals.clone() {
        let bits = v.to_bits();
        let x = bits ^ prev;
        prev = bits;
        if x == 0 {
            g.push(0xFF);
        } else {
            let lz = (x.leading_zeros() / 8) as u8;
            let tz = (x.trailing_zeros() / 8) as u8;
            let len = 8 - lz - tz;
            g.push((lz << 3) | (len - 1));
            g.extend_from_slice(&(x >> (8 * tz as u32)).to_le_bytes()[..len as usize]);
            // trailing-zero count is recovered from lz and len: tz = 8 - lz - len
        }
    }
    // dictionary of distinct bit patterns in first-appearance order
    let mut index: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    let mut table: Vec<u64> = Vec::new();
    let mut idx: Vec<u8> = Vec::new();
    for v in vals {
        let bits = v.to_bits();
        let k = *index.entry(bits).or_insert_with(|| {
            table.push(bits);
            table.len() as u64 - 1
        });
        put_varint(&mut idx, k);
    }
    let dict_len = 10 + table.len() * 8 + idx.len();
    let mut out = Vec::new();
    if dict_len < g.len().min(raw.len()) {
        out.push(2);
        put_varint(&mut out, table.len() as u64);
        for b in table {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out.extend_from_slice(&idx);
    } else if g.len() < raw.len() {
        out.push(1);
        out.extend_from_slice(&g);
    } else {
        out.push(0);
        out.extend_from_slice(&raw);
    }
    out
}

fn decode_axis(r: &mut Reader, n: usize) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(n);
    match r.byte()? {
        0 => {
            for _ in 0..n {
                let b = r.bytes(8)?;
                out.push(f64::from_bits(u64::from_le_bytes(b.try_into().unwrap())));
            }
        }
        1 => {
            let mut prev = 0u64;
            for _ in 0..n {
                let c = r.byte()?;
                let x = if c == 0xFF {
                    0
                } else {
                    let (lz, len) = ((c >> 3) as u32, (c & 7) as usize + 1);
                    if lz as usize + len > 8 {
                        return Err("corrupt coordinate control byte".into());
                    }
                    let tz = 8 - lz - len as u32;
                    let mut buf = [0u8; 8];
                    buf[..len].copy_from_slice(r.bytes(len)?);
                    u64::from_le_bytes(buf) << (8 * tz)
                };
                prev ^= x;
                out.push(f64::from_bits(prev));
            }
        }
        2 => {
            let nt = r.varint()? as usize;
            let mut table = Vec::with_capacity(nt.min(n));
            for _ in 0..nt {
                table.push(f64::from_bits(u64::from_le_bytes(r.bytes(8)?.try_into().unwrap())));
            }
            for _ in 0..n {
                let k = r.varint()? as usize;
                out.push(*table.get(k).ok_or("dictionary index out of range")?);
            }
        }
        f => return Err(format!("unknown coordinate flag {f}")),
    }
    Ok(out)
}

fn put_runs(out: &mut Vec<u8>, vals: impl Iterator<Item = i64>) {
    let mut runs: Vec<(i64, u64)> = Vec::new();
    for v in vals {
        match runs.last_mut() {
            Some((x, n)) if *x == v => *n += 1,
            _ => runs.push((v, 1)),
        }
    }
    put_varint(out, runs.len() as u64);
    for (v, n) in runs {
        put_varint(out, zigzag(v));
        put_varint(out, n);
    }
}

fn get_runs(r: &mut Reader, n: usize) -> Result<Vec<i64>, String> {
    let nr = r.varint()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..nr {
        let v = unzigzag(r.varint()?);
        let c = r.varint()? as usize;
        if out.len() + c > n {
            return Err("run-length overflow".into());
        }
        out.extend(std::iter::repeat_n(v, c));
    }
    if out.len() != n {
        return Err("run-length underflow".into());
    }
    Ok(out)
}

fn put_conn(out: &mut Vec<u8>, conn: &[u32], offsets: &[u32]) {
    let mut prev_first = 0i64;
    for w in offsets.windows(2) {
        let vs = &conn[w[0] as usize..w[1] as usize];
        let first = vs[0] as i64;
        put_varint(out, zigzag(first - prev_first));
        prev_first = first;
        for &v in &vs[1..] {
            put_varint(out, zigzag(v as i64 - first));
        }
    }
}

fn get_conn(r: &mut Reader, sizes: &[usize], n_points: usize) -> Result<Vec<u32>, String> {
    let mut conn = Vec::with_capacity(sizes.iter().sum());
    let mut prev_first = 0i64;
    for &s in sizes {
        let first = prev_first + unzigzag(r.varint()?);
        prev_first = first;
        conn.push(first as u32);
        for _ in 1..s {
            conn.push((first + unzigzag(r.varint()?)) as u32);
        }
    }
    if conn.iter().any(|&v| v as usize >= n_points) {
        return Err("vertex id out of range".into());
    }
    Ok(conn)
}

/// Encode a mixed-cell mesh.
pub fn encode<const D: usize>(m: &CellMesh<D>) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(MAGIC);
    o.push(VERSION);
    o.push(D as u8);
    put_varint(&mut o, m.n_points() as u64);
    put_varint(&mut o, m.n_cells() as u64);
    put_varint(&mut o, m.n_facets() as u64);
    for a in 0..D {
        o.extend(encode_axis(m.points.iter().map(|p| p[a])));
    }
    put_runs(&mut o, m.types.iter().map(|t| t.code() as i64));
    put_conn(&mut o, &m.conn, &m.offsets);
    put_runs(&mut o, m.label.iter().map(|&l| l as i64));
    put_runs(&mut o, (0..m.n_facets()).map(|f| m.facet(f).len() as i64));
    put_conn(&mut o, &m.facet_conn, &m.facet_offsets);
    put_runs(&mut o, m.facet_label.iter().map(|&l| l as i64));
    let crc = crc32(&o);
    o.extend_from_slice(&crc.to_le_bytes());
    o
}

/// Decode a mesh written by [`encode`], verifying the checksum.
pub fn decode<const D: usize>(bytes: &[u8]) -> Result<CellMesh<D>, String> {
    if bytes.len() < 4 + 2 + 4 {
        return Err("too short".into());
    }
    let (body, tail) = bytes.split_at(bytes.len() - 4);
    if crc32(body) != u32::from_le_bytes(tail.try_into().unwrap()) {
        return Err("checksum mismatch".into());
    }
    let mut r = Reader { b: body, i: 0 };
    if r.bytes(4)? != MAGIC {
        return Err("bad magic".into());
    }
    if r.byte()? != VERSION {
        return Err("unsupported version".into());
    }
    if r.byte()? as usize != D {
        return Err("dimension mismatch".into());
    }
    let (np, nc, nf) = (r.varint()? as usize, r.varint()? as usize, r.varint()? as usize);
    let axes: Vec<Vec<f64>> = (0..D).map(|_| decode_axis(&mut r, np)).collect::<Result<_, _>>()?;
    let mut m = CellMesh::<D>::new();
    m.points = (0..np).map(|i| std::array::from_fn(|a| axes[a][i])).collect();
    let tcodes = get_runs(&mut r, nc)?;
    m.types = tcodes
        .iter()
        .map(|&c| CellType::from_code(c as u8).filter(|t| t.dim() == D).ok_or("bad cell type".to_string()))
        .collect::<Result<_, _>>()?;
    let sizes: Vec<usize> = m.types.iter().map(|t| t.nv()).collect();
    m.conn = get_conn(&mut r, &sizes, np)?;
    m.offsets = std::iter::once(0)
        .chain(sizes.iter().scan(0u32, |s, &n| {
            *s += n as u32;
            Some(*s)
        }))
        .collect();
    m.label = get_runs(&mut r, nc)?.into_iter().map(|v| v as i32).collect();
    let fsizes: Vec<usize> = get_runs(&mut r, nf)?.into_iter().map(|v| v as usize).collect();
    m.facet_conn = get_conn(&mut r, &fsizes, np)?;
    m.facet_offsets = std::iter::once(0)
        .chain(fsizes.iter().scan(0u32, |s, &n| {
            *s += n as u32;
            Some(*s)
        }))
        .collect();
    m.facet_label = get_runs(&mut r, nf)?.into_iter().map(|v| v as i32).collect();
    if r.i != body.len() {
        return Err("trailing bytes".into());
    }
    Ok(m)
}

/// Write to a file.
pub fn save<const D: usize>(path: &std::path::Path, m: &CellMesh<D>) -> std::io::Result<()> {
    std::fs::write(path, encode(m))
}

/// Read from a file.
pub fn load<const D: usize>(path: &std::path::Path) -> Result<CellMesh<D>, String> {
    decode(&std::fs::read(path).map_err(|e| e.to_string())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{box_tet, box_tri};

    fn same<const D: usize>(a: &CellMesh<D>, b: &CellMesh<D>) {
        assert_eq!(a.points.len(), b.points.len());
        for (p, q) in a.points.iter().zip(&b.points) {
            for k in 0..D {
                assert_eq!(p[k].to_bits(), q[k].to_bits());
            }
        }
        assert_eq!(a.types, b.types);
        assert_eq!(a.conn, b.conn);
        assert_eq!(a.offsets, b.offsets);
        assert_eq!(a.label, b.label);
        assert_eq!(a.facet_conn, b.facet_conn);
        assert_eq!(a.facet_offsets, b.facet_offsets);
        assert_eq!(a.facet_label, b.facet_label);
    }

    #[test]
    fn varint_and_zigzag_roundtrip() {
        for v in [0i64, 1, -1, 63, -64, 1 << 40, -(1 << 40), i64::MAX, i64::MIN] {
            assert_eq!(unzigzag(zigzag(v)), v);
            let mut o = Vec::new();
            put_varint(&mut o, zigzag(v));
            assert_eq!(unzigzag(Reader { b: &o, i: 0 }.varint().unwrap()), v);
        }
    }

    #[test]
    fn crc_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn roundtrip_2d_3d_and_mixed() {
        let a = CellMesh::from_simplex(&box_tri([[0.0, 1.0], [-2.0, 3.5]], 17, 9));
        same(&a, &decode::<2>(&encode(&a)).unwrap());
        let b = CellMesh::from_simplex(&box_tet([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], [4, 3, 2]));
        same(&b, &decode::<3>(&encode(&b)).unwrap());
        let mut c = CellMesh::<3>::new();
        c.points = vec![
            [0.1, 0.2, 0.3],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
        ];
        c.push_cell(CellType::Hex, &[0, 1, 2, 3, 4, 5, 6, 7], -4);
        c.push_cell(CellType::Prism, &[1, 8, 2, 5, 9, 6], 12);
        c.push_facet(&[0, 1, 2, 3], 5);
        c.push_facet(&[1, 8, 2], 6);
        same(&c, &decode::<3>(&encode(&c)).unwrap());
    }

    #[test]
    fn special_floats_survive() {
        let mut m = CellMesh::<2>::new();
        m.points = vec![[0.0, -0.0], [f64::MIN_POSITIVE, 1e300], [f64::INFINITY, f64::NAN], [0.1, 0.1]];
        m.push_cell(CellType::Tri, &[0, 1, 3], 1);
        same(&m, &decode::<2>(&encode(&m)).unwrap());
    }

    #[test]
    fn detects_corruption_truncation_and_wrong_dimension() {
        let a = CellMesh::from_simplex(&box_tri([[0.0, 1.0], [0.0, 1.0]], 5, 5));
        let e = encode(&a);
        let mut bad = e.clone();
        bad[20] ^= 0x40;
        assert!(decode::<2>(&bad).is_err());
        assert!(decode::<2>(&e[..e.len() - 7]).is_err());
        assert!(decode::<3>(&e).is_err());
    }

    #[test]
    fn structured_meshes_compress_well() {
        let m = box_tri([[0.0, 1.0], [0.0, 1.0]], 100, 100);
        let a = CellMesh::from_simplex(&m);
        let enc = encode(&a);
        let mut txt = Vec::new();
        crate::io::vtk::write_simplex(&mut txt, &m, &Default::default()).unwrap();
        // plain binary arrays: 8 bytes per coordinate + 4 per index + 4 per label
        let plain = a.n_points() * 16 + a.conn.len() * 4 + a.n_cells() * 4 + a.facet_conn.len() * 4 + a.n_facets() * 4;
        println!("fmsh {} B, plain binary {} B, vtk text {} B", enc.len(), plain, txt.len());
        assert!(enc.len() * 4 < plain);
        assert!(enc.len() * 5 < txt.len());
    }
}
