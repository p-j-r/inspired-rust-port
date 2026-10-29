//! Shifted sparse direct solves `(A − σB)⁻¹` with symbolic-factorisation reuse (port item S1).
//!
//! Shift-and-invert eigensolvers, frequency sweeps, the β fixed-point iteration for tilted
//! crystals and contour methods all factorise matrices with the **same sparsity pattern** and a
//! different shift. [`ShiftedFactorization`] builds the union pattern of `A` and `B` once, runs
//! `faer`'s symbolic analysis (ordering, elimination structure) once, and for every new shift
//! only reassembles the values (`A − σB`, one pass over the non-zeros) and re-runs the numeric
//! factorisation with the cached symbolic object.
//!
//! This file and `faer_glue.rs` are the only ones that use `faer`; they target faer 0.24.
//! If nyx pins another version, the calls to adapt are `SymbolicLu::try_new`,
//! `Lu::try_new_with_symbolic`, `SymbolicLlt::try_new`, `Llt::try_new_with_symbolic` and
//! `solve_in_place_with_conj` — all of which exist since faer 0.19 with the same shape.

use crate::csr::Csr;
use crate::faer_glue::FaerScalar;
use crate::linop::LinearOperator;
use crate::scalar::Scalar;
use faer::linalg::solvers::SolveCore;
use faer::sparse::linalg::solvers::{Llt, Lu, SymbolicLlt, SymbolicLu};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Side};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverError {
    /// The matrix is singular (or numerically so) at this shift.
    Singular,
    /// A Cholesky factorisation was required but the matrix is not positive definite.
    NotPositiveDefinite,
    OutOfMemory,
    /// More non-zeros than the 32-bit indices used here can address.
    TooLarge,
    InvalidArgument(&'static str),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Singular => write!(f, "matrix is singular at this shift"),
            SolverError::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            SolverError::OutOfMemory => write!(f, "out of memory in the sparse factorisation"),
            SolverError::TooLarge => write!(f, "matrix exceeds 32-bit index range"),
            SolverError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for SolverError {}

/// What the caller knows about `A − σB`, which decides the factorisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Definiteness {
    /// No assumption: sparse LU with partial pivoting (indefinite, complex symmetric, general).
    General,
    /// `A − σB` is symmetric/Hermitian positive definite at the shifts used (e.g. stiffness −
    /// σ·mass with σ below the spectrum): try Cholesky, fall back to LU if it fails.
    PositiveDefinite,
}

/// Which factorisation was used for the last shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Used {
    None,
    Cholesky,
    Lu,
}

/// Timings and counters of a [`ShiftedFactorization`] (seconds).
#[derive(Debug, Clone, Copy)]
pub struct FactorStats {
    /// Pattern union + symbolic analyses, paid once (LU and Cholesky analyses are separate and
    /// lazy).
    pub analyse_secs: f64,
    /// Number of symbolic analyses performed (stays at 1 per factorisation kind however many
    /// shifts are factorised: that is the reuse).
    pub symbolic_count: usize,
    /// Total and last numeric factorisation time, and the number of numeric factorisations.
    pub numeric_secs_total: f64,
    pub numeric_secs_last: f64,
    pub numeric_count: usize,
    /// Value reassembly time of the last shift.
    pub assemble_secs_last: f64,
    pub used: Used,
    /// Cholesky attempts that failed and fell back to LU.
    pub cholesky_fallbacks: usize,
    /// Right-hand sides solved.
    pub solves: usize,
    pub solve_secs_total: f64,
}

/// "Factorise at a shift, then solve": the interface the eigensolvers use. A backend other than
/// `faer` (MKL PARDISO, a threaded MUMPS, a GPU solver) implements this trait.
pub trait ShiftedSolver<T: Scalar> {
    fn dim(&self) -> usize;
    /// Factorises `A − σB`. May reuse everything that depends only on the sparsity pattern.
    fn set_shift(&mut self, sigma: T) -> Result<(), SolverError>;
    /// Solves `(A − σB) X = RHS` in place for `nrhs` columns of length `dim()`.
    fn solve_block_in_place(&self, rhs: &mut [T], nrhs: usize);
    fn stats(&self) -> FactorStats;
}

/// Union pattern (CSC) of `A`, `B` (or the identity), optionally restricted to the lower
/// triangle, with the values of `A` and `B` scattered onto it.
struct Assembled<T: Scalar> {
    col_ptr: Vec<u32>,
    row_idx: Vec<u32>,
    a: Vec<T>,
    b: Vec<T>,
}

fn assemble<T: Scalar>(a: &Csr<T>, b: Option<&Csr<T>>, lower: bool) -> Result<Assembled<T>, SolverError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SolverError::InvalidArgument("matrix must be square"));
    }
    if b.is_some_and(|b| b.nrows() != n || b.ncols() != n) {
        return Err(SolverError::InvalidArgument("A and B must have the same shape"));
    }
    if a.nnz() + n >= u32::MAX as usize / 2 {
        return Err(SolverError::TooLarge);
    }
    // Entries (col, row, a-value, b-value), gathered from CSR; the identity contributes the
    // diagonal when there is no B.
    let keep = |r: usize, c: usize| !lower || r >= c;
    let mut count = vec![0usize; n + 1];
    let mut tmp: Vec<(u32, u32, T, T)> = Vec::with_capacity(a.nnz() + b.map_or(n, |b| b.nnz()));
    for r in 0..n {
        let (cols, vals) = a.row(r);
        for (&c, &v) in cols.iter().zip(vals) {
            if keep(r, c as usize) {
                tmp.push((c, r as u32, v, T::ZERO));
            }
        }
    }
    match b {
        Some(b) => {
            for r in 0..n {
                let (cols, vals) = b.row(r);
                for (&c, &v) in cols.iter().zip(vals) {
                    if keep(r, c as usize) {
                        tmp.push((c, r as u32, T::ZERO, v));
                    }
                }
            }
        }
        None => {
            for r in 0..n {
                tmp.push((r as u32, r as u32, T::ZERO, T::ONE));
            }
        }
    }
    tmp.sort_by_key(|&(c, r, _, _)| (c, r));
    let (mut col_ptr, mut row_idx) = (vec![0u32; n + 1], Vec::with_capacity(tmp.len()));
    let (mut av, mut bv): (Vec<T>, Vec<T>) = (Vec::with_capacity(tmp.len()), Vec::with_capacity(tmp.len()));
    let mut last: Option<(u32, u32)> = None;
    for &(c, r, x, y) in &tmp {
        if last == Some((c, r)) {
            *av.last_mut().unwrap() += x;
            *bv.last_mut().unwrap() += y;
        } else {
            row_idx.push(r);
            av.push(x);
            bv.push(y);
            count[c as usize + 1] += 1;
            last = Some((c, r));
        }
    }
    for c in 0..n {
        count[c + 1] += count[c];
        col_ptr[c + 1] = count[c + 1] as u32;
    }
    Ok(Assembled { col_ptr, row_idx, a: av, b: bv })
}

/// `faer`-backed [`ShiftedSolver`] with symbolic reuse.
pub struct ShiftedFactorization<T: FaerScalar> {
    n: usize,
    definiteness: Definiteness,
    full: Assembled<T>,
    lower: Option<Assembled<T>>,
    vals: Vec<T>,
    sym_lu: Option<SymbolicLu<u32>>,
    sym_llt: Option<SymbolicLlt<u32>>,
    lu: Option<Lu<u32, T::Elem>>,
    llt: Option<Llt<u32, T::Elem>>,
    stats: FactorStats,
}

impl<T: FaerScalar> ShiftedFactorization<T> {
    /// Prepares the pattern for `A − σB` (`B = None` means the identity). With
    /// [`Definiteness::PositiveDefinite`], `A` and `B` must be symmetric/Hermitian (only their
    /// lower triangles are used by the Cholesky path).
    pub fn new(a: &Csr<T>, b: Option<&Csr<T>>, definiteness: Definiteness) -> Result<Self, SolverError> {
        let t0 = Instant::now();
        let full = assemble(a, b, false)?;
        let lower = if definiteness == Definiteness::PositiveDefinite { Some(assemble(a, b, true)?) } else { None };
        let nnz = full.row_idx.len();
        Ok(Self {
            n: a.nrows(),
            definiteness,
            full,
            lower,
            vals: vec![T::ZERO; nnz],
            sym_lu: None,
            sym_llt: None,
            lu: None,
            llt: None,
            stats: FactorStats {
                analyse_secs: t0.elapsed().as_secs_f64(),
                symbolic_count: 0,
                numeric_secs_total: 0.0,
                numeric_secs_last: 0.0,
                numeric_count: 0,
                assemble_secs_last: 0.0,
                used: Used::None,
                cholesky_fallbacks: 0,
                solves: 0,
                solve_secs_total: 0.0,
            },
        })
    }

    fn pattern(asm: &Assembled<T>, n: usize) -> SymbolicSparseColMatRef<'_, u32> {
        SymbolicSparseColMatRef::new_checked(n, n, &asm.col_ptr, None, &asm.row_idx)
    }

    fn shifted_values(asm: &Assembled<T>, sigma: T, out: &mut Vec<T>) {
        out.clear();
        out.extend(asm.a.iter().zip(&asm.b).map(|(&x, &y)| x - sigma * y));
    }

    fn try_cholesky(&mut self, sigma: T) -> Result<bool, SolverError> {
        let n = self.n;
        let lower = self.lower.as_ref().expect("lower pattern exists for PositiveDefinite");
        if self.sym_llt.is_none() {
            let t0 = Instant::now();
            self.sym_llt =
                Some(SymbolicLlt::try_new(Self::pattern(lower, n), Side::Lower).map_err(|_| SolverError::OutOfMemory)?);
            self.stats.analyse_secs += t0.elapsed().as_secs_f64();
            self.stats.symbolic_count += 1;
        }
        let mut vals = std::mem::take(&mut self.vals);
        Self::shifted_values(lower, sigma, &mut vals);
        let t0 = Instant::now();
        let mat = SparseColMatRef::new(Self::pattern(lower, n), T::faer_slice(&vals));
        let res = Llt::try_new_with_symbolic(self.sym_llt.clone().unwrap(), mat, Side::Lower);
        self.stats.numeric_secs_last = t0.elapsed().as_secs_f64();
        self.vals = vals;
        match res {
            Ok(f) => {
                self.llt = Some(f);
                self.lu = None;
                self.stats.used = Used::Cholesky;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    fn do_lu(&mut self, sigma: T) -> Result<(), SolverError> {
        let n = self.n;
        if self.sym_lu.is_none() {
            let t0 = Instant::now();
            self.sym_lu =
                Some(SymbolicLu::try_new(Self::pattern(&self.full, n)).map_err(|_| SolverError::OutOfMemory)?);
            self.stats.analyse_secs += t0.elapsed().as_secs_f64();
            self.stats.symbolic_count += 1;
        }
        let mut vals = std::mem::take(&mut self.vals);
        Self::shifted_values(&self.full, sigma, &mut vals);
        let t0 = Instant::now();
        let mat = SparseColMatRef::new(Self::pattern(&self.full, n), T::faer_slice(&vals));
        let res = Lu::try_new_with_symbolic(self.sym_lu.clone().unwrap(), mat);
        self.stats.numeric_secs_last = t0.elapsed().as_secs_f64();
        self.vals = vals;
        match res {
            Ok(f) => {
                self.lu = Some(f);
                self.llt = None;
                self.stats.used = Used::Lu;
                Ok(())
            }
            Err(_) => Err(SolverError::Singular),
        }
    }
}

impl<T: FaerScalar> ShiftedSolver<T> for ShiftedFactorization<T> {
    fn dim(&self) -> usize {
        self.n
    }

    fn set_shift(&mut self, sigma: T) -> Result<(), SolverError> {
        let t_asm = Instant::now();
        let mut done = false;
        let result = (|| {
            if self.definiteness == Definiteness::PositiveDefinite {
                if self.try_cholesky(sigma)? {
                    done = true;
                } else {
                    self.stats.cholesky_fallbacks += 1;
                }
            }
            if !done { self.do_lu(sigma) } else { Ok(()) }
        })();
        // Assembly is a few streaming passes; report factor time separately from the total.
        self.stats.assemble_secs_last = t_asm.elapsed().as_secs_f64() - self.stats.numeric_secs_last;
        if result.is_ok() {
            self.stats.numeric_secs_total += self.stats.numeric_secs_last;
            self.stats.numeric_count += 1;
        }
        result
    }

    fn solve_block_in_place(&self, rhs: &mut [T], nrhs: usize) {
        let n = self.n;
        assert!(rhs.len() >= n * nrhs);
        let m = MatMut::from_column_major_slice_mut(T::faer_slice_mut(&mut rhs[..n * nrhs]), n, nrhs);
        match (&self.llt, &self.lu) {
            (Some(f), _) => f.solve_in_place_with_conj(Conj::No, m),
            (None, Some(f)) => f.solve_in_place_with_conj(Conj::No, m),
            (None, None) => panic!("set_shift must succeed before solving"),
        }
    }

    fn stats(&self) -> FactorStats {
        self.stats
    }
}

/// The spectral transformation `x ↦ (A − σB)⁻¹ B x` as a [`LinearOperator`], so that the
/// eigenvalues `θ` of the operator relate to the original ones by `λ = σ + 1/θ`.
///
/// `solver` is shared mutably only through `&self` (solves do not modify the factorisation);
/// call [`ShiftedSolver::set_shift`] on it before constructing the operator.
pub struct ShiftInvertOperator<'a, T: Scalar, S: ShiftedSolver<T>> {
    solver: &'a S,
    b: Option<&'a Csr<T>>,
}

impl<'a, T: Scalar, S: ShiftedSolver<T>> ShiftInvertOperator<'a, T, S> {
    pub fn new(solver: &'a S, b: Option<&'a Csr<T>>) -> Self {
        Self { solver, b }
    }
}

impl<T: Scalar, S: ShiftedSolver<T>> LinearOperator<T> for ShiftInvertOperator<'_, T, S> {
    fn dim(&self) -> usize {
        self.solver.dim()
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.apply_block(x, y, 1)
    }
    fn apply_block(&self, x: &[T], y: &mut [T], nrhs: usize) {
        let n = self.dim();
        match self.b {
            Some(b) => {
                for c in 0..nrhs {
                    b.spmv(&x[c * n..(c + 1) * n], &mut y[c * n..(c + 1) * n]);
                }
            }
            None => y[..n * nrhs].copy_from_slice(&x[..n * nrhs]),
        }
        self.solver.solve_block_in_place(&mut y[..n * nrhs], nrhs);
    }
    fn apply_block_with_bx(&self, _x: &[T], bx: &[T], y: &mut [T], nrhs: usize) {
        let n = self.dim();
        y[..n * nrhs].copy_from_slice(&bx[..n * nrhs]);
        self.solver.solve_block_in_place(&mut y[..n * nrhs], nrhs);
    }
}
