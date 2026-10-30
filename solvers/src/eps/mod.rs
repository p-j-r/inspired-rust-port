//! Eigenvalue solvers (SLEPc `EPS`): Krylov–Schur for symmetric (Hermitian) problems.

mod krylov_schur;

pub use krylov_schur::krylov_schur;

use crate::basis::Basis;

/// Which eigenvalues to compute (SLEPc `EPSWhich`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Which {
    /// Largest `|λ|` (SLEPc default).
    #[default]
    LargestMagnitude,
    SmallestMagnitude,
    LargestReal,
    SmallestReal,
}

impl Which {
    /// Sort key: smaller means "more wanted".
    pub(crate) fn key(self, lambda: f64) -> f64 {
        match self {
            Which::LargestMagnitude => -lambda.abs(),
            Which::SmallestMagnitude => lambda.abs(),
            Which::LargestReal => -lambda,
            Which::SmallestReal => lambda,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EpsOptions {
    /// Number of eigenpairs wanted.
    pub nev: usize,
    /// Maximum dimension of the Krylov basis; default `min(n, max(2·nev, nev + 15))`.
    pub ncv: Option<usize>,
    /// Relative residual tolerance `‖Ax − λx‖ ≤ tol·|λ|` (SLEPc default `1e-8`).
    pub tol: f64,
    /// Maximum number of restarts; default `max(100, 2n/ncv)`.
    pub max_it: Option<usize>,
    pub which: Which,
    /// Fraction of the unconverged Ritz vectors kept at each restart (SLEPc default `0.5`).
    pub keep: f64,
}

impl EpsOptions {
    pub fn new(nev: usize) -> Self {
        Self { nev, ncv: None, tol: 1e-8, max_it: None, which: Which::default(), keep: 0.5 }
    }
}

impl Default for EpsOptions {
    fn default() -> Self {
        Self::new(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsReason {
    /// At least `nev` eigenpairs converged.
    Converged,
    /// Restart limit reached.
    DivergedIts,
    /// The Krylov space became invariant and no further start vector could be built.
    DivergedBreakdown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EpsError {
    InvalidArgument(&'static str),
}

impl std::fmt::Display for EpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EpsError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for EpsError {}

/// Converged (or best available) eigenpairs, ordered from most to least wanted.
#[derive(Debug)]
pub struct EigenResult {
    pub reason: EpsReason,
    pub values: Vec<f64>,
    /// Relative error estimates `‖r‖/|λ|` for each returned pair.
    pub errors: Vec<f64>,
    /// Eigenvectors as the leading `values.len()` columns (unit 2-norm, mutually orthogonal).
    pub vectors: Basis,
    /// Number of converged pairs (may exceed `nev`).
    pub nconv: usize,
    /// Number of restarts performed (SLEPc `EPSGetIterationNumber`).
    pub iterations: usize,
    /// Operator applications.
    pub matvecs: usize,
}

impl EigenResult {
    pub fn vector(&self, i: usize) -> &[f64] {
        self.vectors.col(i)
    }
}
