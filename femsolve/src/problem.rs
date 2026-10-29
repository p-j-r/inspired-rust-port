//! The solver interface used by models: "eigenpairs of `A x = λ B x` near a shift".
//!
//! [`ShiftInvertEigenSolver`] is the small trait the rest of nyx should depend on, so that a
//! backend can be swapped (the existing Arnoldi code, this Krylov–Schur, Jacobi–Davidson later,
//! a contour method). [`KrylovSchurShiftInvert`] implements it with a symbolic-reusing sparse
//! factorisation ([`crate::direct`]) and [`crate::eps::krylov_schur`], and always recomputes the
//! residual `‖Ax − λBx‖` independently of the eigensolver's own estimate.

use crate::basis::Basis;
use crate::csr::Csr;
use crate::direct::{Definiteness, FactorStats, ShiftInvertOperator, ShiftedFactorization, ShiftedSolver, SolverError};
use crate::eps::{EpsError, EpsOptions, EpsReason, Kind, Timings, Which, krylov_schur};
use crate::faer_glue::FaerScalar;
use crate::linop::LinearOperator;
use crate::scalar::{C64, Scalar};
use crate::vecops::norm2;

/// Structure of the pencil `(A, B)`, which selects the projected problem and inner product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// `A` Hermitian (real symmetric for real scalars) and `B` Hermitian positive definite (or
    /// absent), the shift real: the shifted operator is self-adjoint in the `B` inner product.
    Hermitian,
    /// Anything else, including complex-symmetric PML problems (needs a complex scalar).
    General,
}

/// The generalised eigenproblem `A x = λ B x` (`B = None` means the identity).
#[derive(Debug, Clone)]
pub struct EigenProblem<T: Scalar> {
    pub a: Csr<T>,
    pub b: Option<Csr<T>>,
    pub symmetry: Symmetry,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProblemError {
    Solver(SolverError),
    Eps(EpsError),
    InvalidArgument(&'static str),
}

impl std::fmt::Display for ProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProblemError::Solver(e) => write!(f, "factorisation failed: {e}"),
            ProblemError::Eps(e) => write!(f, "eigensolver failed: {e}"),
            ProblemError::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for ProblemError {}

impl From<SolverError> for ProblemError {
    fn from(e: SolverError) -> Self {
        ProblemError::Solver(e)
    }
}
impl From<EpsError> for ProblemError {
    fn from(e: EpsError) -> Self {
        ProblemError::Eps(e)
    }
}

/// Result of one `solve_near`.
#[derive(Debug)]
pub struct EigenPairs<T: Scalar> {
    pub shift: T,
    /// Eigenvalues `λ` of the original problem, nearest to the shift first.
    pub values: Vec<C64>,
    /// Eigenvectors (columns): `B`-orthonormal for [`Symmetry::Hermitian`], unit 2-norm otherwise.
    pub vectors: Basis<T>,
    /// Independently recomputed relative residuals `‖Ax − λBx‖₂ / (|λ|·‖x‖₂)`.
    pub residuals: Vec<f64>,
    pub converged: bool,
    pub restarts: usize,
    pub matvecs: usize,
    pub breakdowns: usize,
    pub eps_timings: Timings,
    pub factor_stats: FactorStats,
}

/// "Eigenpairs near a shift" interface (see module docs).
pub trait ShiftInvertEigenSolver<T: Scalar> {
    /// The `nev` eigenvalues of `A x = λ B x` closest to `sigma`, with vectors.
    fn solve_near(&mut self, sigma: T, nev: usize) -> Result<EigenPairs<T>, ProblemError>;
}

/// Krylov–Schur with shift-and-invert on a symbolic-reusing sparse factorisation.
///
/// Create once per `(A, B)` pattern; every `solve_near` with a new shift redoes only the
/// numeric factorisation.
pub struct KrylovSchurShiftInvert<'p, T: FaerScalar> {
    problem: &'p EigenProblem<T>,
    solver: ShiftedFactorization<T>,
    pub opts: EpsOptions,
}

impl<'p, T: FaerScalar> KrylovSchurShiftInvert<'p, T> {
    /// `definiteness`: pass [`Definiteness::PositiveDefinite`] when `A − σB` is positive
    /// definite at the shifts that will be used (stiffness − σ·mass below the spectrum), to use
    /// the much cheaper Cholesky factorisation; it falls back to LU automatically if it fails.
    pub fn new(problem: &'p EigenProblem<T>, definiteness: Definiteness) -> Result<Self, ProblemError> {
        if problem.symmetry == Symmetry::General && !T::IS_COMPLEX {
            return Err(ProblemError::InvalidArgument(
                "General symmetry needs a complex scalar (real non-symmetric Schur forms are not supported)",
            ));
        }
        let solver = ShiftedFactorization::new(&problem.a, problem.b.as_ref(), definiteness)?;
        let kind = match problem.symmetry {
            Symmetry::Hermitian => Kind::Hermitian,
            Symmetry::General => Kind::General,
        };
        let opts = EpsOptions { which: Which::LargestMagnitude, ..match kind {
            Kind::Hermitian => EpsOptions::hermitian(1),
            Kind::General => EpsOptions::general(1),
        } };
        Ok(Self { problem, solver, opts })
    }

    pub fn factor_stats(&self) -> FactorStats {
        self.solver.stats()
    }
}

impl<T: FaerScalar> ShiftInvertEigenSolver<T> for KrylovSchurShiftInvert<'_, T> {
    fn solve_near(&mut self, sigma: T, nev: usize) -> Result<EigenPairs<T>, ProblemError> {
        if self.problem.symmetry == Symmetry::Hermitian && sigma.im() != 0.0 {
            return Err(ProblemError::InvalidArgument("a Hermitian problem needs a real shift"));
        }
        self.solver.set_shift(sigma)?;
        let p = self.problem;
        let op = ShiftInvertOperator::new(&self.solver, p.b.as_ref());
        // The shifted operator is self-adjoint in the B inner product only for Hermitian
        // problems; general ones use the Euclidean product.
        let weight: Option<&dyn LinearOperator<T>> = match (p.symmetry, p.b.as_ref()) {
            (Symmetry::Hermitian, Some(b)) => Some(b),
            _ => None,
        };
        let opts = EpsOptions { nev, ..self.opts.clone() };
        let res = krylov_schur(&op, weight, None, &opts)?;

        let n = p.a.nrows();
        let s = sigma.to_c64();
        let values: Vec<C64> = res.values.iter().map(|&th| s + C64::ONE / th).collect();
        let mut residuals = Vec::with_capacity(values.len());
        let (mut ax, mut bx) = (vec![T::ZERO; n], vec![T::ZERO; n]);
        for (i, &lam) in values.iter().enumerate() {
            let x = res.vectors.col(i);
            p.a.spmv(x, &mut ax);
            match &p.b {
                Some(b) => b.spmv(x, &mut bx),
                None => bx.copy_from_slice(x),
            }
            let r2: f64 = ax.iter().zip(&bx).map(|(a, b)| (a.to_c64() - lam * b.to_c64()).norm_sqr()).sum();
            residuals.push(r2.sqrt() / (lam.norm() * norm2(x)));
        }
        Ok(EigenPairs {
            shift: sigma,
            values,
            vectors: res.vectors,
            residuals,
            converged: res.reason == EpsReason::Converged,
            restarts: res.iterations,
            matvecs: res.matvecs,
            breakdowns: res.breakdowns,
            eps_timings: res.timings,
            factor_stats: self.solver.stats(),
        })
    }
}
