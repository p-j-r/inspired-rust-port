//! Preconditioners (PETSc `PC`): Jacobi, SOR/SSOR, ILU(0) and IC(0).

mod ic0;
mod ilu0;
mod jacobi;
mod sor;

pub use crate::operator::{Identity, Preconditioner};
pub use ic0::Ic0;
pub use ilu0::Ilu0;
pub use jacobi::Jacobi;
pub use sor::Sor;

use std::fmt;

/// Failure to set up a preconditioner (PETSc `PC_FACTOR_ZEROPIVOT` and friends).
#[derive(Debug, Clone, PartialEq)]
pub enum PcError {
    NotSquare,
    /// Row `i` has no diagonal entry stored.
    MissingDiagonal(usize),
    /// A pivot at row `i` was exactly zero (or not positive for IC).
    ZeroPivot(usize),
    /// Rows must have strictly increasing column indices.
    UnsortedRows,
}

impl fmt::Display for PcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcError::NotSquare => write!(f, "matrix is not square"),
            PcError::MissingDiagonal(i) => write!(f, "no diagonal entry in row {i}"),
            PcError::ZeroPivot(i) => write!(f, "zero or non-positive pivot at row {i}"),
            PcError::UnsortedRows => write!(f, "matrix rows must be sorted with unique columns"),
        }
    }
}

impl std::error::Error for PcError {}

pub(crate) fn check_square_sorted(a: &crate::sparse::CsrMatrix) -> Result<(), PcError> {
    if a.nrows() != a.ncols() {
        return Err(PcError::NotSquare);
    }
    if !a.has_sorted_unique_rows() {
        return Err(PcError::UnsortedRows);
    }
    Ok(())
}
