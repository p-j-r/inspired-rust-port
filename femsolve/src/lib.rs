//! # femsolve
//!
//! Solver building blocks for finite-element eigenproblems, written once for real and complex
//! scalars: a thick-restart (block) Krylov–Schur eigensolver, shift-and-invert with symbolic
//! factorisation reuse, Matrix Market export of problems, and a harness to compare against
//! SLEPc. See `docs/STATUS.md` for what is done, what was measured and what is still open.

pub mod basis;
pub mod bundle;
pub mod csr;
pub mod dense;
pub mod direct;
pub mod eps;
pub mod faer_glue;
pub mod kernels;
pub mod linop;
pub mod mm;
pub mod par;
pub mod problem;
pub mod rng;
pub mod scalar;
pub mod vecops;

pub use csr::Csr;
pub use linop::{FnOperator, Identity, LinearOperator, Preconditioner};
pub use scalar::{C64, Scalar};
