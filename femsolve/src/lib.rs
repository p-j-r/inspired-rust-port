//! # femsolve
//!
//! Solver building blocks for finite-element eigenproblems, written once for real and complex
//! scalars: a thick-restart (block) Krylov–Schur eigensolver, shift-and-invert with symbolic
//! factorisation reuse, Matrix Market export of problems, and a harness to compare against
//! SLEPc. See `docs/STATUS.md` for what is done, what was measured and what is still open.

pub mod amg;
pub mod ams;
pub mod basis;
pub mod bsr;
pub mod bundle;
pub mod chebfsi;
pub mod contour;
pub mod csr;
pub mod dense;
pub mod direct;
pub mod eps;
pub mod expm;
pub mod faer_glue;
pub mod fieldsplit;
pub mod ilu;
pub mod jd;
pub mod kernels;
pub mod ksp;
pub mod linop;
pub mod lobpcg;
pub mod maxwell;
pub mod mixed;
pub mod mm;
pub mod nep;
pub mod opt;
pub mod ordering;
pub mod par;
pub mod pc;
pub mod problem;
pub mod rng;
pub mod scalar;
pub mod schwarz;
pub mod sens;
pub mod snes;
pub mod sparse_lu;
pub mod sweep;
pub mod toar;
pub mod ts;
pub mod vecops;

pub use csr::Csr;
pub use linop::{FnOperator, Identity, LinearOperator, Preconditioner};
pub use scalar::{C64, Scalar};
