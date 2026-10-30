//! # rsolvers
//!
//! Rust ports of representative solvers from [PETSc](https://petsc.org) (`KSP`/`PC`) and
//! [SLEPc](https://slepc.upv.es) (`EPS`), both BSD-2-Clause. See `THIRD_PARTY_NOTICES.md`.
//!
//! * [`ksp::Cg`], [`ksp::Gmres`] — Krylov linear solvers with PETSc's defaults and convergence tests.
//! * [`pc`] — Jacobi, SSOR, ILU(0), IC(0) preconditioners.
//! * [`eps::krylov_schur`] — Krylov–Schur for symmetric eigenproblems (SLEPc's default method).
//!
//! All reductions are evaluated over fixed-size chunks in index order, so results are
//! bit-for-bit identical regardless of the number of threads (feature `parallel`).

pub mod basis;
pub mod dense;
pub mod eps;
pub mod kernels;
pub mod ksp;
pub mod operator;
pub mod par;
pub mod pc;
pub mod rng;
pub mod sparse;
pub mod vecops;

pub use operator::{FnOperator, Identity, LinearOperator, Preconditioner};
pub use sparse::CsrMatrix;
