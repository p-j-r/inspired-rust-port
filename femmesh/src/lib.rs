//! `femmesh`: pure-Rust metric-based mesh adaptation, structured PML meshes and mixed-cell
//! meshes for finite-element solvers.
//!
//! Layout mirrors the planned `femcore::adapt` folder so it can be moved over with the stand-in
//! mesh types (`mesh`, `cellmesh`) replaced by nyx's own.

// Index loops over the spatial dimension are the clearest form for small fixed-size matrix code, and
// `!(x > 0.0)` is deliberate: it also rejects NaN.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

pub mod adapt;
pub mod cellmesh;
pub mod fixtures;
pub mod io;
pub mod mesh;
pub mod par;
pub mod quality;
pub mod timer;
pub mod validate;
