//! `femmesh`: pure-Rust metric-based mesh adaptation, structured PML meshes and mixed-cell
//! meshes for finite-element solvers.
//!
//! Layout mirrors the planned `femcore::adapt` folder so it can be moved over with the stand-in
//! mesh types (`mesh`, `cellmesh`) replaced by nyx's own.

pub mod adapt;
pub mod cellmesh;
pub mod fixtures;
pub mod io;
pub mod mesh;
pub mod par;
pub mod quality;
pub mod timer;
