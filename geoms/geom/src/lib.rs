//! `geom`: a lean CSG-over-B-rep geometry kernel for the nyx FEM stack.
pub mod arrange;
pub mod complex;
pub mod curve;
pub mod error;
pub mod eval;
pub mod expr;
pub mod labels;
pub mod math;
pub mod model;
pub mod prim;
pub mod print;
pub mod scene;
pub mod select;
pub mod sim;
pub use error::{GeomError, GeomResult};
