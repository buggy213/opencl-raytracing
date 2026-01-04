// used for #[rustfmt::skip]. it's still "unstable"
// due to strangeness around parsing, but as long as you don't abuse it
// it's basically fine
#![feature(stmt_expr_attributes)]

pub mod accel;
pub mod geometry;
pub mod lights;
pub mod macros;
pub mod materials;
pub mod sampling;
pub mod scene;

pub mod renderer;
