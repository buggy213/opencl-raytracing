//! Acceleration structure
//! Will be used for OpenCL and CPU backends, since neither has support for hardware-accelerated
//! raytracing queries.

pub mod bvh2;

pub use bvh2::BVH2;
