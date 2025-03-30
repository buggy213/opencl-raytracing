pub mod device;
pub use device::Device;

pub mod bvh;
pub use bvh::BVH;
use embree4_sys::{RTCBounds, RTCBuildPrimitive};


pub type Bounds = RTCBounds;
pub type BuildPrimitive = RTCBuildPrimitive;