mod camera;
mod primitive;
mod scene;
mod pbrt;

pub mod test_scenes;

pub use camera::{Camera, CameraType};
pub use primitive::{
    AggregatePrimitive, AggregatePrimitiveIndex, BasicPrimitive, BasicPrimitiveIndex, Primitive,
    PrimitiveIndex, TransformPrimitive, TransformPrimitiveIndex,
};

pub use scene::gltf::scene_from_gltf_file;
pub use pbrt::scene_from_pbrt_file;

pub use scene::Scene;
pub(crate) use scene::SceneBuilder;
