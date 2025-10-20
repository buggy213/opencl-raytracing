mod scene;
mod camera;
mod primitive;

pub use scene::Scene;
pub use scene::test_scenes;
pub use primitive::{
    Primitive, BasicPrimitive, TransformPrimitive, AggregatePrimitive,
    PrimitiveIndex, BasicPrimitiveIndex, TransformPrimitiveIndex, AggregatePrimitiveIndex
};
pub use camera::Camera;
pub use camera::RenderTile;