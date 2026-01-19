#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
mod detail {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub(crate) mod kernels {
    // we include the text of the kernels directly; it's more convenient than
    // doing it at runtime and doesn't require linker shenanigans

    pub(crate) const NORMALS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/normals.optixir"));
}

pub(crate) use detail::{
    Vec3,
    Vec3u,
    Matrix4x4,

    OptixDeviceContext,
    initOptix,
    destroyOptix,

    OptixAccelerationStructure,
    makeSphereAccelerationStructure,
    makeMeshAccelerationStructure,
    makeInstanceAccelerationStructure,

    OptixPipelineWrapper,
    makeBasicPipeline,
    launchBasicPipeline,
};

// conversion functions for vocabulary types
impl From<raytracing::geometry::Vec3> for Vec3 {
    fn from(value: raytracing::geometry::Vec3) -> Self {
        Vec3 { x: value.0, y: value.1, z: value.2 }
    }
}

impl From<Vec3> for raytracing::geometry::Vec3 {
    fn from(value: Vec3) -> Self {
        raytracing::geometry::Vec3(value.x, value.y, value.z)
    }
}

const _: () = {
    // ensure that layouts match
    assert!(std::mem::size_of::<raytracing::geometry::Vec3>() == std::mem::size_of::<Vec3>());
    assert!(std::mem::align_of::<raytracing::geometry::Vec3>() == std::mem::align_of::<Vec3>());
};

impl From<raytracing::geometry::Vec3u> for Vec3u {
    fn from(value: raytracing::geometry::Vec3u) -> Self {
        Vec3u { x: value.0, y: value.1, z: value.2 }
    }
}

impl From<Vec3u> for raytracing::geometry::Vec3u {
    fn from(value: Vec3u) -> Self {
        raytracing::geometry::Vec3u(value.x, value.y, value.z)
    }
}

const _: () = {
    // ensure that layouts match
    assert!(std::mem::size_of::<raytracing::geometry::Vec3u>() == std::mem::size_of::<Vec3u>());
    assert!(std::mem::align_of::<raytracing::geometry::Vec3u>() == std::mem::align_of::<Vec3u>());
};

impl From<raytracing::geometry::Matrix4x4> for Matrix4x4 {
    fn from(value: raytracing::geometry::Matrix4x4) -> Self {
        Matrix4x4 {
            m: [
                value.data[0][0], value.data[0][1], value.data[0][2], value.data[0][3],
                value.data[1][0], value.data[1][1], value.data[1][2], value.data[1][3],
                value.data[2][0], value.data[2][1], value.data[2][2], value.data[2][3],
                value.data[3][0], value.data[3][1], value.data[3][2], value.data[3][3],
            ]
        }
    }
}

impl From<Matrix4x4> for raytracing::geometry::Matrix4x4 {
    fn from(value: Matrix4x4) -> Self {
        raytracing::geometry::Matrix4x4 {
            data: [
                [value.m[0], value.m[1], value.m[2], value.m[3]],
                [value.m[4], value.m[5], value.m[6], value.m[7]],
                [value.m[8], value.m[9], value.m[10], value.m[11]],
                [value.m[12], value.m[13], value.m[14], value.m[15]],
            ]
        }
    }
}