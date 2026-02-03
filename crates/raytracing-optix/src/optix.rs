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
    Quaternion,
    Transform,

    OptixDeviceContext,
    initOptix,
    destroyOptix,

    OptixAccelerationStructure,
    makeSphereAccelerationStructure,
    makeMeshAccelerationStructure,
    makeInstanceAccelerationStructure,

    OptixPipelineWrapper,
    makeAovPipeline,
    launchAovPipeline,

    Camera,
};

use detail::{
    CameraType,
    CameraTypeKind,
    CameraType_CameraVariant,
    CameraType_CameraVariant_Orthographic,
    CameraType_CameraVariant_PinholePerspective,
    CameraType_CameraVariant_ThinLensPerspective,
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

impl From<raytracing::geometry::Quaternion> for Quaternion {
    fn from(value: raytracing::geometry::Quaternion) -> Self {
        Quaternion { real: value.0, pure_: value.1.into() }
    }
}

impl From<Quaternion> for raytracing::geometry::Quaternion {
    fn from(value: Quaternion) -> Self {
        raytracing::geometry::Quaternion(value.real, value.pure_.into())
    }
}

impl From<raytracing::geometry::Transform> for Transform {
    fn from(value: raytracing::geometry::Transform) -> Self {
        Transform { forward: value.forward.into(), inverse: value.inverse.into() }
    }
}

impl From<Transform> for raytracing::geometry::Transform {
    fn from(value: Transform) -> Self {
        raytracing::geometry::Transform {
            forward: value.forward.into(),
            inverse: value.inverse.into(),
        }
    }
}

// conversion functions for scene description types. these should only be one way
impl From<raytracing::scene::CameraType> for CameraType {
    fn from(value: raytracing::scene::CameraType) -> Self {
        match value {
            raytracing::scene::CameraType::Orthographic { screen_space_width, screen_space_height } => {
                CameraType { 
                    kind: CameraTypeKind::Orthographic, 
                    variant: CameraType_CameraVariant { 
                        orthographic: CameraType_CameraVariant_Orthographic { screen_space_width, screen_space_height }
                    } 
                }
            },
            raytracing::scene::CameraType::PinholePerspective { yfov } => {
                CameraType { 
                    kind: CameraTypeKind::PinholePerspective, 
                    variant: CameraType_CameraVariant {
                        pinhole_perspective: CameraType_CameraVariant_PinholePerspective { yfov }
                    } 
                }
            },
            raytracing::scene::CameraType::ThinLensPerspective { yfov, aperture_radius, focal_distance } => {
                CameraType { 
                    kind: CameraTypeKind::ThinLensPerspective, 
                    variant: CameraType_CameraVariant {
                        thin_lens_perspective: CameraType_CameraVariant_ThinLensPerspective { yfov, aperture_radius, focal_distance }
                    } 
                }
            },
        }
    }
}

impl From<raytracing::scene::Camera> for Camera {
    fn from(value: raytracing::scene::Camera) -> Self {
        Camera { 
            camera_position: value.camera_position.into(), 
            camera_rotation: value.camera_rotation.into(), 
            camera_type: value.camera_type.into(), 
            raster_width: value.raster_width, 
            raster_height: value.raster_height, 
            near_clip: value.near_clip, 
            far_clip: value.far_clip, 
            world_to_raster: value.world_to_raster.into(), 
            camera_to_world: value.camera_to_world.into(), 
            raster_to_camera: value.raster_to_camera.into() 
        }
    }
}
