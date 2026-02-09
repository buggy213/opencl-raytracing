#[allow(clippy::all)]
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod detail {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub(crate) mod kernels {
    // we include the text of the kernels directly; it's more convenient than
    // doing it at runtime and doesn't require linker shenanigans

    pub(crate) const NORMALS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/normals.optixir"));
    pub(crate) const PATHTRACER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pathtracer.optixir"));
}

use std::{ops::Deref, slice};

pub(crate) use detail::{
    Vec2,
    Vec3,
    Vec3u,
    Vec4,
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

    AovPipelineWrapper,
    makeAovPipeline,
    launchAovPipeline,

    PathtracerPipelineWrapper,
    makePathtracerPipeline,
    launchPathtracerPipeline,

    GeometryData,
    GeometryKind,

    AovSbtWrapper,
    makeAovSbt,
    addHitRecordAovSbt,
    finalizeAovSbt,
    releaseAovSbt,

    PathtracerSbtWrapper,
    makePathtracerSbt,
    addHitRecordPathtracerSbt,
    finalizePathtracerSbt,
    releasePathtracerSbt,

    CudaArray,
    CudaTextureObject,
    TextureFormat,
    makeCudaArray,
    makeCudaTexture,

    Camera,
    Light,
    Texture,
    TextureSampler,
    Scene,
    Material,
};

use detail::{
    CameraType,
    CameraTypeKind,
    CameraType_CameraVariant,
    CameraType_CameraVariant_Orthographic,
    CameraType_CameraVariant_PinholePerspective,
    CameraType_CameraVariant_ThinLensPerspective,

    Light_LightKind,
    Light_LightVariant,
    Light_LightVariant_PointLight,
    Light_LightVariant_DirectionLight,
    Light_LightVariant_DiffuseAreaLight,
};

// crate::scene needs to use these, so we export them
pub(crate) use detail::{
    Texture_TextureKind,
    Texture_TextureVariant,
    Texture_TextureVariant_ImageTexture,
    Texture_TextureVariant_ConstantTexture,
    Texture_TextureVariant_CheckerTexture,
    Texture_TextureVariant_MixTexture,
    Texture_TextureVariant_ScaleTexture,

    Material_MaterialKind,
    Material_MaterialVariant,
    Material_MaterialVariant_Diffuse,
};

use detail::{
    WrapMode,
    FilterMode
};


// conversion functions for vocabulary types
impl From<raytracing::geometry::Vec4> for Vec4 {
    fn from(value: raytracing::geometry::Vec4) -> Self {
        Vec4 { x: value.0, y: value.1, z: value.2, w: value.3 }
    }
}


impl From<Vec4> for raytracing::geometry::Vec4 {
    fn from(value: Vec4) -> Self {
        raytracing::geometry::Vec4(value.x, value.y, value.z, value.w)
    }
}

const _: () = {
    // ensure that layouts match
    assert!(std::mem::size_of::<raytracing::geometry::Vec4>() == std::mem::size_of::<Vec4>());
    assert!(std::mem::align_of::<raytracing::geometry::Vec4>() == std::mem::align_of::<Vec4>());
};



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

pub(crate) struct Vec3SliceWrap<'s>(&'s [Vec3]);
impl<'s> From<&'s [raytracing::geometry::Vec3]> for Vec3SliceWrap<'s> {
    fn from(value: &'s [raytracing::geometry::Vec3]) -> Vec3SliceWrap<'s> {
        // SAFETY: rust's own allocated memory surely meets safety criteria for `from_raw_parts`, and layout check above
        // should ensure pointer cast is safe
        Vec3SliceWrap(
            unsafe { slice::from_raw_parts(value.as_ptr() as *const Vec3, value.len()) }
        )
    }
}

impl Deref for Vec3SliceWrap<'_> {
    type Target = [Vec3];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

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

pub(crate) struct Vec3uSliceWrap<'s>(&'s [Vec3u]);
impl<'s> From<&'s [raytracing::geometry::Vec3u]> for Vec3uSliceWrap<'s> {
    fn from(value: &'s [raytracing::geometry::Vec3u]) -> Vec3uSliceWrap<'s> {
        // SAFETY: rust's own allocated memory surely meets safety criteria for `from_raw_parts`, and layout check above
        // should ensure pointer cast is safe
        Vec3uSliceWrap(
            unsafe { slice::from_raw_parts(value.as_ptr() as *const Vec3u, value.len()) }
        )
    }
}

impl Deref for Vec3uSliceWrap<'_> {
    type Target = [Vec3u];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl From<raytracing::geometry::Vec2> for Vec2 {
    fn from(value: raytracing::geometry::Vec2) -> Self {
        Vec2 { x: value.0, y: value.1 }
    }
}

impl From<Vec2> for raytracing::geometry::Vec2 {
    fn from(value: Vec2) -> Self {
        raytracing::geometry::Vec2(value.x, value.y)
    }
}

const _: () = {
    // ensure that layouts match
    assert!(std::mem::size_of::<raytracing::geometry::Vec2>() == std::mem::size_of::<Vec2>());
    assert!(std::mem::align_of::<raytracing::geometry::Vec2>() == std::mem::align_of::<Vec2>());
};

pub(crate) struct Vec2SliceWrap<'s>(&'s [Vec2]);
impl<'s> From<&'s [raytracing::geometry::Vec2]> for Vec2SliceWrap<'s> {
    fn from(value: &'s [raytracing::geometry::Vec2]) -> Vec2SliceWrap<'s> {
        // SAFETY: rust's own allocated memory surely meets safety criteria for `from_raw_parts`, and layout check above
        // should ensure pointer cast is safe
        Vec2SliceWrap(
            unsafe { std::slice::from_raw_parts(value.as_ptr() as *const Vec2, value.len()) }
        )
    }
}

impl std::ops::Deref for Vec2SliceWrap<'_> {
    type Target = [Vec2];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

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

impl From<raytracing::lights::Light> for Light {
    fn from(value: raytracing::lights::Light) -> Self {
        match value {
            raytracing::lights::Light::PointLight { position, intensity } => {
                Light { 
                    kind: Light_LightKind::PointLight, 
                    variant: Light_LightVariant { 
                        point_light: Light_LightVariant_PointLight { position: position.into(), intensity: intensity.into() } 
                    } 
                }
            },
            raytracing::lights::Light::DirectionLight { direction, radiance } => {
                Light {
                    kind: Light_LightKind::DirectionLight,
                    variant: Light_LightVariant {
                        direction_light: Light_LightVariant_DirectionLight {
                            direction: direction.into(),
                            radiance: radiance.into(),
                        }
                    }
                }
            },
            raytracing::lights::Light::DiffuseAreaLight { prim_id: _, radiance, light_to_world } => {
                Light {
                    kind: Light_LightKind::DiffuseAreaLight,
                    variant: Light_LightVariant {
                        area_light: Light_LightVariant_DiffuseAreaLight {
                            prim_id: 0, // TODO: support area light sampling
                            radiance: radiance.into(),
                            light_to_world: light_to_world.into(),
                        }
                    }
                }
            },
        }
    }
}

impl From<raytracing::materials::WrapMode> for WrapMode {
    fn from(value: raytracing::materials::WrapMode) -> Self {
        match value {
            raytracing::materials::WrapMode::Repeat => WrapMode::Repeat,
            raytracing::materials::WrapMode::Mirror => WrapMode::Mirror,
            raytracing::materials::WrapMode::Clamp => WrapMode::Clamp,
        }
    }
}

impl From<raytracing::materials::FilterMode> for FilterMode {
    fn from(value: raytracing::materials::FilterMode) -> Self {
        match value {
            raytracing::materials::FilterMode::Nearest => FilterMode::Nearest,
            raytracing::materials::FilterMode::Bilinear => FilterMode::Bilinear,
            raytracing::materials::FilterMode::Trilinear => FilterMode::Trilinear,
        }
    }
}

impl From<raytracing::materials::TextureSampler> for TextureSampler {
    fn from(value: raytracing::materials::TextureSampler) -> Self {
        TextureSampler { filter: value.filter.into(), wrap: value.wrap.into() }
    }
}
