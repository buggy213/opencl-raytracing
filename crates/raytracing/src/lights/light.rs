use tracing::warn;

use crate::{
    geometry::{Matrix4x4, Quaternion, Vec3}, materials::TextureId, scene::BasicPrimitiveIndex
};

#[derive(Debug)]
#[repr(C)]
pub enum Light {
    PointLight {
        position: Vec3,
        intensity: Vec3,
    },
    DirectionLight {
        // oriented *towards* direction that radiant energy is flowing
        direction: Vec3,

        // irradiance, implicitly multiplied by δ(direction), gives units of radiance
        radiance: Vec3,
    },
    DiffuseAreaLight {
        prim_id: BasicPrimitiveIndex,

        // uniform directional distribution
        radiance: Vec3,
        light_to_world: Matrix4x4,
    },
}

impl Light {
    pub fn is_delta_light(&self) -> bool {
        match self {
            Light::PointLight { .. } => true,
            Light::DirectionLight { .. } => true,
            Light::DiffuseAreaLight { .. } => false,
        }
    }
}

impl Light {
    pub fn from_gltf_punctual_light(
        light_node: &gltf::Node,
        light: &gltf::khr_lights_punctual::Light,
    ) -> Option<Light> {
        if light.range().is_some() {
            warn!("`range` property of light not supported");
        }

        match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => {
                let color: Vec3 = light.color().into();
                let radiance = color * light.intensity();

                // from GLTF spec: "untransformed light points down the -Z axis"
                let (_, rotation, _) = light_node.transform().decomposed();
                let rotation: Quaternion =
                    Quaternion(rotation[3], Vec3(rotation[0], rotation[1], rotation[2]));

                let direction = rotation.rotate(Vec3(0.0, 0.0, -1.0));

                Some(Light::DirectionLight {
                    direction,
                    radiance,
                })
            }
            gltf::khr_lights_punctual::Kind::Point => {
                let color: Vec3 = light.color().into();
                let intensity = color * light.intensity();
                let (position, _, _) = light_node.transform().decomposed();

                Some(Light::PointLight {
                    position: position.into(),
                    intensity,
                })
            }
            gltf::khr_lights_punctual::Kind::Spot { .. } => {
                warn!("gltf spot light not implemented");

                None
            }
        }
    }

    pub fn from_emissive_geometry(
        prim_id: BasicPrimitiveIndex,
        radiance: Vec3,
        light_to_world: Matrix4x4,
    ) -> Light {
        Light::DiffuseAreaLight {
            prim_id,
            radiance,
            light_to_world,
        }
    }
}

// environment lighting only done by rays which miss; easiest to implement
// TODO: importance sampling the environment light too
// TODO: different texture mapping functions for general textures too (need this to support pbrt scenes)
#[derive(Debug, Clone, Copy)]
pub enum TextureMapping {
    Spherical
}

#[derive(Debug)]
pub struct EnvironmentLight {
    pub mapping: TextureMapping,
    pub radiance: TextureId,
}
