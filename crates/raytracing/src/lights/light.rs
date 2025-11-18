use crate::{geometry::{Transform, Vec3}, scene::BasicPrimitiveIndex};

#[derive(Debug)]
#[repr(C)]
pub enum Light {
    PointLight {
        position: Vec3,
        intensity: Vec3
    },
    DiffuseAreaLight {
        prim_id: BasicPrimitiveIndex,
        radiance: Vec3,
        transform: Transform,
    }
}

impl Light {
    pub fn is_delta_light(&self) -> bool {
        match self {
            Light::PointLight { .. } => true,
            Light::DiffuseAreaLight { .. } => false,
        }
    }
}

impl Light {
    pub fn from_gltf_punctual_light(light: &gltf::Node) -> Light {
        let light_properties = light.light().unwrap();
        assert!(matches!(light_properties.kind(), gltf::khr_lights_punctual::Kind::Point)); // only point lights for now...
        let intensity: Vec3 = <[f32; 3] as Into<Vec3>>::into(light_properties.color()) * light_properties.intensity();
        let (light_position, _, _) = light.transform().decomposed(); // T * R * S

        Light::PointLight { position: light_position.into(), intensity }
    }

    pub fn from_emissive_geometry(prim_id: BasicPrimitiveIndex, radiance: Vec3, transform: Transform) -> Light {
        Light::DiffuseAreaLight { prim_id, radiance, transform }
    }
}