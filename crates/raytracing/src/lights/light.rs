use std::{ptr::null_mut, cmp::max};
use crate::geometry::Vec3;

#[derive(Debug)]
#[repr(C)]
pub enum Light {
    PointLight {
        position: Vec3,
        intensity: Vec3
    }
}

impl Light {
    pub fn from_gltf_light(light: &gltf::Node) -> Light {
        let light_properties = light.light().unwrap();
        assert!(matches!(light_properties.kind(), gltf::khr_lights_punctual::Kind::Point)); // only point lights for now...
        let intensity: Vec3 = <[f32; 3] as Into<Vec3>>::into(light_properties.color()) * light_properties.intensity();
        let (light_position, _, _) = light.transform().decomposed(); // T * R * S

        Light::PointLight { position: light_position.into(), intensity }
    }
}