use crate::geometry::Vec3;

#[repr(C)]
pub enum Light {
    PointLight {
        position: Vec3,
        intensity: Vec3
    }
}

impl Light {
    pub fn from_gltf_light(light: &gltf::khr_lights_punctual::Light) -> Light {
        light.intensity();
        todo!();
    }
}