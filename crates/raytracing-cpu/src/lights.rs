use raytracing::{accel::bvh2::LinearizedBVHNode, geometry::Vec3, lights::Light, scene::Scene};

use crate::{ray::Ray, traverse_bvh};

#[derive(Clone, Copy, Debug)]
pub(crate) struct LightSample {
    pub(crate) radiance: Vec3,
    pub(crate) shadow_ray: Ray,
    pub(crate) distance: f32
}

pub(crate) fn sample_light(light: &Light, point: Vec3) -> LightSample {
    match light {
        Light::PointLight { position, intensity } => {
            let dir = point - *position;
            let d = dir.length();
            let d2 = d * d;
            LightSample {
                radiance: *intensity / d2,
                shadow_ray: Ray { origin: *position, direction: dir, time: 0.0 },
                distance: d,
            }
        },
    }
}

pub(crate) fn occluded(bvh: &[LinearizedBVHNode], scene: &Scene, light_sample: LightSample) -> bool {
    traverse_bvh(
        light_sample.shadow_ray, 
        f32::EPSILON, 
        light_sample.distance - f32::EPSILON, 
        bvh, 
        &scene.mesh.0, 
        true
    ).is_some()
}