use raytracing::{accel::bvh2::LinearizedBVHNode, geometry::Vec3, lights::Light, scene::Scene};

use crate::{ray::Ray, traverse_bvh, BVHData};

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
                shadow_ray: Ray { origin: *position, direction: dir / d, time: 0.0 },
                distance: d,
            }
        },
    }
}

pub(crate) fn occluded(bvh: &BVHData<'_>, light_sample: LightSample) -> bool {
    traverse_bvh(
        light_sample.shadow_ray, 
        0.001, 
        light_sample.distance - 0.001, 
        bvh, 
        true
    ).is_some()
}