use raytracing::{geometry::{Shape, Vec2, Vec3}, lights::Light, scene::Scene};
use crate::{ray::Ray, sample::sample_uniform2, traverse_bvh, BVHData};

#[derive(Clone, Copy, Debug)]
pub(crate) struct LightSample {
    pub(crate) radiance: Vec3,
    pub(crate) shadow_ray: Ray, // world-space
    pub(crate) distance: f32,   // world-space
    pub(crate) pdf: f32,
}

pub(crate) fn sample_light(
    light: &Light, 
    scene: &Scene, 
    point: Vec3 // world-space
) -> LightSample {
    match light {
        Light::PointLight { position, intensity } => {
            let dir = point - *position;
            let d = dir.length();
            let d2 = d * d;
            LightSample {
                radiance: *intensity / d2,
                shadow_ray: Ray { origin: *position, direction: dir / d, debug: false },
                distance: d,
                pdf: 1.0 // technically delta
            }
        },
        Light::DiffuseAreaLight { prim_id, radiance, transform } => {
            // have to sample a point on surface
            // for now, we do this in a pretty naive way
            // TODO: pbrt appears to have every triangle be a separate emitter
            // this might make implementing light sampling easier but still seems overkill? 
            let emitter = &scene.get_basic_primitive(*prim_id).shape;
            let emitter = match emitter {
                Shape::TriangleMesh(mesh) => mesh,
                Shape::Sphere { .. } => todo!(),
            };
            let mut pdf = 1.0;

            // uniformly pick triangle
            pdf /= emitter.tris.len() as f32;
            let random_tri_idx = rand::random_range(0..emitter.tris.len());
            
            // uniformly generate sample on triangle
            let sample = sample_uniform2();
            let bary = if sample.0 < sample.1 {
                let b0 = sample.0 / 2.0;
                let b1 = sample.1 - sample.0 / 2.0;
                let b2 = 1.0 - b0 - b1;
                Vec3(b0, b1, b2)
            }
            else {
                let b0 = sample.0 - sample.1 / 2.0;
                let b1 = sample.1 / 2.0;
                let b2 = 1.0 - b0 - b1;
                Vec3(b0, b1, b2)
            };

            pdf /= emitter.tri_area(random_tri_idx);
            
            let tri = emitter.tris[random_tri_idx];
            let p0 = emitter.vertices[tri.0 as usize];
            let p1 = emitter.vertices[tri.1 as usize];
            let p2 = emitter.vertices[tri.2 as usize];

            let p_local = bary.0 * p0 + bary.1 * p1 + bary.2 * p2;
            let p_world = transform.apply_point(p_local);
            let dir_world = point - p_world;
            let d = dir_world.length();
            let shadow_ray = Ray { origin: p_world, direction: dir_world / d, debug: false };
            
            // no backface emission
            let n0 = emitter.normals[tri.0 as usize];
            let n1 = emitter.normals[tri.1 as usize];
            let n2 = emitter.normals[tri.2 as usize];

            let n = bary.0 * n0 + bary.1 * n1 + bary.2 * n2;
            let n = n.unit();
            let radiance = if Vec3::dot(dir_world, n) < 0.0 {
                Vec3::zero()
            }
            else {
                *radiance
            };

            // use pdf to convert to solid angle integral
            pdf *= (d * d) / f32::abs(Vec3::dot(dir_world, n));

            LightSample {
                radiance,
                shadow_ray,
                distance: d,
                pdf,
            }
        }
    }
}

pub(crate) fn light_radiance(light: &Light, _hit_point: Vec3) -> Vec3 {
    match light {
        Light::PointLight { .. } => {
            Vec3::zero() // delta lights cannot be intersected
        },
        Light::DiffuseAreaLight { radiance, .. } => {
            *radiance
        },
    }
}

pub(crate) fn occluded(bvh: &BVHData<'_>, light_sample: LightSample) -> bool {
    traverse_bvh(
        light_sample.shadow_ray, 
        0.001, 
        light_sample.distance - 0.001, 
        bvh, 
        true,
    ).is_some()
}