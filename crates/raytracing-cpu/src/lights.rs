use std::f32;

use raytracing::{geometry::{Shape, Vec2, Vec3}, lights::{EnvironmentLight, Light}};
use crate::{CpuRaytracingContext, accel::TraversalCache, materials::MaterialEvalContext, ray::Ray, sample::CpuSampler, texture::CpuTextures, traverse_bvh};

#[derive(Clone, Copy, Debug)]
pub(crate) struct LightSample {
    pub(crate) radiance: Vec3,
    pub(crate) shadow_ray: Ray, // world-space
    pub(crate) distance: f32,   // world-space
    pub(crate) pdf: f32,
}

pub(crate) fn sample_light(
    context: &CpuRaytracingContext,
    light: &Light, 
    point: Vec3, // world-space
    sampler: &mut CpuSampler,
) -> LightSample {
    match light {
        Light::PointLight { position, intensity } => {
            let dir = point - *position;
            let d = dir.length();
            let d2 = d * d;
            LightSample {
                radiance: *intensity / d2,
                shadow_ray: Ray { origin: *position, direction: dir / d, },
                distance: d,
                pdf: 1.0 // contains delta
            }
        },
        Light::DirectionLight { direction, radiance } => {
            let scene_diameter = context.scene_bounds.radius * 2.0;
            let light_origin = point - (*direction) * scene_diameter;
            
            LightSample {
                radiance: *radiance,
                shadow_ray: Ray {
                    origin: light_origin,
                    direction: direction.unit(),
                },
                distance: scene_diameter,
                pdf: 1.0, // contains delta
            }
        },
        Light::DiffuseAreaLight { prim_id, radiance, transform } => {
            // have to sample a point on surface
            // for now, we do this in a pretty naive way
            // TODO: pbrt appears to have every triangle be a separate emitter
            // this might make implementing light sampling easier but still seems overkill? 
            let emitter = &context.scene.get_basic_primitive(*prim_id).shape;
            let emitter = match emitter {
                Shape::TriangleMesh(mesh) => mesh,
                Shape::Sphere { .. } => todo!(),
            };
            let mut pdf = 1.0;

            // uniformly pick triangle
            pdf /= emitter.tris.len() as f32;

            let all_tris = 0..emitter.tris.len() as u32;
            let random_tri_idx = sampler.sample_u32(all_tris) as usize;
            
            // uniformly generate sample on triangle
            let sample = sampler.sample_uniform2();
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
            let shadow_ray = Ray { origin: p_world, direction: dir_world / d, };
            
            // no backface emission
            let n = if emitter.normals.is_empty() {
                Vec3::cross(p1 - p0, p2 - p0).unit()
            }
            else {
                let n0 = emitter.normals[tri.0 as usize];
                let n1 = emitter.normals[tri.1 as usize];
                let n2 = emitter.normals[tri.2 as usize];
            
                Vec3::normalized(bary.0 * n0 + bary.1 * n1 + bary.2 * n2)
            };

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
        Light::PointLight { .. }
        | Light::DirectionLight { .. } => {
            // delta lights cannot be intersected
            Vec3::zero() 
        },
        Light::DiffuseAreaLight { radiance, .. } => {
            *radiance
        },
    }
}

pub(crate) fn environment_light_radiance(
    environment_light: &EnvironmentLight, 
    direction: Vec3,
    textures: &CpuTextures,
) -> Vec3 {
    // compute texture mapping function from world-space direction
    let direction = direction.unit();
    let st = match environment_light.mapping {
        raytracing::lights::TextureMapping::Spherical => {
            let t = f32::acos(direction.z()) * f32::consts::FRAC_1_PI;
            let s = (f32::atan2(direction.x(), direction.y()) + f32::consts::PI) * f32::consts::FRAC_1_PI * 0.5;
            
            Vec2(s, t)
        },
    };

    let eval_ctx = MaterialEvalContext::new_without_antialiasing(st);
    let sampled_radiance = textures.sample(environment_light.radiance, &eval_ctx);

    Vec3(sampled_radiance.0, sampled_radiance.1, sampled_radiance.2)
}

pub(crate) fn occluded(context: &CpuRaytracingContext, traversal_cache: &mut TraversalCache, light_sample: LightSample) -> bool {
    traverse_bvh(
        light_sample.shadow_ray, 
        0.001, 
        light_sample.distance - 0.001, 
        context,
        traversal_cache, 
        true,
    ).is_some()
}
