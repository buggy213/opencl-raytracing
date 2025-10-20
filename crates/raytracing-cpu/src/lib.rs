use accel::{traverse_bvh};
use lights::{light_radiance, occluded, sample_light};
use materials::CpuMaterial;
use ray::Ray;
use raytracing::{geometry::{Matrix4x4, Vec3}, scene::{Camera, Scene}};

use crate::accel::CPUTraversalContext;

mod ray;
mod accel;
mod geometry;
mod lights;
mod materials;
mod sample;
mod scene;

fn generate_ray(camera: &Camera, x: u32, y: u32) -> Ray {
    let x_disp = 0.0;
    let y_disp = 0.0;

    let raster_loc = Vec3((x as f32) + x_disp, (y as f32) + y_disp, 0.0);
    let camera_space_loc = camera.world_to_raster.apply_inverse_point(raster_loc);
    
    let ray_o = camera.camera_position.into();
    let ray_d = Vec3::normalized(camera_space_loc - ray_o);

    Ray {
        origin: ray_o,
        direction: ray_d,
        debug: false
    }
}

fn ray_radiance(
    ray: Ray, 
    traversal_context: CPUTraversalContext,

    raytracer_settings: &RaytracerSettings
) -> Vec3 {
    let mut ray = ray;
    let mut depth = 0;

    // initial bounce is "specular" (i.e. it needs to sample zero bounce radiance)
    let mut specular_bounce = true;
    let mut radiance = Vec3::zero();
    let mut path_weight = Vec3(1.0, 1.0, 1.0);

    let CPUTraversalContext {
        scene,
        acceleration_structures: _,
    } = traversal_context;

    loop {
        // respect near/far clip settings for primary ray
        let (t_min, t_max) = if depth == 0 {
            let camera = &scene.camera;
            (camera.near_clip, camera.far_clip)
        } else {
            (0.0, f32::INFINITY)
        };

        let hit_info = traverse_bvh(
            ray, 
            t_min,
            t_max,
            traversal_context, 
            false,
        );

        let hit = match hit_info {
            Some(hit) => hit,
            None => break
        };

        let add_zero_bounce = raytracer_settings.accumulate_bounces || raytracer_settings.max_ray_depth == depth;
        if specular_bounce && add_zero_bounce {
            if let Some(light_idx) = hit.light_idx {
                let light = &scene.lights[light_idx as usize];
                radiance += path_weight * light_radiance(light, hit.point);
            }
        }

        let material = &scene.materials[hit.material_idx as usize];
        let w2o = Matrix4x4::make_w2o(hit.normal);
        let o2w = w2o.transposed();
        let wo = w2o.apply_vector(-ray.direction);

        depth += 1;
        specular_bounce = material.is_delta_bsdf();
        if depth > raytracer_settings.max_ray_depth {
            break;
        }
        
        // direct illumination
        let add_direct_illumination = raytracer_settings.accumulate_bounces || raytracer_settings.max_ray_depth == depth;
        if !specular_bounce && add_direct_illumination {     
            let mut direct_illumination = Vec3::zero();

            for light in &scene.lights {
                for _ in 0..raytracer_settings.light_sample_count {
                    let light_sample = sample_light(light, scene, hit.point);
                    let occluded = occluded(traversal_context, light_sample);
                    if !occluded {
                        let wi = w2o.apply_vector(-light_sample.shadow_ray.direction); // shadow ray from light to hit point, we want other way
                        let bsdf_value = material.get_bsdf(wo, wi);
                        
                        let cos_theta = wi.z();

                        direct_illumination += bsdf_value * light_sample.radiance * f32::max(0.0, cos_theta) / light_sample.pdf; 
                    }
                }

                direct_illumination /= raytracer_settings.light_sample_count as f32;
            }

            radiance += path_weight * direct_illumination;
        }

        // indirect illumination
        let bsdf_sample = material.sample_bsdf(wo);
        let cos_theta = bsdf_sample.wi.z();
        path_weight *= bsdf_sample.bsdf * cos_theta / bsdf_sample.pdf;

        let world_dir = o2w.apply_vector(bsdf_sample.wi);
        let new_ray = Ray {
            origin: hit.point + world_dir * 0.0001,
            direction: world_dir,
            debug: ray.debug
        };

        ray = new_ray;
    }

    radiance
}

fn first_hit_normals(
    ray: Ray, 
    traversal_context: CPUTraversalContext
) -> Vec3 {
    let hit_info = traverse_bvh(
        ray, 
        0.0,
        f32::INFINITY,
        traversal_context, 
        false,
    );

    if let Some(hit_info) = hit_info {
        hit_info.normal
    }
    else {
        Vec3::zero()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RaytracerSettings {
    pub max_ray_depth: u32,
    pub light_sample_count: u32,
    pub samples_per_pixel: u32,
    pub accumulate_bounces: bool,

    pub debug_normals: bool,
}

pub fn render(scene: &Scene, raytracer_settings: RaytracerSettings) -> Vec<Vec3> {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    let camera = &scene.camera;

    // construct BVH using embree
    let cpu_acceleration_structures = scene::prepare_cpu_scene(scene);
    let cpu_traversal_context = accel::CPUTraversalContext::new(
        &scene,
        &cpu_acceleration_structures
    );
    
    let mut radiance_buffer: Vec<Vec3> = Vec::with_capacity(width * height);

    // enter main tracing loop
    for j in 0..height {
        for i in 0..width {
            let mut radiance = Vec3(0.0, 0.0, 0.0);
            
            if raytracer_settings.debug_normals {
                let ray = generate_ray(camera, i as u32, j as u32);
                radiance = first_hit_normals(ray, cpu_traversal_context);
            }
            else {
                for _ in 0..raytracer_settings.samples_per_pixel {
                    let ray = generate_ray(camera, i as u32, j as u32);
                    radiance += ray_radiance(
                        ray, 
                        cpu_traversal_context, 
                        &raytracer_settings
                    );
                }
                
                radiance /= raytracer_settings.samples_per_pixel as f32; 
            }
            
            radiance_buffer.push(radiance);
        }
    }

    radiance_buffer
}