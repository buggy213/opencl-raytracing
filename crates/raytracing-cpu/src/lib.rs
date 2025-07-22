use accel::{traverse_bvh, BVHData};
use lights::{light_radiance, occluded, sample_light};
use materials::CpuMaterial;
use ray::Ray;
use raytracing::{accel::{bvh2::LinearizedBVHNode, BVH2}, geometry::{Matrix4x4, Vec3}, scene::{Camera, Scene}};
use embree4::Device;

mod ray;
mod accel;
mod geometry;
mod lights;
mod materials;
mod sample;

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

fn ray_color(
    ray: Ray, 
    bvh: &BVHData, 
    scene: &Scene, 
    
    depth: u32,

    raytracer_settings: &RaytracerSettings
) -> Vec3 {
    if depth >= raytracer_settings.max_ray_depth && depth != 0 {
        return Vec3::zero();
    }

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
        bvh, 
        false,
    );

    if let Some(hit) = hit_info {
        // zero bounce illumination
        let zero_bounce = if depth != 0 {
            Vec3::zero() // was already counted by the previous bounce
        } else if let Some(light_idx) = hit.light_idx {
            let light = &scene.lights[light_idx as usize];
            light_radiance(light, hit.point)
        } else {
            Vec3::zero()
        };

        // direct illumination
        let mut direct_illumination = Vec3::zero();

        let material = &scene.materials[hit.material_idx as usize];
        let w2o = Matrix4x4::make_w2o(hit.normal);
        let o2w = w2o.transposed();
        let wo = w2o.apply_vector(-ray.direction);

        for light in &scene.lights {
            for _ in 0..raytracer_settings.light_sample_count {
                let light_sample = sample_light(light, scene, hit.point);
                let occluded = occluded(bvh, light_sample);
                if !occluded {
                    let wi = w2o.apply_vector(-light_sample.shadow_ray.direction); // shadow ray from light to hit point, we want other way
                    let bsdf_value = material.get_bsdf(wo, wi);
                    
                    let cos_theta = wi.z();

                    direct_illumination += bsdf_value * light_sample.radiance * f32::max(0.0, cos_theta) / light_sample.pdf; 
                }
            }

            direct_illumination /= raytracer_settings.light_sample_count as f32;
        }

        // indirect illumination
        let bsdf_sample = material.sample_bsdf(wo);
        let world_dir = o2w.apply_vector(bsdf_sample.wi);
        let secondary_ray = Ray {
            origin: hit.point + world_dir * 0.01,
            direction: world_dir,
            debug: ray.debug
        };

        let incident_radiance = ray_color(
            secondary_ray,
            bvh,
            scene,
            depth + 1,
            
            raytracer_settings
        );

        let cos_theta = bsdf_sample.wi.z();
        let indirect_illumination = bsdf_sample.bsdf * incident_radiance * cos_theta / bsdf_sample.pdf;
        
        if raytracer_settings.accumulate_bounces {
            if raytracer_settings.max_ray_depth == 0 {
                zero_bounce
            }
            else {
                zero_bounce + direct_illumination + indirect_illumination
            }
        }
        else {
            if depth == raytracer_settings.max_ray_depth {
                zero_bounce
            }
            else if depth + 1 == raytracer_settings.max_ray_depth {
                direct_illumination
            }
            else {
                indirect_illumination
            }
        }
    }
    else {
        Vec3::zero()
    }
}

fn first_hit_normals(
    ray: Ray, 
    bvh: &BVHData, 
    scene: &Scene
) -> Vec3 {
    let hit_info = traverse_bvh(
        ray, 
        0.0,
        f32::INFINITY,
        bvh, 
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

pub fn render(scene: &mut Scene, raytracer_settings: RaytracerSettings) -> Vec<Vec3> {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    let camera = &scene.camera;


    // construct BVH using embree
    let embree_device = Device::new();
    let mesh_bvh: BVH2 = BVH2::create(&embree_device, &scene.meshes);
    let (bvh_indices, linearized_bvh) = LinearizedBVHNode::linearize_bvh_mesh(&mesh_bvh, &scene.meshes);
    let bvh = BVHData {
        nodes: &linearized_bvh,
        meshes: &scene.meshes,
        indices: &bvh_indices,
    };

    let mut radiance_buffer: Vec<Vec3> = Vec::with_capacity(width * height);

    // enter main tracing loop
    for j in 0..height {
        for i in 0..width {
            let mut radiance = Vec3(0.0, 0.0, 0.0);
            
            if raytracer_settings.debug_normals {
                let ray = generate_ray(camera, i as u32, j as u32);
                radiance = first_hit_normals(ray, &bvh, scene);
            }
            else {
                for s in 0..raytracer_settings.samples_per_pixel {
                    let ray = generate_ray(camera, i as u32, j as u32);
                    radiance += ray_color(
                        ray, 
                        &bvh, 
                        &scene,  
                        0,
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