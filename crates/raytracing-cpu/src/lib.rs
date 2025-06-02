use accel::{traverse_bvh, BVHData};
use lights::{occluded, sample_light, LightSample};
use materials::CpuMaterial;
use ray::Ray;
use raytracing::{accel::{bvh2::LinearizedBVHNode, BVH2}, geometry::{Matrix4x4, Vec3}, lights::Light, scene::{Camera, Scene}};
use embree4::Device;

mod ray;
mod accel;
mod geometry;
mod lights;
mod materials;

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
        time: 0.0,
    }
}

fn ray_color(ray: Ray, bvh: &BVHData<'_>, scene: &Scene) -> Vec3 {
    let camera = &scene.camera;

    let hit_info = traverse_bvh(
        ray, 
        camera.near_clip, 
        camera.far_clip, 
        bvh, 
        false
    );

    if let Some(hit) = hit_info {
        // direct illumination
        let mut direct_illumination = Vec3::zero();

        for light in &scene.lights {
            let light_sample = sample_light(light, hit.point);
            let occluded = occluded(bvh, light_sample);
            if !occluded {
                let material = &scene.materials[hit.material_idx as usize];
                
                let o2w = Matrix4x4::make_w2o(hit.normal);
                let wo = o2w.apply_vector(-ray.direction);
                let wi = o2w.apply_vector(-light_sample.shadow_ray.direction); // shadow ray from light to hit point, we want other way
                let bsdf_value = material.get_bsdf(wo, wi);
                
                let cos_theta = Vec3::dot(hit.normal, -light_sample.shadow_ray.direction);
                
                // no backface culling for now - blender gltf export seems to flip sometimes
                direct_illumination += bsdf_value * light_sample.radiance * f32::abs(cos_theta); 
            }
        }

        direct_illumination
    }
    else {
        Vec3::zero()
    }
}

pub fn render(scene: &mut Scene, spp: u32) -> Vec<Vec3> {
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
            for s in 0..spp {
                let ray = generate_ray(camera, i as u32, j as u32);
                radiance += ray_color(ray, &bvh, &scene);
            }

            radiance /= spp as f32;
            radiance_buffer.push(radiance);
        }
    }

    radiance_buffer
}