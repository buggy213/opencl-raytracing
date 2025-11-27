use std::sync::{Arc, Mutex};

use accel::{traverse_bvh};
use lights::{light_radiance, occluded, sample_light};
use materials::CpuMaterial;
use ray::Ray;
use raytracing::{geometry::{AABB, Matrix4x4, Vec3}, scene::{Camera, Scene}};
use tracing::warn;

use crate::{accel::TraversalCache, scene::CpuAccelerationStructures, texture::CpuTextures};

mod ray;
mod accel;
mod geometry;
mod lights;
mod materials;
mod sample;
pub use sample::set_seed;
mod scene;
mod texture;
pub mod utils;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub(crate) struct SceneBounds {
    bounding_box: AABB,
    center: Vec3,
    radius: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct CpuRaytracingContext<'scene> {
    pub(crate) scene: &'scene Scene,
    pub(crate) acceleration_structures: &'scene CpuAccelerationStructures,
    pub(crate) cpu_textures: CpuTextures<'scene>,
    
    // scene bounds not calculated for target-independent scene description
    // since it's easy to calculate it alongside the BVH (? maybe not for gpu backends)
    // but we need it for directional lights
    pub(crate) scene_bounds: SceneBounds,
}

impl<'scene> CpuRaytracingContext<'scene> {
    pub(crate) fn new(
        scene: &'scene Scene, 
        acceleration_structures: &'scene CpuAccelerationStructures
    ) -> CpuRaytracingContext<'scene> {
        let root_index = acceleration_structures.root_bvh_index();
        let root_bounding_box = acceleration_structures.bvhs[root_index].bounds();
        let root_bounds_center = root_bounding_box.center();
        let root_bounds_radius = root_bounding_box.radius();

        CpuRaytracingContext { 
            scene, 
            acceleration_structures,
            cpu_textures: CpuTextures::new(scene),

            scene_bounds: SceneBounds { 
                bounding_box: root_bounding_box, 
                center: root_bounds_center, 
                radius: root_bounds_radius 
            },
        }
    }
}

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
    context: &CpuRaytracingContext,
    traversal_cache: &mut TraversalCache,
    raytracer_settings: &RaytracerSettings
) -> Vec3 {
    let mut ray = ray;
    let mut depth = 0;

    // initial bounce is "specular" (i.e. it needs to sample zero bounce radiance)
    let mut specular_bounce = true;
    let mut radiance = Vec3::zero();
    let mut path_weight = Vec3(1.0, 1.0, 1.0);

    loop {
        // respect near/far clip settings for primary ray
        let (t_min, t_max) = if depth == 0 {
            let camera = &context.scene.camera;
            (camera.near_clip, camera.far_clip)
        } else {
            (0.0, f32::INFINITY)
        };

        let hit_info = traverse_bvh(
            ray, 
            t_min,
            t_max,
            context, 
            traversal_cache,
            false,
        );

        let hit = match hit_info {
            Some(hit) => hit,
            None => break
        };

        let add_zero_bounce = raytracer_settings.accumulate_bounces || raytracer_settings.max_ray_depth == depth;
        if specular_bounce && add_zero_bounce {
            if let Some(light_idx) = hit.light_idx {
                let light = &context.scene.lights[light_idx as usize];
                radiance += path_weight * light_radiance(light, hit.point);
            }
        }

        let material = &context.scene.materials[hit.material_idx as usize];
        let bsdf = material.get_bsdf(hit.uv, &context.cpu_textures);
        let w2o = Matrix4x4::make_w2o(hit.normal);
        let o2w = w2o.transposed();
        let wo = w2o.apply_vector(-ray.direction);

        depth += 1;
        let delta_bsdf = bsdf.is_delta_bsdf();
        if depth > raytracer_settings.max_ray_depth {
            break;
        }
        
        // direct illumination
        let add_direct_illumination = raytracer_settings.accumulate_bounces || raytracer_settings.max_ray_depth == depth;
        if !delta_bsdf && add_direct_illumination {     
            let mut direct_illumination = Vec3::zero();

            for light in &context.scene.lights {
                let light_samples = if light.is_delta_light() {
                    1
                }
                else {
                    raytracer_settings.light_sample_count
                };

                for _ in 0..light_samples {
                    let light_sample = sample_light(context, light, hit.point);
                    let occluded = occluded(context, traversal_cache, light_sample);
                    if !occluded {
                        let wi = w2o.apply_vector(-light_sample.shadow_ray.direction); // shadow ray from light to hit point, we want other way
                        let bsdf_value = bsdf.evaluate_bsdf(wo, wi);
                        
                        let cos_theta = wi.z();

                        direct_illumination += bsdf_value * light_sample.radiance * f32::max(0.0, cos_theta) / light_sample.pdf; 
                    }
                }

                direct_illumination /= light_samples as f32;
            }

            radiance += path_weight * direct_illumination;
        }

        // indirect illumination
        // wi and wo can be facing opposite the normal (e.g. reflection inside a glass sphere)
        // cos_theta terms need to use absolute value to account for this
        let bsdf_sample = bsdf.sample_bsdf(wo);
        let cos_theta = bsdf_sample.wi.z().abs();
        path_weight *= bsdf_sample.bsdf * cos_theta / bsdf_sample.pdf;

        specular_bounce = bsdf_sample.specular;
        
        // if we sampled a bsdf value of zero, terminate this ray
        // TODO: use russian roulette termination condition
        if bsdf_sample.bsdf == Vec3::zero() {
            break;
        }

        if cfg!(debug_assertions) && bsdf_sample.pdf == 0.0 {
            warn!("pdf of 0.0 encountered");
        }

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
    context: &CpuRaytracingContext,
    traversal_cache: &mut TraversalCache,
) -> Vec3 {
    let hit_info = traverse_bvh(
        ray, 
        0.0,
        f32::INFINITY,
        context, 
        traversal_cache,
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

    pub num_threads: u32,

    pub debug_normals: bool,
}

#[derive(Debug, Clone, Copy)]
struct RenderTile {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize
}

impl RenderTile {
    const fn width(&self) -> usize {
        self.x1 - self.x0
    }

    const fn height(&self) -> usize {
        self.y1 - self.y0
    }

    const fn size(&self) -> usize {
        self.width() * self.height()
    }
}

fn create_render_jobs(scene: &Scene) -> Vec<RenderTile> {
    const TILE_SIZE: usize = 64;
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    let tiles_x = width.div_ceil(TILE_SIZE);
    let tiles_y = height.div_ceil(TILE_SIZE);
    
    let mut render_jobs = Vec::with_capacity(tiles_x * tiles_y);
    for j in 0..tiles_y {
        for i in 0..tiles_x {
            let render_job = RenderTile {
                x0: i * TILE_SIZE,
                x1: usize::min(width, (i+1) * TILE_SIZE),
                y0: j * TILE_SIZE,
                y1: usize::min(height, (j+1) * TILE_SIZE),
            };

            render_jobs.push(render_job)
        }
    }

    render_jobs
}

fn render_tile(
    context: &CpuRaytracingContext,
    traversal_cache: &mut TraversalCache,
    tile: RenderTile, 
    raytracer_settings: &RaytracerSettings
) -> Vec<Vec3> {
    let mut tile_radiance_buffer = Vec::with_capacity(tile.size());
    let tile_width = tile.width();
    let tile_height = tile.height();

    for j in 0..tile_height {
        for i in 0..tile_width {
            let mut radiance = Vec3(0.0, 0.0, 0.0);
            
            if raytracer_settings.debug_normals {
                let ray = generate_ray(&context.scene.camera, (tile.x0 + i) as u32, (tile.y0 + j) as u32);
                radiance = first_hit_normals(
                    ray, 
                    context,
                    traversal_cache,
                );
            }
            else {
                for _ in 0..raytracer_settings.samples_per_pixel {
                    let ray = generate_ray(&context.scene.camera, (tile.x0 + i) as u32, (tile.y0 + j) as u32);

                    radiance += ray_radiance(
                        ray, 
                        context,
                        traversal_cache, 
                        raytracer_settings
                    );
                }
                
                radiance /= raytracer_settings.samples_per_pixel as f32; 
            }
            
            tile_radiance_buffer.push(radiance);
        }
    }

    tile_radiance_buffer
}

fn merge_tile(
    radiance_buffer_width: usize,
    radiance_buffer: &mut [Vec3], 
    tile: RenderTile,
    tile_radiance: Vec<Vec3>
) {
    for j in 0..tile.height() {
        for i in 0..tile.width() {
            let tile_idx = i + j * tile.width();
            let global_idx = (tile.x0 + i) + (tile.y0 + j) * radiance_buffer_width;
            if radiance_buffer[global_idx] != Vec3::zero() {
                panic!("overwriting stuff");
            }
            radiance_buffer[global_idx] = tile_radiance[tile_idx];
        }
    }
}

pub fn render(scene: &Scene, raytracer_settings: RaytracerSettings) -> Vec<Vec3> {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    // construct BVH using embree
    let cpu_acceleration_structures = scene::prepare_cpu_acceleration_structures(scene);
    let cpu_raytracing_context = CpuRaytracingContext::new(
        &scene,
        &cpu_acceleration_structures
    );

    let single_threaded = if raytracer_settings.debug_normals {
        // debug normals is very cheap, so just keep it single threaded
        true
    } else {
        raytracer_settings.num_threads == 1
    };

    let radiance = if single_threaded {
        let mut traversal_cache = TraversalCache::new(&cpu_raytracing_context);
        let full_screen = RenderTile {
            x0: 0,
            x1: width,
            y0: 0,
            y1: height,
        };

        render_tile(
            &cpu_raytracing_context,
            &mut traversal_cache,
            full_screen,
            &raytracer_settings
        )
    }
    else {
        // shared work queue that multiple threads pop from
        // and mpsc channel that they send results back on
        // main thread is responsible for aggregating results
        let render_jobs = create_render_jobs(scene);

        let mut render_job_count = render_jobs.len();
        let mut radiance_buffer = vec![Vec3::zero(); width * height];
        let work_queue = Arc::new(
            Mutex::new(render_jobs)
        );

        let (result_channel_tx, result_channel_rx) = 
            std::sync::mpsc::channel::<(RenderTile, Vec<Vec3>)>();

        fn render_thread(
            context: &CpuRaytracingContext,
            work_queue: Arc<Mutex<Vec<RenderTile>>>,
            result_channel_tx: std::sync::mpsc::Sender<(RenderTile, Vec<Vec3>)>,
            raytracer_settings: &RaytracerSettings
        ) {
            let mut traversal_cache = TraversalCache::new(context);

            loop {
                let mut work_queue_guard = work_queue.lock().expect("lock poisoned");
                let job = work_queue_guard.pop();
                drop(work_queue_guard);

                let Some(tile) = job else {
                    break
                };

                let tile_radiance = render_tile(
                    context, 
                    &mut traversal_cache, 
                    tile, 
                    raytracer_settings
                );

                result_channel_tx.send((tile, tile_radiance))
                    .expect("no receiver (is main thread gone?)");
            }
        }
        
        std::thread::scope(|scope| {
            for _ in 0..raytracer_settings.num_threads {
                let thread_work_queue_handle = Arc::clone(&work_queue);
                let thread_result_channel_tx = result_channel_tx.clone();
                scope.spawn(|| {
                    render_thread(
                        &cpu_raytracing_context, 
                        thread_work_queue_handle, 
                        thread_result_channel_tx, 
                        &raytracer_settings
                    );
                });
            }

            while render_job_count > 0 {
                let (tile, tile_radiance) = result_channel_rx.recv()
                    .expect("no sender (are workers gone?)");

                merge_tile(
                    width, 
                    &mut radiance_buffer, 
                    tile, 
                    tile_radiance
                );

                render_job_count -= 1;
            }
        });

        radiance_buffer
    };

    // TODO: maybe generalize this / make it parameterizable or something
    // check for NaNs and warn, even in release builds (should be quick)
    {
        let mut warns = 0;
        let mut idx = 0;

        let mut warn = |msg: String| {
            if warns < 10 {
                warn!(msg);    
            }
            warns += 1
        };

        for j in 0..scene.camera.raster_height {
            for i in 0..scene.camera.raster_width {
                let v = radiance[idx];
                idx += 1;

                match v.0.classify() {
                    std::num::FpCategory::Nan => warn(format!("R component of ({i}, {j}) is NaN")),
                    std::num::FpCategory::Infinite => warn(format!("R component of ({i}, {j}) is infty")),
                    _ => ()
                }

                match v.1.classify() {
                    std::num::FpCategory::Nan => warn(format!("G component of ({i}, {j}) is NaN")),
                    std::num::FpCategory::Infinite => warn(format!("G component of ({i}, {j}) is infty")),
                    _ => ()
                }

                match v.2.classify() {
                    std::num::FpCategory::Nan => warn(format!("B component of ({i}, {j}) is NaN")),
                    std::num::FpCategory::Infinite => warn(format!("B component of ({i}, {j}) is infty")),
                    _ => ()
                }
            }
        }

        if warns != 0 {
            warn!("encountered {warns} NaN and infty values in radiance buffer")
        }
    }

    radiance
}

pub fn render_single_pixel(
    scene: &Scene, 
    raytracer_settings: RaytracerSettings,
    x: u32,
    y: u32,
) -> Vec3 {
    let (x, y) = if x >= scene.camera.raster_width as u32 || y >= scene.camera.raster_height as u32 {
        let clamped_x = u32::clamp(x, 0, scene.camera.raster_width as u32 - 1);
        let clamped_y = u32::clamp(y, 0, scene.camera.raster_height as u32 - 1);
        warn!("out-of-bounds coordinates for `render_single_pixel`, clamping to x = {clamped_x}, y = {clamped_y}");

        (clamped_x, clamped_y)
    }
    else {
        (x, y)
    };

    // force single threaded, since it's only 1 pixel
    let single_threaded_settings = RaytracerSettings {
        num_threads: 1,
        ..raytracer_settings
    };

    let cpu_acceleration_structures = scene::prepare_cpu_acceleration_structures(scene);
    let cpu_raytracing_context = CpuRaytracingContext::new(
        &scene,
        &cpu_acceleration_structures
    );

    let mut traversal_cache = TraversalCache::new(&cpu_raytracing_context);
    let single_pixel = RenderTile {
        x0: x as usize,
        x1: x as usize + 1,
        y0: y as usize,
        y1: y as usize + 1,
    };

    let pixel = render_tile(
        &cpu_raytracing_context,
        &mut traversal_cache,
        single_pixel,
        &single_threaded_settings
    )[0];

    
    pixel
}