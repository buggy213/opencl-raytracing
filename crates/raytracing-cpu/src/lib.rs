#![feature(float_erf)]

use std::sync::{Arc, Mutex};

use accel::{traverse_bvh};
use lights::{light_radiance, occluded, sample_light};
use materials::CpuMaterial;
use ray::Ray;
use raytracing::{geometry::{AABB, Matrix4x4, Vec2, Vec3}, renderer::{AOVFlags, RaytracerSettings, RenderOutput}, scene::{Camera, Scene}};
use tracing::{info, warn};

use crate::{accel::TraversalCache, lights::environment_light_radiance, materials::MaterialEvalContext, ray::RayDifferentials, sample::CpuSampler, scene::CpuAccelerationStructures, texture::CpuTextures};

mod ray;
mod accel;
mod geometry;
mod lights;
mod materials;
mod sample;
mod scene;
mod texture;
pub mod utils;

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

    // maybe this should be calculated as part of target-independent scene description...
    pub(crate) min_differentials: RayDifferentials,
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

        let min_differentials = minimum_differentials(&scene.camera);

        CpuRaytracingContext { 
            scene, 
            acceleration_structures,
            cpu_textures: CpuTextures::new(scene),

            scene_bounds: SceneBounds { 
                bounding_box: root_bounding_box, 
                center: root_bounds_center, 
                radius: root_bounds_radius 
            },

            min_differentials
        }
    }
}

// finds the minimum ray differentials across all pixels for a given camera
// for pinhole perspective camera + orthographic camera, can actually get an analytic result
// once we support realistic cameras, this is no longer sufficient
fn minimum_differentials(camera: &Camera) -> RayDifferentials {
    match camera.camera_type {
        raytracing::scene::CameraType::Orthographic { .. } => {
            // for orthographic cameras, all rays are parallel and their origins are equally spaced
            let origin = camera.world_to_raster.apply_inverse_point(Vec3::zero());
            let dx = camera.world_to_raster.apply_inverse_point(Vec3(1.0, 0.0, 0.0));
            let dy = camera.world_to_raster.apply_inverse_point(Vec3(0.0, 1.0, 0.0));
            RayDifferentials {
                x_origin: dx - origin,
                y_origin: dy - origin,
                x_direction: Vec3::zero(),
                y_direction: Vec3::zero()
            }
        },
        raytracing::scene::CameraType::PinholePerspective { .. }
        | raytracing::scene::CameraType::ThinLensPerspective { .. } => {
            // for perspective cameras, all rays originate from the pinhole (or lens center),
            // and "direction" differential is guaranteed to be constant if primary
            // camera rays were unnormalized
            let center_x = (camera.raster_width as f32) / 2.0;
            let center_y = (camera.raster_height as f32) / 2.0;
            let center = camera.world_to_raster.apply_inverse_point(Vec3(center_x, center_y, 0.0));
            let dx = camera.world_to_raster.apply_inverse_point(Vec3(center_x + 1.0, center_y, 0.0));
            let dy = camera.world_to_raster.apply_inverse_point(Vec3(center_x, center_y + 1.0, 0.0));
            RayDifferentials {
                x_origin: Vec3::zero(),
                y_origin: Vec3::zero(),
                x_direction: dx - center,
                y_direction: dy - center,
            }
        },
    }
}

fn camera_ray(
    camera: &Camera,
    x: f32,
    y: f32,
    lens_sample: Option<Vec2>,
) -> Ray {
    let raster_loc = Vec3(x, y, 0.0);

    match camera.camera_type {
        raytracing::scene::CameraType::Orthographic { .. } => {
            let camera_space_o = camera.raster_to_camera.apply_point(raster_loc);
            // camera points down +z in it's local coordinate frame
            let camera_space_d = Vec3(0.0, 0.0, 1.0);

            let ray_o = camera.camera_to_world.apply_point(camera_space_o);
            let ray_d = camera.camera_to_world.apply_vector(camera_space_d).unit();

            Ray { origin: ray_o, direction: ray_d }
        },
        raytracing::scene::CameraType::PinholePerspective { .. } => {
            let cam_point = camera.raster_to_camera.apply_point(raster_loc);
            let cam_ray_dir = cam_point.unit();

            let world_origin = camera.camera_to_world.apply_point(Vec3::zero());
            let world_dir = camera.camera_to_world.apply_vector(cam_ray_dir).unit();

            Ray { origin: world_origin, direction: world_dir }
        },
        raytracing::scene::CameraType::ThinLensPerspective { aperture_radius, focal_distance, .. } => {
            let cam_point = camera.raster_to_camera.apply_point(raster_loc);

            let t = focal_distance / cam_point.z();
            let focus_point = cam_point * t;

            // Lens sample for depth of field
            let (cam_origin, cam_dir) = if let Some(Vec2(lens_u, lens_v)) = lens_sample {
                let lens_origin = Vec3(lens_u * aperture_radius, lens_v * aperture_radius, 0.0);
                let dir = (focus_point - lens_origin).unit();
                (lens_origin, dir)
            } else {
                // No lens sample - behave like pinhole
                (Vec3::zero(), cam_point.unit())
            };

            let world_origin = camera.camera_to_world.apply_point(cam_origin);
            let world_dir = camera.camera_to_world.apply_vector(cam_dir).unit();

            Ray { origin: world_origin, direction: world_dir }
        },
    }
}



fn generate_ray(
    camera: &Camera,
    x: u32,
    y: u32,
    sampler: &mut CpuSampler,
    spp: u32,
) -> (Ray, RayDifferentials) {
    let x_disp = sampler.sample_uniform();
    let y_disp = sampler.sample_uniform();
    let x = (x as f32) + x_disp;
    let y = (y as f32) + y_disp;

    // For thin-lens camera, sample the disk
    let lens_sample = match camera.camera_type {
        raytracing::scene::CameraType::ThinLensPerspective { .. } => {
            let u = sampler.sample_uniform2();
            Some(sample::sample_unit_disk_concentric(u))
        },
        _ => None,
    };

    let ray = camera_ray(camera, x, y, lens_sample);

    let ray_x = camera_ray(camera, x + 1.0, y, lens_sample);
    let ray_y = camera_ray(camera, x, y + 1.0, lens_sample);

    // we want to scale ray differentials to account for the supersampling that is already done when spp > 1
    // this makes the "effective" texture footprint smaller, though only up to a point
    // note that i don't think this is exactly what pbrt does since pbrt doesn't require
    // primary camera rays to be normalized, but it should be similar enough
    // TODO: do we need primary camera rays to be normalized???
    let scale = f32::max(0.125, f32::sqrt(1.0 / spp as f32));
    let scaled_x = ray.direction + (ray_x.direction - ray.direction) * scale;
    let scaled_y = ray.direction + (ray_y.direction - ray.direction) * scale;

    let ray_differentials = RayDifferentials {
        x_origin: ray_x.origin - ray.origin,
        y_origin: ray_y.origin - ray.origin,
        x_direction: scaled_x.unit() - ray.direction,
        y_direction: scaled_y.unit() - ray.direction,
    };

    (ray, ray_differentials)
}

fn ray_radiance(
    camera_ray: Ray, 
    camera_ray_differentials: RayDifferentials,
    context: &CpuRaytracingContext,
    sampler: &mut CpuSampler,
    traversal_cache: &mut TraversalCache,
    raytracer_settings: &RaytracerSettings
) -> Vec3 {
    let mut ray = camera_ray;
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
            None => {
                // unconditionally add environment lighting, as direct lighting can't sample it (yet)
                if let Some(environment_light) = &context.scene.environment_light {
                    radiance += path_weight * environment_light_radiance(environment_light, ray.direction, &context.cpu_textures);
                }

                break;
            }
        };

        let add_zero_bounce = raytracer_settings.accumulate_bounces || raytracer_settings.max_ray_depth == depth;
        if specular_bounce && add_zero_bounce {
            if let Some(light_idx) = hit.light_idx {
                let light = &context.scene.lights[light_idx as usize];
                radiance += path_weight * light_radiance(light, hit.point);
            }
        }

        let material = &context.scene.materials[hit.material_idx as usize];
        let material_eval_ctx = if depth == 0 && raytracer_settings.antialias_primary_rays {
            MaterialEvalContext::new_from_ray_differentials(&hit, ray, camera_ray_differentials)
        } else if depth > 0 && raytracer_settings.antialias_secondary_rays {
            // TODO: implement secondary ray antialiasing
            MaterialEvalContext::new_without_antialiasing(hit.uv)
        } else {
            MaterialEvalContext::new_without_antialiasing(hit.uv)
        };

        let bsdf = material.get_bsdf(&material_eval_ctx, &context.cpu_textures);
        let o2w = {
            let (x, y) = geometry::make_orthonormal_basis(hit.normal);
            Matrix4x4::create_from_basis(x, y, hit.normal)
        };
        let w2o = o2w.transposed();
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
                    let light_sample = sample_light(context, light, hit.point, sampler);
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
        let bsdf_sample = bsdf.sample_bsdf(wo, sampler);
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
        };

        ray = new_ray;
    }

    radiance
}

struct FirstHitAOVData {
    hit: bool,
    uv: Vec2,
    normals: Vec3,
    mip_level: Option<f32>,
}

fn first_hit_aovs(
    camera_ray: Ray,
    camera_ray_differentials: RayDifferentials, 
    context: &CpuRaytracingContext,
    traversal_cache: &mut TraversalCache,
) -> FirstHitAOVData {
    let hit_info = traverse_bvh(
        camera_ray, 
        0.0,
        f32::INFINITY,
        context, 
        traversal_cache,
        false,
    );

    let Some(hit) = hit_info else {
        return FirstHitAOVData { 
            hit: false, 
            uv: Vec2::zero(), 
            normals: Vec3::zero(),
            mip_level: None
        };
    };

    // evaluate material mip-level aov
    let material = &context.scene.materials[hit.material_idx as usize];
    let material_eval_ctx = 
        MaterialEvalContext::new_from_ray_differentials(&hit, camera_ray, camera_ray_differentials);
    let mip_level = material.get_mip_level(&material_eval_ctx, &context.cpu_textures);

    FirstHitAOVData { 
        hit: true, 
        uv: hit.uv, 
        normals: hit.normal, 
        mip_level,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuBackendSettings {
    pub num_threads: u32 
}

impl Default for CpuBackendSettings {
    fn default() -> Self {
        Self { 
            num_threads: num_cpus::get() as u32
        }
    }
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
    sampler: &mut CpuSampler,
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
            
            for sample_index in 0..raytracer_settings.samples_per_pixel {
                let x = (tile.x0 + i) as u32;
                let y = (tile.y0 + j) as u32;

                sampler.start_sample((x, y), sample_index as u64);
                
                let (camera_ray, camera_ray_differentials) = generate_ray(
                    &context.scene.camera, 
                    x, 
                    y,
                    sampler,
                    raytracer_settings.samples_per_pixel,
                );

                radiance += ray_radiance(
                    camera_ray,
                    camera_ray_differentials,
                    context,
                    sampler,
                    traversal_cache, 
                    raytracer_settings
                );
            }
            
            radiance /= raytracer_settings.samples_per_pixel as f32; 
            tile_radiance_buffer.push(radiance);
        }
    }

    tile_radiance_buffer
}

fn render_aovs(
    context: &CpuRaytracingContext,
    sampler: &mut CpuSampler,
    traversal_cache: &mut TraversalCache,
    raytracer_settings: &RaytracerSettings,

    render_output: &mut RenderOutput
) {
    let output_buffer_size = (render_output.width * render_output.height) as usize;

    if raytracer_settings.outputs.contains(AOVFlags::NORMALS) {
        render_output.normals = Some(Vec::with_capacity(output_buffer_size));
    }
    if raytracer_settings.outputs.contains(AOVFlags::UV_COORDS) {
        render_output.uv = Some(Vec::with_capacity(output_buffer_size));
    }
    if raytracer_settings.outputs.contains(AOVFlags::MIP_LEVEL) {
        render_output.mip_level = Some(Vec::with_capacity(output_buffer_size));
    }

    for j in 0..render_output.height {
        for i in 0..render_output.width {
            let x = i;
            let y = j;
            sampler.start_sample((x, y), 0);
            
            let (camera_ray, camera_ray_differentials) = generate_ray(
                &context.scene.camera, 
                x, 
                y,
                sampler,
                raytracer_settings.samples_per_pixel,
            );

            let first_hit_aovs = first_hit_aovs(
                camera_ray, 
                camera_ray_differentials,
                context, 
                traversal_cache
            );

            let FirstHitAOVData { 
                hit: _, 
                uv, 
                normals, 
                mip_level 
            } = first_hit_aovs;

            if raytracer_settings.outputs.contains(AOVFlags::NORMALS) {
                render_output.normals.as_mut().unwrap().push(normals);
            }
            if raytracer_settings.outputs.contains(AOVFlags::UV_COORDS) {
                render_output.uv.as_mut().unwrap().push(uv);
            }
            if raytracer_settings.outputs.contains(AOVFlags::MIP_LEVEL) {
                render_output.mip_level.as_mut().unwrap().push(mip_level.unwrap_or(0.0));
            }
        }
    }
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

pub fn render(
    scene: &Scene, 
    raytracer_settings: &RaytracerSettings, 
    backend_settings: CpuBackendSettings
) -> RenderOutput {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;
    let mut render_output = RenderOutput::new(width as u32, height as u32);

    // construct BVH using embree
    let cpu_acceleration_structures = scene::prepare_cpu_acceleration_structures(scene);
    let cpu_raytracing_context = CpuRaytracingContext::new(
        &scene,
        &cpu_acceleration_structures
    );

    let mut single_threaded_sampler = CpuSampler::from_sampler(&raytracer_settings.sampler, raytracer_settings.seed);

    // first hit AOVs, then render beauty
    if raytracer_settings.outputs.intersects(AOVFlags::FIRST_HIT_AOVS) {
        let mut traversal_cache = TraversalCache::new(&cpu_raytracing_context);
        let start_aov = std::time::Instant::now();
        render_aovs(
            &cpu_raytracing_context, 
            &mut single_threaded_sampler,
            &mut traversal_cache, 
            &raytracer_settings, 
            &mut render_output
        );

        let end_aov = std::time::Instant::now();
        info!("finished rendering aovs in {} seconds", (end_aov - start_aov).as_secs_f32());
    }

    // exit early if beauty is not needed
    if !raytracer_settings.outputs.contains(AOVFlags::BEAUTY) {
        return render_output;
    }

    let single_threaded = backend_settings.num_threads == 1;
    let start_beauty = std::time::Instant::now();
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
            &mut single_threaded_sampler,
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
            sampler_template: &CpuSampler,
            work_queue: Arc<Mutex<Vec<RenderTile>>>,
            result_channel_tx: std::sync::mpsc::Sender<(RenderTile, Vec<Vec3>)>,
            raytracer_settings: &RaytracerSettings
        ) {
            let mut traversal_cache = TraversalCache::new(context);
            let mut sampler = sampler_template.clone();

            loop {
                let mut work_queue_guard = work_queue.lock().expect("lock poisoned");
                let job = work_queue_guard.pop();
                drop(work_queue_guard);

                let Some(tile) = job else {
                    break
                };

                let tile_radiance = render_tile(
                    context, 
                    &mut sampler,
                    &mut traversal_cache, 
                    tile, 
                    raytracer_settings
                );

                result_channel_tx.send((tile, tile_radiance))
                    .expect("no receiver (is main thread gone?)");
            }
        }
        
        std::thread::scope(|scope| {
            for _ in 0..backend_settings.num_threads {
                let thread_work_queue_handle = Arc::clone(&work_queue);
                let thread_result_channel_tx = result_channel_tx.clone();
                scope.spawn(|| {
                    render_thread(
                        &cpu_raytracing_context, 
                        &single_threaded_sampler,
                        thread_work_queue_handle, 
                        thread_result_channel_tx, 
                        &raytracer_settings
                    );
                });
            }

            // TODO: if a thread panics, should handle this (probably also panic)
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

    let end_beauty = std::time::Instant::now();
    info!("finished rendering beauty in {} seconds", (end_beauty - start_beauty).as_secs_f32());

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

    render_output.beauty = Some(radiance);
    render_output
}

#[derive(Debug)]
pub struct SinglePixelOutput {
    pub hit: bool,
    pub uv: Vec2,
    pub normal: Vec3,
    pub radiance: Vec3,
}

pub fn render_single_pixel(
    scene: &Scene, 
    raytracer_settings: &RaytracerSettings,
    x: u32,
    y: u32,
    sample_index: Option<u32>,
) -> SinglePixelOutput {
    let (x, y) = if x >= scene.camera.raster_width as u32 || y >= scene.camera.raster_height as u32 {
        let clamped_x = u32::clamp(x, 0, scene.camera.raster_width as u32 - 1);
        let clamped_y = u32::clamp(y, 0, scene.camera.raster_height as u32 - 1);
        warn!("out-of-bounds coordinates for `render_single_pixel`, clamping to x = {clamped_x}, y = {clamped_y}");

        (clamped_x, clamped_y)
    }
    else {
        (x, y)
    };

    let sample_index = sample_index.unwrap_or(0) as u64;

    let cpu_acceleration_structures = scene::prepare_cpu_acceleration_structures(scene);
    let cpu_raytracing_context = CpuRaytracingContext::new(
        &scene,
        &cpu_acceleration_structures
    );

    let mut traversal_cache = TraversalCache::new(&cpu_raytracing_context);
    let mut single_pixel_sampler = CpuSampler::from_sampler(&raytracer_settings.sampler, raytracer_settings.seed);
    single_pixel_sampler.start_sample((x, y), sample_index);

    let (ray, ray_differentials) = generate_ray(
        &scene.camera, 
        x, 
        y,
        &mut single_pixel_sampler,
        raytracer_settings.samples_per_pixel
    );

    let aovs = first_hit_aovs(
        ray, 
        ray_differentials, 
        &cpu_raytracing_context, 
        &mut traversal_cache
    );

    let radiance = ray_radiance(
        ray, 
        ray_differentials, 
        &cpu_raytracing_context, 
        &mut single_pixel_sampler,
        &mut traversal_cache, 
        &raytracer_settings
    );

    SinglePixelOutput { hit: aovs.hit, uv: aovs.uv, normal: aovs.normals, radiance }
}