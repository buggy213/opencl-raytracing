//! # OptiX backend for raytracer
//!
//! This is mostly a stub library; the actual code lives in C++
//! The surface area of OptiX library is quite large, and no good FFI library exists (yet)
//! `rust-cuda` looks promising, but doesn't seem fully fleshed out
//! So, interfacing w/ OptiX + CUDA (runtime + driver) is done in C++
//! Can always move code into Rust later if need be...

use std::{collections::HashMap, ops::Range, time::Instant};

use indicatif::ProgressStyle;
use tracing::{info, trace_span};
use tracing_indicatif::span_ext::IndicatifSpanExt;

use raytracing::{
    geometry::{Vec2, Vec3}, renderer::{AovFlags, RaytracerSettings, RenderOutput, SinglePixelOutput}, scene::Scene
};

use crate::scene::OptixScene;

mod optix;
mod scene;
mod sbt;

#[derive(Debug, Clone, Copy, Default)]
pub struct OptixBackendSettings {
    // Future: device_id, memory_limit, etc.
}

pub fn render_normals(
    scene: &Scene,
    raytracer_settings: &RaytracerSettings,
    _backend_settings: OptixBackendSettings,
) -> RenderOutput {
    if raytracer_settings.outputs != AovFlags::NORMALS {
        todo!("only normals for now");
    }

    let t = Instant::now();

    // SAFETY: no preconditions
    let optix_ctx = unsafe { optix::initOptix(true) };

    let normals_kernel = optix::kernels::AOV;
    let normals_pipeline = unsafe {
        optix::makeAovPipeline(
            optix_ctx,
            normals_kernel.as_ptr(),
            normals_kernel.len(),
        )
    };

    let mut scene_sbt = sbt::AovSbtBuilder::new(scene);
    let mut primitive_to_sbt_map = HashMap::new();
    let scene_as = scene::prepare_optix_acceleration_structures(optix_ctx, scene, &mut scene_sbt, &mut primitive_to_sbt_map);
    let scene_sbt = scene_sbt.finalize(normals_pipeline);

    info!("setup: {:.2}s", t.elapsed().as_secs_f64());

    let camera: optix::Camera = scene.camera.clone().into();
    let mut render_output = RenderOutput::new(scene.camera.raster_width as u32, scene.camera.raster_height as u32);
    let buffer_size = scene.camera.raster_width * scene.camera.raster_height;
    let zero: optix::Vec3 = raytracing::geometry::Vec3::zero().into();
    let mut normals = vec![
        zero;
        buffer_size
    ];

    let t = Instant::now();
    let render_span = trace_span!("render");
    render_span.pb_set_style(
        &ProgressStyle::with_template("[{elapsed_precise}] {spinner} rendering...")
            .unwrap(),
    );
    let _render_guard = render_span.enter();

    unsafe {
        optix::launchAovPipeline(
            normals_pipeline,
            scene_sbt.ptr,
            &camera,
            scene_as.handle,
            normals.as_mut_ptr()
        );
    }

    drop(_render_guard);
    info!("render: {:.2}s", t.elapsed().as_secs_f64());

    let normals = normals.iter().map(|v| raytracing::geometry::Vec3(v.x, v.y, v.z)).collect();
    render_output.normals = Some(normals);
    render_output
}

pub fn render(
    scene: &Scene,
    raytracer_settings: &RaytracerSettings,
    _backend_settings: OptixBackendSettings,
) -> RenderOutput {
    if raytracer_settings.outputs != AovFlags::BEAUTY {
        todo!("only beauty for now");
    }

    let t = Instant::now();

    // SAFETY: no preconditions
    let optix_ctx = unsafe { optix::initOptix(true) };

    let pathtracer_kernel = optix::kernels::PATHTRACER;
    let pathtracer_pipeline = unsafe {
        optix::makePathtracerPipeline(
            optix_ctx,
            pathtracer_kernel.as_ptr(),
            pathtracer_kernel.len(),
            false,
        )
    };

    info!("optix init + pipeline: {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();

    let mut scene_sbt = sbt::PathtracerSbtBuilder::new(scene);
    let mut primitive_to_sbt_map = HashMap::new();
    let scene_as = scene::prepare_optix_acceleration_structures(optix_ctx, scene, &mut scene_sbt, &mut primitive_to_sbt_map);
    let scene_sbt = scene_sbt.finalize(pathtracer_pipeline);

    info!("acceleration structures: {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();

    let scene_textures = scene::prepare_optix_textures(scene);
    let scene_data = scene::prepare_optix_scene_data(scene, scene_textures, &primitive_to_sbt_map);
    let optix_scene = OptixScene::new(&scene_data);
    let optix_settings: optix::OptixRaytracerSettings = raytracer_settings.clone().into();

    info!("textures + scene data: {:.2}s", t.elapsed().as_secs_f64());

    let mut render_output = RenderOutput::new(scene.camera.raster_width as u32, scene.camera.raster_height as u32);
    let buffer_size = scene.camera.raster_width * scene.camera.raster_height;
    let zero: optix::Vec4 = raytracing::geometry::Vec4::zero().into();
    let mut radiance = vec![
        zero;
        buffer_size
    ];

    let t = Instant::now();
    let render_span = trace_span!("render");
    render_span.pb_set_style(
        &ProgressStyle::with_template("[{elapsed_precise}] {spinner} rendering...")
            .unwrap(),
    );
    let _render_guard = render_span.enter();

    unsafe {
        optix::launchPathtracerPipeline(
            pathtracer_pipeline,
            scene_sbt.ptr,
            optix_settings,
            optix_scene.ffi,
            scene_as.handle,
            radiance.as_mut_ptr()
        );
    }

    drop(_render_guard);
    info!("render: {:.2}s", t.elapsed().as_secs_f64());

    let radiance = radiance.iter().map(|v| raytracing::geometry::Vec3(v.x, v.y, v.z)).collect();
    render_output.beauty = Some(radiance);
    render_output
}

pub fn render_single_pixel(
    scene: &Scene, 
    raytracer_settings: &RaytracerSettings,
    x: u32,
    y: u32,
    sample_indices: Range<u32>,
) -> Vec<SinglePixelOutput> {
    // SAFETY: no preconditions
    let optix_ctx = unsafe { optix::initOptix(true) };

    let pathtracer_kernel = optix::kernels::PATHTRACER_DEBUG;
    let pathtracer_pipeline = unsafe {
        optix::makePathtracerPipeline(
            optix_ctx,
            pathtracer_kernel.as_ptr(),
            pathtracer_kernel.len(),
            true
        )
    };

    let mut scene_sbt = sbt::PathtracerSbtBuilder::new(scene);
    let mut primitive_to_sbt_map = HashMap::new();
    let scene_as = scene::prepare_optix_acceleration_structures(optix_ctx, scene, &mut scene_sbt, &mut primitive_to_sbt_map);
    let scene_sbt = scene_sbt.finalize(pathtracer_pipeline);
    
    let scene_textures = scene::prepare_optix_textures(scene);
    let scene_data = scene::prepare_optix_scene_data(scene, scene_textures, &primitive_to_sbt_map);
    let optix_scene = OptixScene::new(&scene_data);
    let optix_settings: optix::OptixRaytracerSettings = raytracer_settings.clone().into();

    let zero: optix::Vec4 = raytracing::geometry::Vec4::zero().into();
    let mut radiance = vec![
        zero;
        sample_indices.len()
    ];
    unsafe {
        optix::launchPathtracerPipelineDebug(
            pathtracer_pipeline,
            scene_sbt.ptr,
            optix_settings,
            optix_scene.ffi,
            scene_as.handle,
            radiance.as_mut_ptr(),
            optix::SinglePixelDebug { 
                x, 
                y, 
                sample_index_lo: sample_indices.start, 
                sample_index_hi: sample_indices.end 
            }
        );
    };

    radiance.into_iter()
        .enumerate()
        .map(|(idx, radiance)| SinglePixelOutput { 
            sample_index: idx as u32, 
            hit: true, 
            uv: Vec2::zero(), 
            normal: Vec3::zero(), 
            radiance: Vec3(radiance.x, radiance.y, radiance.z) 
        })
        .collect()
}