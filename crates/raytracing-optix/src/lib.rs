//! # OptiX backend for raytracer
//!
//! This is mostly a stub library; the actual code lives in C++
//! The surface area of OptiX library is quite large, and no good FFI library exists (yet)
//! `rust-cuda` looks promising, but doesn't seem fully fleshed out
//! So, interfacing w/ OptiX + CUDA (runtime + driver) is done in C++
//! Can always move code into Rust later if need be...

use raytracing::{
    renderer::{AOVFlags, RaytracerSettings, RenderOutput},
    scene::Scene,
};

mod optix;
mod scene;
mod sbt;

#[derive(Debug, Clone, Copy, Default)]
pub struct OptixBackendSettings {
    // Future: device_id, memory_limit, etc.
}

pub fn render(
    scene: &Scene,
    raytracer_settings: &RaytracerSettings,
    _backend_settings: OptixBackendSettings,
) -> RenderOutput {
    if raytracer_settings.outputs != AOVFlags::NORMALS {
        todo!("only normals for now");
    }

    // SAFETY: no preconditions
    let optix_ctx = unsafe { optix::initOptix(true) };

    let normals_kernel = optix::kernels::NORMALS;
    let normals_pipeline = unsafe { 
        optix::makeAovPipeline(
            optix_ctx, 
            normals_kernel.as_ptr(), 
            normals_kernel.len(),
        ) 
    };
    
    let mut scene_sbt = sbt::AovSbtBuilder::new(scene);
    let scene_as = scene::prepare_optix_acceleration_structures(optix_ctx, scene, &mut scene_sbt);
    let scene_sbt = scene_sbt.finalize(normals_pipeline);
    let scene_textures: optix::OptixTexturesWrapper = scene::prepare_optix_textures(scene);

    let camera: optix::Camera = scene.camera.clone().into();
    let mut render_output = RenderOutput::new(camera.raster_width as u32, camera.raster_height as u32);
    let buffer_size = camera.raster_width * camera.raster_height;
    let zero: optix::Vec3 = raytracing::geometry::Vec3::zero().into();
    let mut normals = vec![
        zero; 
        buffer_size
    ];
    unsafe { 
        optix::launchAovPipeline(
            normals_pipeline,
            scene_sbt.ptr,
            &camera,
            scene_as.handle,
            normals.as_mut_ptr()
        ); 
    }
    
    // in theory, this should be optimized away
    let normals = normals.iter().map(|v| raytracing::geometry::Vec3(v.x, v.y, v.z)).collect();
    render_output.normals = Some(normals);
    render_output
}