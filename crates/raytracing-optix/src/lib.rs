//! # OptiX backend for raytracer
//!
//! This is mostly a stub library; the actual code lives in C++
//! The surface area of OptiX library is quite large, and no good FFI library exists (yet)
//! `rust-cuda` looks promising, but doesn't seem fully fleshed out
//! So, interfacing w/ OptiX + CUDA (runtime + driver) is done in C++
//! Can always move code into Rust later if need be...

use raytracing::{
    renderer::{RaytracerSettings, RenderOutput},
    scene::Scene,
};

mod optix;
mod scene;

#[derive(Debug, Clone, Copy, Default)]
pub struct OptixBackendSettings {
    // Future: device_id, memory_limit, etc.
}

pub fn render(
    scene: &Scene,
    _raytracer_settings: &RaytracerSettings,
    _backend_settings: OptixBackendSettings,
) -> RenderOutput {
    // SAFETY: no preconditions
    let optix_ctx = unsafe { optix::initOptix(true) };
    
    let scene_as = scene::prepare_optix_acceleration_structures(optix_ctx, scene);
    let normals_kernel = optix::kernels::NORMALS;
    let normals_pipeline = unsafe { 
        optix::makeBasicPipeline(
            optix_ctx, 
            normals_kernel.as_ptr(), 
            normals_kernel.len()
        ) 
    };

    let camera: optix::Camera = scene.camera.clone().into();
    unsafe { 
        optix::launchBasicPipeline(
            normals_pipeline,
            &camera,
            scene_as.handle
        ); 
    }

    // SAFETY: optix_ctx is valid
    unsafe {
        optix::destroyOptix(optix_ctx);
    }

    todo!("OptiX render not fully implemented")
}