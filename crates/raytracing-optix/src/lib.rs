//! # OptiX backend for raytracer
//! 
//! This is mostly a stub library; the actual code lives in C++
//! The surface area of OptiX library is quite large, and no good FFI library exists (yet)
//! `rust-cuda` looks promising, but doesn't seem fully fleshed out
//! So, interfacing w/ OptiX + CUDA (runtime + driver) is done in C++
//! Can always move code into Rust later if need be...

use raytracing::{geometry::Vec3, scene::Scene};

mod scene;
mod optix;

// TODO: common raytracer settings should be factored out of cpu and gpu backends
pub fn render(scene: &Scene, /* raytracer_settings: RaytracerSettings */) -> &[Vec3] {
    // SAFETY: no preconditions
    let optix_ctx = unsafe { optix::initOptix() };
    
    let scene_as = scene::prepare_optix_acceleration_structures(optix_ctx, scene);
    let normals_kernel = optix::kernels::NORMALS;
    let normals_pipeline = unsafe { 
        optix::makeBasicPipeline(
            optix_ctx, 
            normals_kernel.as_ptr(), 
            normals_kernel.len()
        ) 
    };

    // SAFETY: optix_ctx is valid
    unsafe { optix::destroyOptix(optix_ctx); }

    todo!()
}