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

unsafe extern "C" {
    unsafe fn test();
}

pub fn test_rs() {
    unsafe { test(); }
}

// TODO: common raytracer settings should be factored out of cpu and gpu backends
pub fn render(scene: &Scene, /* raytracer_settings: RaytracerSettings */) -> &[Vec3] {
    todo!()
}