use std::path::Path;

use crate::{CpuBackendSettings, RaytracerSettings};

#[test]
fn sanity_tests() {
    // make test directory, if it doesn't exist already
    _ = std::fs::create_dir("test_output");

    for test_scene_descriptor in raytracing::scene::test_scenes::all_test_scenes() {
        let filename = format!("test_output/{}.png", test_scene_descriptor.name);
        let scene = (test_scene_descriptor.func)();

        let raytracer_settings = RaytracerSettings {
            max_ray_depth: 8,
            light_sample_count: 1,
            samples_per_pixel: 32,
            accumulate_bounces: true,
    
            debug_normals: false,
        };

        let backend_settings = CpuBackendSettings {
            num_threads: 16
        };

        let output_radiance = crate::render(&scene, raytracer_settings, backend_settings);
        crate::utils::save_png(&output_radiance, 1000.0, &scene, Path::new(&filename));
    }
}