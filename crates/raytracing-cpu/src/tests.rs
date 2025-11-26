use std::path::Path;

use crate::RaytracerSettings;

#[test]
fn sanity_tests() {
    // make test directory, if it doesn't exist already
    _ = std::fs::create_dir("test_output");

    for test_scene_descriptor in raytracing::scene::test_scenes::all_test_scenes() {
        let filename = format!("test_output/{}.png", test_scene_descriptor.name);
        let scene = (test_scene_descriptor.func)();

        let raytracer_settings = RaytracerSettings {
            max_ray_depth: 5,
            light_sample_count: 1,
            samples_per_pixel: 32,
            accumulate_bounces: true,
    
            num_threads: 16,
    
            debug_normals: false,
        };

        let output_radiance = crate::render(&scene, raytracer_settings);
        crate::utils::save_png(&output_radiance, &scene, Path::new(&filename));
    }
}