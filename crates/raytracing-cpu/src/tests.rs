use std::path::Path;

use crate::{CpuBackendSettings};

#[test]
fn sanity_tests() {
    // make test directory, if it doesn't exist already
    _ = std::fs::create_dir("test_output");

    for test_scene_descriptor in raytracing::scene::test_scenes::all_test_scenes() {
        let filename = format!("test_output/{}.png", test_scene_descriptor.name);
        let scene = (test_scene_descriptor.scene_func)();
        let raytracer_settings = (test_scene_descriptor.settings_func)();

        let backend_settings = CpuBackendSettings {
            num_threads: 16
        };

        let output_radiance = crate::render(&scene, raytracer_settings, backend_settings);
        crate::utils::save_png(&output_radiance, 1000.0, &scene, Path::new(&filename));
    }
}