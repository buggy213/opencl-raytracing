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

        let mut output_radiance = crate::render(&scene, &raytracer_settings, backend_settings);
        
        // TODO: replace w/ snapshot testing, and move tests into their own crate
        // which interfaces with all backends
        /* 
        if raytracer_settings.debug_normals {
            crate::utils::normals_to_rgb(&mut output_radiance);
        }

        let exposure = if raytracer_settings.debug_normals {
            1.0
        }
        else {
            1000.0
        };

        crate::utils::save_png(&output_radiance, exposure, &scene, Path::new(&filename));
        */
    }
}