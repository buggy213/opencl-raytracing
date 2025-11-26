use std::path::{Path, PathBuf};

use clap::Parser;
use raytracing::scene::Scene;
use raytracing_cpu::{render, RaytracerSettings};
use raytracing::scene::test_scenes;

#[derive(Debug, Parser)]
struct CommandLineArguments {
    #[arg(short, long)]
    input: Option<PathBuf>,
    #[arg(long)]
    scene: Option<String>,
    #[arg(short = 't', long, default_value_t = 1)]
    num_threads: u32,
    #[arg(short = 'd', long, default_value_t = 1)]
    ray_depth: u32,
    #[arg(short, long, default_value_t = 1)]
    spp: u32,
    #[arg(short, long, default_value_t = 1)]
    light_samples: u32,
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli_args = CommandLineArguments::parse();
    let scene = if let Some(filename) = cli_args.input {
        Scene::from_gltf_file(&filename)
            .expect("failed to load scene")
    } else if let Some(name) = cli_args.scene {
        let scene_func = test_scenes::all_test_scenes()
            .iter()
            .find(|s| s.name == name)
            .expect("failed to find scene")
            .func;

        scene_func()
    } else {
        let default_scene = PathBuf::from("scenes/cbbunny.glb");
        Scene::from_gltf_file(&default_scene)
            .expect("failed to load scene")
    };

    let raytracer_settings = RaytracerSettings {
        max_ray_depth: cli_args.ray_depth,
        light_sample_count: cli_args.light_samples,
        samples_per_pixel: cli_args.spp,
        accumulate_bounces: true,

        num_threads: cli_args.num_threads,

        debug_normals: false,
    };

    let mut output = render(&scene, raytracer_settings);
    
    if raytracer_settings.debug_normals {
        raytracing_cpu::utils::normals_to_rgb(&mut output);
    }

    let output_path = Path::new("scenes/output/test.png");
    raytracing_cpu::utils::save_png(&output, &scene, output_path);
}