use std::{fs::File, path::{Path, PathBuf}};

use clap::Parser;
use raytracing::{geometry::Vec3, scene::Scene};
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
    #[arg(short, long, default_value_t = 1)]
    spp: u32,
    #[arg(short, long, default_value_t = 1)]
    light_samples: u32,
}

fn save_png(radiance: &[Vec3], scene: &Scene, output_path: &Path) {
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;

    let file = File::create(output_path).expect("failed to create output file");
    let mut encoder = png::Encoder::new(file, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0));

    let mut writer = encoder
        .write_header()
        .expect("failed to write PNG header");

    let image_data: Vec<u8> = radiance.iter().flat_map(|v| {
        let r = (v.x() / 1000.0 * 255.0).clamp(0.0, 255.0) as u8;
        let g = (v.y() / 1000.0 * 255.0).clamp(0.0, 255.0) as u8;
        let b = (v.z() / 1000.0 * 255.0).clamp(0.0, 255.0) as u8;
        [r, g, b]
    }).collect();

    writer.write_image_data(&image_data).expect("failed to write PNG data");
}

fn normals_to_rgb(normals: &mut [Vec3]) {
    for normal in normals {
        *normal += Vec3(1.0, 1.0, 1.0);
        *normal /= 2.0;
    }
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
        max_ray_depth: 1,
        light_sample_count: cli_args.light_samples,
        samples_per_pixel: cli_args.spp,
        accumulate_bounces: true,

        num_threads: cli_args.num_threads,

        debug_normals: false,
    };

    let mut output = render(&scene, raytracer_settings);
    
    if raytracer_settings.debug_normals {
        normals_to_rgb(&mut output);
    }

    let output_path = Path::new("scenes/test.png");
    save_png(&output, &scene, output_path);
}