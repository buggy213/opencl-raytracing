use std::{fs::File, path::{Path, PathBuf}};

use clap::Parser;
use raytracing::{geometry::Vec3, scene::Scene};
use raytracing_cpu::{render, RaytracerSettings};

#[derive(Debug, Parser)]
struct CommandLineArguments {
    #[arg(short, long)]
    input: Option<PathBuf>,
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

fn main() {
    tracing_subscriber::fmt::init();

    let cli_args = CommandLineArguments::parse();
    let default_scene = PathBuf::from("scenes/cb.glb");
    let path = cli_args.input.unwrap_or(default_scene);
    let mut scene = Scene::from_gltf_file(&path, None)
        .expect("failed to load scene");

    let raytracer_settings = RaytracerSettings {
        max_ray_depth: 1,
        light_sample_count: cli_args.light_samples,
        samples_per_pixel: cli_args.spp,
        accumulate_bounces: false,

        debug_normals: true,
    };

    let output = render(&mut scene, raytracer_settings);

    let output_path = Path::new("scenes/cb.png");
    save_png(&output, &scene, output_path);
}