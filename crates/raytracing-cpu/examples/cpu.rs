use std::{num::NonZero, path::{Path, PathBuf}};

use clap::Parser;

use raytracing::{scene, settings::RaytracerSettings};
use raytracing_cpu::{CpuBackendSettings, render, render_single_pixel};
use raytracing::scene::test_scenes;

#[derive(Debug, clap::Parser)]
struct CommandLineArguments {
    #[command(flatten)]
    input: InputScene,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(short = 't', long)]
    num_threads: Option<u32>,
    #[arg(short = 'd', long)]
    ray_depth: Option<u32>,
    #[arg(short, long)]
    spp: Option<u32>,
    #[arg(short, long)]
    light_samples: Option<u32>,

    #[command(subcommand)]
    render_command: Option<RenderCommand>
}

#[derive(Debug, clap::Args)]
#[group(required = true, multiple = false)]
struct InputScene {
    #[arg(long)]
    scene_path: Option<PathBuf>,
    #[arg(long)]
    scene_name: Option<String>,
}

#[derive(Debug, clap::Subcommand)]
enum RenderCommand {
    Full,
    Normals,
    Pixel {
        x: u32,
        y: u32
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli_args = CommandLineArguments::parse();
    let scene = if let Some(filename) = cli_args.input.scene_path {
        scene::scene_from_gltf_file(&filename)
            .expect("failed to load scene")
    } else if let Some(name) = cli_args.input.scene_name {
        let scene_func = test_scenes::all_test_scenes()
            .iter()
            .find(|s| s.name == name)
            .expect("failed to find scene")
            .scene_func;

        scene_func()
    } else {
        unreachable!("clap should prevent this");
    };

    let render_command = cli_args.render_command.unwrap_or(RenderCommand::Full);
    
    let mut raytracer_settings = RaytracerSettings::default();
    raytracer_settings.max_ray_depth = cli_args.ray_depth.unwrap_or(raytracer_settings.max_ray_depth);
    raytracer_settings.light_sample_count = cli_args.ray_depth.unwrap_or(raytracer_settings.light_sample_count);
    raytracer_settings.samples_per_pixel = cli_args.ray_depth.unwrap_or(raytracer_settings.samples_per_pixel);
    raytracer_settings.accumulate_bounces = true;
    raytracer_settings.debug_normals = matches!(render_command, RenderCommand::Normals);

    let mut backend_settings = CpuBackendSettings::default();
    backend_settings.num_threads = cli_args.num_threads.unwrap_or(backend_settings.num_threads);

    if let RenderCommand::Pixel { x, y } = render_command {
        for i in 0..8 {
            raytracing_cpu::set_seed(i);
            let pixel_radiance = render_single_pixel(&scene, raytracer_settings, x, y);
            dbg!(i);
            dbg!(pixel_radiance);
        }

        return;
    }
    
    let mut output = render(&scene, raytracer_settings, backend_settings);
    
    if raytracer_settings.debug_normals {
        raytracing_cpu::utils::normals_to_rgb(&mut output);
    }

    let output_folder = Path::new("scenes/output");
    let output_file = output_folder.join(
        cli_args.output.as_ref().unwrap_or(&PathBuf::from("test.png"))
    );

    let exposure = if raytracer_settings.debug_normals {
        1.0
    }
    else {
        1000.0
    };

    raytracing_cpu::utils::save_png(&output, exposure, &scene, &output_file);
}