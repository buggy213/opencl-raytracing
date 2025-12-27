use std::path::{Path, PathBuf};

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
    let (builtin_scene_settings, scene) = if let Some(filename) = cli_args.input.scene_path {
        let scene = scene::scene_from_gltf_file(&filename)
            .expect("failed to load scene");
        (None, scene)
    } else if let Some(name) = cli_args.input.scene_name {
        let scene_descriptor = test_scenes::all_test_scenes()
            .iter()
            .find(|s| s.name == name)
            .expect("failed to find scene");

        let settings = (scene_descriptor.settings_func)();
        let scene = (scene_descriptor.scene_func)();
        (Some(settings), scene)
    } else {
        unreachable!("clap should prevent this");
    };

    let render_command = cli_args.render_command.unwrap_or(RenderCommand::Full);
    
    let raytracer_settings = if let Some(builtin_scene_settings) = builtin_scene_settings {
        builtin_scene_settings
    }
    else {
        let mut settings = RaytracerSettings::default();
        settings.max_ray_depth = cli_args.ray_depth.unwrap_or(settings.max_ray_depth);
        settings.light_sample_count = cli_args.light_samples.unwrap_or(settings.light_sample_count);
        settings.samples_per_pixel = cli_args.spp.unwrap_or(settings.samples_per_pixel);
        settings.accumulate_bounces = true;
        settings.debug_normals = matches!(render_command, RenderCommand::Normals);
        
        settings
    };

    let mut backend_settings = CpuBackendSettings::default();
    backend_settings.num_threads = cli_args.num_threads.unwrap_or(backend_settings.num_threads);

    if let RenderCommand::Pixel { x, y } = render_command {
        for i in 0..1 {
            raytracing_cpu::set_seed(i);
            let pixel_radiance = render_single_pixel(&scene, raytracer_settings, x, y);
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