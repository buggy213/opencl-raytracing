use std::path::{Path, PathBuf};

use bitflags::bitflags_match;
use clap::Parser;

use raytracing::scene::test_scenes;

mod tui;
use raytracing::{
    renderer::{AOVFlags, RaytracerSettings, RenderOutput},
    sampling::Sampler,
    scene,
};
use raytracing_cpu::{CpuBackendSettings, render, render_single_pixel};
use tracing::warn;

#[derive(Debug, clap::Parser)]
struct CommandLineArguments {
    #[arg(short, long, help = "Launch interactive TUI for configuration")]
    interactive: bool,

    #[command(flatten)]
    input: InputScene,

    #[arg(short, long, help = "Output filename (written under scenes/output/)")]
    output: Option<PathBuf>,
    #[arg(long, value_enum, help = "Force output format (otherwise inferred from extension)")]
    output_format: Option<OutputFormat>,

    #[arg(long, value_enum, default_value_t = Backend::Cpu, help = "Rendering backend")]
    backend: Backend,

    #[arg(short = 't', long, help = "CPU worker threads")]
    num_threads: Option<u32>,
    #[arg(short = 'd', long, help = "Maximum ray depth (bounces)")]
    ray_depth: Option<u32>,
    #[arg(short, long, help = "Samples per pixel")]
    spp: Option<u32>,
    #[arg(short, long, help = "Light sample count")]
    light_samples: Option<u32>,

    #[arg(long, value_enum, help = "Sampler type")]
    sampler: Option<SamplerType>,

    #[command(subcommand)]
    render_command: Option<RenderCommand>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SamplerType {
    Independent,
    Stratified,
}

#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
enum Backend {
    #[default]
    Cpu,
    Optix,
}

#[derive(Debug, clap::Args)]
#[group(multiple = false)]
struct InputScene {
    #[arg(long, help = "Load a scene from disk (GLTF or PBRT)")]
    scene_path: Option<PathBuf>,
    #[arg(long, help = "Load a builtin test scene by name")]
    scene_name: Option<String>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum OutputFormat {
    Png,
    Exr,
}

#[derive(Debug, clap::Subcommand)]
enum RenderCommand {
    #[command(about = "Full frame render with AOV control")]
    Full {
        #[arg(
            long,
            value_delimiter = ',',
            num_args = 1..,
            help = "Comma-separated AOV list (e.g. normal,uv or n,u)"
        )]
        aov: Option<Vec<String>>,
        #[arg(long, action, help = "Disable beauty output (useful when only AOVs are desired)")]
        no_beauty: bool,
    },
    #[command(about = "Render a single pixel and print diagnostics")]
    Pixel {
        #[arg(help = "Pixel x coordinate")]
        x: u32,
        #[arg(help = "Pixel y coordinate")]
        y: u32,
    },
    #[command(about = "List all builtin test scenes as JSON")]
    ListScenes,
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli_args = CommandLineArguments::parse();

    if let Some(RenderCommand::ListScenes) = cli_args.render_command {
        let scenes: Vec<&str> = test_scenes::all_test_scenes()
            .iter()
            .map(|s| s.name)
            .collect();
        println!("{}", serde_json::to_string(&scenes).unwrap());
        return;
    }

    // Handle interactive mode
    let cli_args = if cli_args.interactive {
        match tui::run() {
            Ok(Some(args)) => args,
            Ok(None) => {
                println!("Render cancelled.");
                return;
            }
            Err(e) => {
                eprintln!("TUI error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        cli_args
    };

    if cli_args.input.scene_path.is_none() && cli_args.input.scene_name.is_none() {
        eprintln!("error: either --scene-path or --scene-name is required");
        std::process::exit(1);
    }

    let (builtin_scene_settings, scene) = if let Some(filename) = cli_args.input.scene_path.clone() {
        let scene = match filename.extension().and_then(|e| e.to_str()) {
            Some("pbrt") => scene::scene_from_pbrt_file(&filename).expect("failed to load PBRT scene"),
            Some("gltf") | Some("glb") => scene::scene_from_gltf_file(&filename).expect("failed to load GLTF scene"),
            ext => {
                warn!("unrecognized file extension {ext:?}, trying to import as gltf");
                scene::scene_from_gltf_file(&filename).expect("failed to load GLTF")
            }
        };
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

    let mut raytracer_settings = if let Some(builtin_scene_settings) = builtin_scene_settings {
        builtin_scene_settings
    } else {
        RaytracerSettings::default()
    };

    // override builtin / default settings
    raytracer_settings.max_ray_depth = cli_args
        .ray_depth
        .unwrap_or(raytracer_settings.max_ray_depth);
    raytracer_settings.light_sample_count = cli_args
        .light_samples
        .unwrap_or(raytracer_settings.light_sample_count);
    raytracer_settings.samples_per_pixel =
        cli_args.spp.unwrap_or(raytracer_settings.samples_per_pixel);
    raytracer_settings.accumulate_bounces = true;

    if let Some(sampler_type) = cli_args.sampler {
        raytracer_settings.sampler = match sampler_type {
            SamplerType::Independent => Sampler::Independent,
            SamplerType::Stratified => {
                let spp = raytracer_settings.samples_per_pixel;
                let strata = (spp as f32).sqrt().ceil() as u32;
                Sampler::Stratified {
                    jitter: true,
                    x_strata: strata,
                    y_strata: strata,
                }
            }
        };
    }

    let render_command = cli_args.render_command;

    if let Some(RenderCommand::Pixel { x, y }) = render_command {
        for i in 0..1 {
            let pixel = render_single_pixel(&scene, &raytracer_settings, x, y, Some(i));

            println!("sample {i}");
            println!("hit: {}", pixel.hit);
            println!("uv: {}", pixel.uv);
            println!("normal: {}", pixel.normal);
            println!("radiance: {}", pixel.radiance);
        }

        return;
    }

    // hand-parse aovs, i'm no clap expert
    if let Some(RenderCommand::Full { aov, no_beauty }) = &render_command {
        let mut aov_flags = raytracer_settings.outputs;
        for aov_str in aov.iter().flatten() {
            match aov_str.as_str() {
                "n" | "normal" => aov_flags.insert(AOVFlags::NORMALS),
                "u" | "uv" => aov_flags.insert(AOVFlags::UV_COORDS),
                "m" | "mip" => aov_flags.insert(AOVFlags::MIP_LEVEL),
                "b" | "beauty" => warn!("beauty is implicit"),
                _ => warn!("unknown AOV specified: {aov_str}"),
            }
        }

        if *no_beauty {
            aov_flags.remove(AOVFlags::BEAUTY);
        }

        raytracer_settings.outputs = aov_flags;
    }

    if raytracer_settings.outputs.is_empty() {
        warn!("no outputs specified (--no-beauty, and no AOVs), quitting...");
        return;
    }

    if matches!(cli_args.backend, Backend::Optix) && cli_args.num_threads.is_some() {
        eprintln!("error: --threads is not supported with OptiX backend");
        std::process::exit(1);
    }

    let render_output = match cli_args.backend {
        Backend::Cpu => {
            let mut backend_settings = CpuBackendSettings::default();
            backend_settings.num_threads =
                cli_args.num_threads.unwrap_or(backend_settings.num_threads);
            render(&scene, &raytracer_settings, backend_settings)
        }
        Backend::Optix => {
            #[cfg(feature = "optix")]
            {
                let backend_settings = raytracing_optix::OptixBackendSettings::default();
                raytracing_optix::render(&scene, &raytracer_settings, backend_settings)
            }
            #[cfg(not(feature = "optix"))]
            {
                eprintln!("error: OptiX backend not compiled (enable 'optix' feature)");
                std::process::exit(1);
            }
        }
    };

    let output_folder = Path::new("scenes/output");
    let output_file = output_folder.join(
        cli_args
            .output
            .as_ref()
            .unwrap_or(&PathBuf::from("output.exr")),
    );

    save_render_output(
        render_output,
        raytracer_settings.outputs,
        cli_args.output_format,
        &output_file,
    );
}

// TODO: this should probably return a Result, so the user can try a different path in a loop
// and not lose their render
fn save_render_output(
    render_output: RenderOutput,
    aov_flags: AOVFlags,
    output_format: Option<OutputFormat>,
    output_path: &Path,
) {
    let output_format = output_format.unwrap_or_else(|| match output_path.extension() {
        Some(ext) => match ext.to_str() {
            Some("png") => OutputFormat::Png,
            Some("exr") => OutputFormat::Exr,
            Some(_) => {
                warn!("extension not recognized, defaulting to exr");
                OutputFormat::Exr
            }
            None => {
                warn!("weird filename; defaulting to exr");
                OutputFormat::Exr
            }
        },
        None => {
            warn!("no extension provided; defaulting to exr");
            OutputFormat::Exr
        }
    });

    match output_format {
        OutputFormat::Png => save_to_png(render_output, aov_flags, output_path),
        OutputFormat::Exr => save_to_exr(render_output, aov_flags, output_path),
    }
}

// 1 png file for each output, not suffixed w/ aov name if it's the beauty output
fn save_to_png(mut render_output: RenderOutput, aov_flags: AOVFlags, output_path: &Path) {
    fn add_suffix(path: &Path, suffix: &str) -> PathBuf {
        let dir = path.parent().unwrap();
        let base_name = path.file_stem().map(|x| x.to_str()).flatten().unwrap();
        dir.join(format!("{}_{}.{}", base_name, suffix, "png"))
    }

    for (name, flag) in aov_flags.iter_names() {
        bitflags_match!(flag, {
            AOVFlags::BEAUTY => {
                raytracing_cpu::utils::png::save_png(
                    render_output.beauty.as_ref().unwrap(),
                    1000.0,
                    render_output.width,
                    render_output.height,
                    output_path
                );
            }
            AOVFlags::NORMALS => {
                let output_path = add_suffix(&output_path, name);

                raytracing_cpu::utils::png::normals_to_rgb(render_output.normals.as_mut().unwrap());
                raytracing_cpu::utils::png::save_png(
                    render_output.normals.as_ref().unwrap(),
                    1.0,
                    render_output.width,
                    render_output.height,
                    &output_path
                );
            }
            AOVFlags::UV_COORDS => {
                let output_path = add_suffix(&output_path, name);
                let uv_rgb = raytracing_cpu::utils::png::uvs_to_rgb(render_output.uv.as_ref().unwrap());
                raytracing_cpu::utils::png::save_png(
                    &uv_rgb,
                    1.0,
                    render_output.width,
                    render_output.height,
                    &output_path
                );
            }
            AOVFlags::MIP_LEVEL => {
                // TODO: ideally, we would palettize + round to get a nice plot, using something like matplotlib to generate
                // a legend
                warn!("MIP_LEVEL png output not supported (yet)");
            }

            _ => ()
        })
    }
}

fn save_to_exr(render_output: RenderOutput, aov_flags: AOVFlags, output_path: &Path) {
    let mut channels = Vec::new();

    for (_, flag) in aov_flags.iter_names() {
        bitflags_match!(flag, {
            AOVFlags::BEAUTY => {
                if let Some(ref beauty) = render_output.beauty {
                    raytracing_cpu::utils::exr::channels_from_vec3(
                        &mut channels,
                        &["R", "G", "B"],
                        beauty
                    );
                }
            }
            AOVFlags::NORMALS => {
                if let Some(ref normals) = render_output.normals {
                    raytracing_cpu::utils::exr::channels_from_vec3(
                        &mut channels,
                        &["Normal.X", "Normal.Y", "Normal.Z"],
                        normals
                    );
                }
            }
            AOVFlags::UV_COORDS => {
                if let Some(ref uvs) = render_output.uv {
                    let u: Vec<f32> = uvs.iter().map(|v| v.0).collect();
                    let v: Vec<f32> = uvs.iter().map(|v| v.1).collect();

                    raytracing_cpu::utils::exr::channel_from_f32_array(
                        &mut channels,
                        "U",
                        &u
                    );
                    raytracing_cpu::utils::exr::channel_from_f32_array(
                        &mut channels,
                        "V",
                        &v
                    );
                }
            }

            AOVFlags::MIP_LEVEL => {
                if let Some(ref mip_level) = render_output.mip_level {
                    raytracing_cpu::utils::exr::channel_from_f32_array(
                        &mut channels,
                        "Mip Level",
                        &mip_level
                    );
                }
            }
            _ => ()
        })
    }

    raytracing_cpu::utils::exr::save_openexr(
        channels,
        render_output.width,
        render_output.height,
        output_path,
    );
}
