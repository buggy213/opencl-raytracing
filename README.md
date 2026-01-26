# opencl-raytracing

Physically based renderer that uses ray tracing to solve light transport, heavily inspired by PBRT. 

## Crate architecture

Everything lives under a shared Cargo workspace; a brief description of each crate is given below:

- `crates/cli`: command-line interface (currently targets the CPU backend; intended to generalize to other backends).
- `crates/embree4`: higher-level wrapper around Embree used by the CPU backend for BVH construction.
- `crates/embree4-sys`: raw Embree FFI bindings used by `embree4`.
- `crates/raytracing`: scene description and shared math/vocabulary types (e.g. `Vec3`).
- `crates/raytracing-cl`: old OpenCL backend (likely rotted; may be removed).
- `crates/raytracing-cpu`: primary reference renderer that runs on the CPU.
- `crates/raytracing-optix`: NVIDIA OptiX backend (mostly a stub at the moment).
- `crates/viewer`: WebGPU-based viewer for render output; intended for AoV/BVH visualization later.

The layout is split between scene description / IO (`crates/raytracing`, `crates/cli`, `crates/viewer`) and rendering backends (`raytracing-cpu`, `raytracing-optix`, `raytracing-cl`).

Future ideas: DXR and/or Vulkan ray tracing support.

## Notable dependencies / system requirements

- CPU backend: Embree 4 is required. Set `EMBREE_DIR` to the Embree install directory. On Linux, `LD_LIBRARY_PATH` needs to include `${EMBREE_DIR}/lib` because of rpath issues.
- OptiX backend: CUDA toolkit installed for your platform (currently tested on Linux). OptiX 9.0 should also be installed; set `OPTIX90_PATH` to the OptiX 9.0 install directory.
- Both CPU and OptiX backends: `libclang` is required for `bindgen` at build time.

## CLI usage

See `crates/cli/src/main.rs` for the CLI definition. You can also inspect the help output:

```bash
cargo run -p cli -- --help
```

Typical invocation from the repo root:

```bash
cargo run -p cli -- --scene-path path/to/scene.gltf --output out/render.exr
```

Scenes are specified as GLTF files (or a builtin scene name). Outputs are written under `scenes/output/` using the path you pass (default: `output.exr`). 

### Backend selection

The CLI supports multiple rendering backends via the `--backend` flag:

- `cpu` (default): Reference CPU renderer
- `optix`: NVIDIA OptiX backend (requires building with `--features optix`)

```bash
# CPU backend (default)
cargo run -p cli -- --scene-name sphere -s 4 full

# OptiX backend (requires optix feature)
cargo run -p cli --features optix -- --backend optix --scene-name sphere -s 4 full
```

### CLI help output

```text
Usage: cli [OPTIONS] [COMMAND]

Commands:
  full         Full frame render with AOV control
  pixel        Render a single pixel and print diagnostics
  list-scenes  List all builtin test scenes as JSON
  help         Print this message or the help of the given subcommand(s)

Options:
      --scene-path <SCENE_PATH>        Load a GLTF scene from disk
      --scene-name <SCENE_NAME>        Load a builtin test scene by name
  -o, --output <OUTPUT>                Output filename (written under scenes/output/)
      --output-format <OUTPUT_FORMAT>  Force output format (otherwise inferred from extension) [possible values: png, exr]
      --backend <BACKEND>              Rendering backend [default: cpu] [possible values: cpu, optix]
  -t, --num-threads <NUM_THREADS>      CPU worker threads
  -d, --ray-depth <RAY_DEPTH>          Maximum ray depth (bounces)
  -s, --spp <SPP>                      Samples per pixel
  -l, --light-samples <LIGHT_SAMPLES>  Light sample count
      --sampler <SAMPLER>              Sampler type [possible values: independent, stratified]
  -h, --help                           Print help
```

### CLI help output (`full` subcommand)

```text
Full frame render with AOV control

Usage: cli <--scene-path <SCENE_PATH>|--scene-name <SCENE_NAME>> full [OPTIONS]

Options:
      --aov <AOV>...           Comma-separated AOV list (e.g. normal,uv or n,u)
      --no-beauty <NO_BEAUTY>  Disable beauty output (useful when only AOVs are desired) [possible values: true, false]
  -h, --help                   Print help
```

### CLI help output (`pixel` subcommand)

```text
Render a single pixel and print diagnostics

Usage: cli <--scene-path <SCENE_PATH>|--scene-name <SCENE_NAME>> pixel <X> <Y>

Arguments:
  <X>  Pixel x coordinate
  <Y>  Pixel y coordinate

Options:
  -h, --help  Print help
```

### Interactive mode
CLI also has the `--interactive` flag, which allows you to set these different settings within a user-friendly terminal interface. All of the same settings are exposed. 

## Tests

- **Unit tests**: Typically used for math and utility routines. Run with `cargo test`.
- **Visual regression tests**: Snapshot-based image comparison using `crates/visual-testing`. See [crates/visual-testing/README.md](crates/visual-testing/README.md) for details.

Quick start for visual testing:

```bash
cargo install --path crates/cli
cd crates/visual-testing
uv run rttest cpu --bless -- -s 1 -l 1   # create reference images
uv run rttest cpu -- -s 1 -l 1           # run tests
```
