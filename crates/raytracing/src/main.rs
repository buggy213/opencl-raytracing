

use std::{fs::{self, read_to_string}, ptr::{null_mut}, path::Path};

use accel::{bvh2::BVH2, LinearizedBVHNode};
use cl3::{ext::CL_DEVICE_TYPE_GPU, memory::CL_MEM_READ_WRITE, types::CL_TRUE, platform::CL_PLATFORM_NAME};
use geometry::{Transform, Vec3, CLMesh};
use lights::Light;
use opencl3::{program::Program, platform::Platform, device::Device, context::Context, memory::Buffer, command_queue::CommandQueue, kernel::{Kernel, ExecuteKernel}};
use scene::{Scene, RenderTile};
use cli::Cli;
use clap::Parser;

mod geometry;
mod accel;
mod macros;
mod scene;
mod lights;
mod cli;
mod backends;

fn main() {
    let cli: Cli = Cli::parse();
    let use_tile: bool = cli.tile_x.and(cli.tile_y).and(cli.tile_height).and(cli.tile_width).is_some();
    let render_tile: Option<RenderTile> = if use_tile { 
        Some(RenderTile { x0: cli.tile_x.unwrap(), y0: cli.tile_y.unwrap(), x1: cli.tile_x.unwrap() + cli.tile_width.unwrap(), y1: cli.tile_y.unwrap() + cli.tile_height.unwrap() }) } 
    else { 
        None 
    };
    let (_platform, _device, context, _program, kernel, command_queue) = initialize_cl(false);
    let scene: Scene = Scene::from_file(Path::new("scenes/sponza.gltf"), render_tile).expect("failed to load gltf file");
    let width: usize = if !use_tile { scene.camera.raster_width } else { cli.tile_width.unwrap() };
    let height: usize = if !use_tile { scene.camera.raster_height } else { cli.tile_height.unwrap() };
    let seed_buffer: Buffer<u32> = create_seed_buffer(&context, width, height);
    let image_buffer: Buffer<f32> = create_image_buffer(&context, width, height);
    let camera_transform: Transform = scene.camera.world_to_raster;
    let camera_transform_cl: Buffer<f32> = camera_transform.to_opencl_buffer(&context, &command_queue);
    let (mut mesh, mesh_to_world_transform) = scene.mesh;
    mesh.apply_transform(mesh_to_world_transform);
    let embree_device: embree4_sys::Device = embree4_sys::Device::new();
    let mesh_bvh: BVH2 = BVH2::create(&embree_device, &mesh);
    let bvh_device: Vec<LinearizedBVHNode> = LinearizedBVHNode::linearize_bvh_mesh(&mesh_bvh, &mut mesh);
    let bvh_device: Buffer<LinearizedBVHNode> = LinearizedBVHNode::to_cl_buffer(&bvh_device, &context, &command_queue);
    let mesh_device = mesh.to_cl_mesh(&context, &command_queue);
    let lights: Buffer<Light> = Light::to_cl_lights(&scene.lights, &context, &command_queue);
    
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&image_buffer)
            .set_arg(&(width as i32))
            .set_arg(&(height as i32))
            .set_arg(&1)
            .set_arg(&seed_buffer)
            .set_arg(&camera_transform_cl)
            .set_arg(&scene.camera.camera_position[0])
            .set_arg(&scene.camera.camera_position[1])
            .set_arg(&scene.camera.camera_position[2])
            .set_arg(&(scene.camera.is_perspective as i32))
            .set_arg(&mesh_device.vertices)
            .set_arg(&mesh_device.triangles)
            .set_arg(&bvh_device)
            .set_arg(&lights)
            .set_arg(&(scene.lights.len() as i32))
            .set_global_work_sizes(&[width, height])
            .enqueue_nd_range(&command_queue).expect("failed to enqueue kernel")
    };

    kernel_event.wait().expect("failed to run kernel");

    output_image_buffer(&image_buffer, &command_queue, width, height, cli.output_file.as_deref());
}
