

use std::{fs::{self, read_to_string}, ptr::{null_mut}, path::Path};

use accel::{bvh2::BVH2, LinearizedBVHNode};
use cl3::{ext::CL_DEVICE_TYPE_GPU, memory::CL_MEM_READ_WRITE, types::CL_TRUE};
use embree4_sys::rtcGetDeviceError;
use geometry::{Transform, Matrix4x4, Vec3};
use lights::Light;
use opencl3::{program::Program, platform::Platform, device::Device, context::Context, memory::Buffer, command_queue::CommandQueue, kernel::{Kernel, ExecuteKernel}};
use scene::Scene;

use crate::geometry::Mesh;

mod geometry;
mod accel;
mod macros;
mod scene;
mod lights;

fn initialize_cl() -> (Platform, Device, Context, Program, Kernel, CommandQueue) {
    let platform_ids = cl3::platform::get_platform_ids().expect("unable to get platform ids");
    assert!(platform_ids.len() == 1, "require only one platform be present");

    let platform = Platform::new(platform_ids[0]);
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).expect("unable to get device ids");
    assert!(device_ids.len() == 1, "require only one device be present");

    let device = Device::new(device_ids[0]);
    let context = Context::from_device(&device).expect("unable to create context");

    let source_strings: Vec<String> = fs::read_dir("cl").expect("unable to open directory").filter_map(|p| {
        let p = p.expect("unable to read directory entry");
        if p.file_name() == "kernel.cl" {
            Some(read_to_string(p.path()).expect("unable to read file"))
        }
        else {
            None
        }
    }).collect();

    let source_strs: Vec<&str> = source_strings.iter().map(|s| s.as_str()).collect();
    let program = Program::create_and_build_from_sources(
        &context, 
        source_strs.as_slice(), 
        "-cl-std=CL3.0"
    ).expect("unable to build program");

    let kernel = Kernel::create(
        &program, 
        "render"
    ).expect("unable to get kernel");

    let command_queue = CommandQueue::create_default_with_properties(
        &context,
        0,
        0
    ).expect("unable to create command queue");

    (platform, device, context, program, kernel, command_queue)
}

fn create_seed_buffer(context: &Context, command_queue: &CommandQueue, width: usize, height: usize) -> Buffer<u32> {
    let mut seed_buffer;
    let mut seed_buffer_host = vec![0; width*height*2];
    for i in 0..width*height*2 {
        seed_buffer_host[i] = rand::random();
    }
    unsafe {
        seed_buffer = Buffer::create(
            context, 
            CL_MEM_READ_WRITE, 
            width * height * 2, 
            null_mut()
        ).expect("unable to create seed buffer");

        command_queue.enqueue_write_buffer(
            &mut seed_buffer, 
            CL_TRUE, 
            0, 
            &seed_buffer_host, 
            &[]
        ).expect("failed to write seed buffer");
    }
    seed_buffer
}

fn create_image_buffer(context: &Context, command_queue: &CommandQueue, width: usize, height: usize) -> Buffer<f32> {
    let mut image_buffer;
    unsafe {
        image_buffer = Buffer::create(
            context, 
            CL_MEM_READ_WRITE, 
            width * height * 3, 
            null_mut()
        ).expect("unable to create image buffer");
    }
    image_buffer
}

fn output_image_buffer(buffer: &Buffer<f32>, command_queue: &CommandQueue, width: usize, height: usize) {
    let mut color_data: Vec<f32> = vec![0.0f32; width*height*3];
    unsafe {
        command_queue.enqueue_read_buffer(
            buffer, 
            CL_TRUE, 
            0, 
            &mut color_data, 
            &[]
        ).expect("failed to enqueue read").wait().expect("failed to read");
    }

    let buf: Vec<u8> = color_data.iter().map(|x| {
        (x.sqrt().clamp(0.0, 1.0) * 256.0) as u8
    }).collect();
    image::save_buffer_with_format(
        "output/output.png", 
        buf.as_slice(),
        width as u32, 
        height as u32, 
        image::ColorType::Rgb8, 
        image::ImageFormat::Png
    ).expect("failed to write png");
}

fn main() {
    let (platform, device, context, program, kernel, command_queue) = initialize_cl();
    let scene = Scene::from_file(Path::new("scenes/sibenik.gltf")).expect("failed to load gltf file");
    let width = scene.camera.raster_width;
    let height = scene.camera.raster_height;
    let seed_buffer = create_seed_buffer(&context, &command_queue, width, height);
    let image_buffer = create_image_buffer(&context, &command_queue, width, height);
    let camera_transform: Transform = scene.camera.world_to_raster;
    let camera_transform_cl: Buffer<f32> = camera_transform.to_opencl_buffer(&context, &command_queue);
    let (mut mesh, mesh_to_world_transform) = scene.mesh;
    mesh.apply_transform(mesh_to_world_transform);
    let embree_device = embree4_sys::Device::new();
    let mesh_bvh = BVH2::create(&embree_device, &mesh);
    let bvh_device = LinearizedBVHNode::linearize_bvh_mesh(&mesh_bvh, &mut mesh);
    let bvh_device = LinearizedBVHNode::to_cl_buffer(&bvh_device, &context, &command_queue);
    let mesh_device = mesh.to_cl_mesh(&context, &command_queue);

    let lights = Light::to_cl_lights(&scene.lights, &context, &command_queue);

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&image_buffer)
            .set_arg(&(width as i32))
            .set_arg(&(height as i32))
            .set_arg(&128)
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

    output_image_buffer(&image_buffer, &command_queue, width, height);
}
