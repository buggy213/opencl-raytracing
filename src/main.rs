use std::{fs::{self, read_to_string}, ptr::{null_mut}};

use cl3::{ext::CL_DEVICE_TYPE_GPU, memory::CL_MEM_READ_WRITE, types::CL_TRUE};
use geometry::{Transform, Matrix4x4, Vec3};
use opencl3::{program::Program, platform::Platform, device::Device, context::Context, memory::Buffer, command_queue::CommandQueue, kernel::{Kernel, ExecuteKernel}};

mod geometry;

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn initialize_cl() -> (Platform, Device, Context, Program, Kernel, CommandQueue) {
    let platform_ids = cl3::platform::get_platform_ids().expect("unable to get platform ids");
    assert!(platform_ids.len() == 1, "require only one platform be present");

    let platform = Platform::new(platform_ids[0]);
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).expect("unable to get device ids");
    assert!(device_ids.len() == 1, "require only one device be present");

    let device = Device::new(device_ids[0]);
    let context = Context::from_device(&device).expect("unable to create context");

    let source_strings: Vec<String> = fs::read_dir("cl").expect("unable to open directory").map(|p| {
        let p = p.expect("unable to read directory entry");
        read_to_string(p.path()).expect("unable to read file")
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

fn create_seed_buffer(context: &Context, command_queue: &CommandQueue) -> Buffer<u32> {
    let mut seed_buffer;
    let mut seed_buffer_host = [0; WIDTH*HEIGHT*2];
    for i in 0..WIDTH*HEIGHT*2 {
        seed_buffer_host[i] = rand::random();
    }
    unsafe {
        seed_buffer = Buffer::create(
            context, 
            CL_MEM_READ_WRITE, 
            WIDTH * HEIGHT * 2, 
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

fn create_image_buffer(context: &Context, command_queue: &CommandQueue) -> Buffer<f32> {
    let mut image_buffer;
    unsafe {
        image_buffer = Buffer::create(
            context, 
            CL_MEM_READ_WRITE, 
            WIDTH * HEIGHT * 3, 
            null_mut()
        ).expect("unable to create image buffer");
    }
    image_buffer
}

fn output_image_buffer(buffer: &Buffer<f32>, command_queue: &CommandQueue) {
    let mut color_data: Vec<f32> = vec![0.0f32; WIDTH*HEIGHT*3];
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
        "output.png", 
        buf.as_slice(),
        WIDTH as u32, 
        HEIGHT as u32, 
        image::ColorType::Rgb8, 
        image::ImageFormat::Png
    ).expect("failed to write png");
}

// Creates transform from camera space to raster space through screen space
// fov given in degrees
fn create_perspective_transform(far_clip: f32, near_clip: f32, fov: f32) -> Transform {
    let persp: Matrix4x4 = Matrix4x4::create(
        1.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, far_clip / (far_clip - near_clip), - (far_clip * near_clip) / (far_clip - near_clip), 
        0.0, 0.0, 1.0, 0.0
    );

    let persp: Transform = Transform::from(persp);
    let invt = 1.0 / f32::tan(fov.to_radians() / 2.0);
    let fov_scale = Transform::scale(Vec3(invt, invt, 1.0));
    let screen_to_raster = Transform::translate(Vec3(1.0, 1.0, 0.0)).compose(Transform::scale(Vec3(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0, 1.0)));
    return persp.compose(fov_scale).compose(screen_to_raster);
}

fn main() {
    let (platform, device, context, program, kernel, command_queue) = initialize_cl();
    let seed_buffer = create_seed_buffer(&context, &command_queue);
    let image_buffer = create_image_buffer(&context, &command_queue);
    let camera_transform = create_perspective_transform(1000.0, 0.01, 30.0);
    let camera_transform_cl = camera_transform.to_opencl_buffer(&context, &command_queue);

    println!("{:?}", camera_transform);

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&image_buffer)
            .set_arg(&(WIDTH as i32))
            .set_arg(&(HEIGHT as i32))
            .set_arg(&1)
            .set_arg(&seed_buffer)
            .set_arg(&camera_transform_cl)
            .set_global_work_sizes(&[WIDTH, HEIGHT])
            .enqueue_nd_range(&command_queue).expect("failed to enqueue kernel")
    };

    kernel_event.wait().expect("failed to run kernel");

    output_image_buffer(&image_buffer, &command_queue);
}
