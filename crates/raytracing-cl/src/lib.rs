use std::{fs::{self, read_to_string}, ptr::null_mut};

use cl3::ext::{CL_DEVICE_TYPE_GPU, CL_MEM_READ_WRITE, CL_PLATFORM_NAME};
use opencl3::{command_queue::CommandQueue, context::Context, device::Device, kernel::Kernel, memory::Buffer, platform::Platform, program::Program};


fn compile_kernel(context: &Context) -> Program {
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
        context, 
        source_strs.as_slice(), 
        "-cl-std=CL3.0"
    ).expect("unable to build program from source");

    program
}

fn load_binary_kernel(context: &Context) -> Program {
    let source_strings: Vec<u8> = fs::read("cl/kernel.pocl").expect("failed to read binary kernel file");

    Program::create_and_build_from_binary(
        context, 
        &[&source_strings], 
        ""    
    ).expect("unable to build program from binary")
}

fn initialize_cl(use_binary: bool) -> (Platform, Device, Context, Program, Kernel, CommandQueue) {
    let platform_ids = cl3::platform::get_platform_ids().expect("unable to get platform ids");
    let platform_name: cl3::info_type::InfoType = cl3::platform::get_platform_info(platform_ids[0], CL_PLATFORM_NAME).expect("failed to query platform info");
    assert!(platform_ids.len() == 1, "require only one platform be present");
    println!("using {} as opencl platform", platform_name);

    let platform = Platform::new(platform_ids[0]);
    let device_ids = platform.get_devices(CL_DEVICE_TYPE_GPU).expect("unable to get device ids");
    assert!(device_ids.len() == 1, "require only one device be present");

    let device = Device::new(device_ids[0]);
    let context = Context::from_device(&device).expect("unable to create context");

    let program = if use_binary { load_binary_kernel(&context) } else { compile_kernel(&context) };

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

fn create_seed_buffer(context: &Context, width: usize, height: usize) -> Buffer<u32> {
    let mut seed_buffer;
    unsafe {
        seed_buffer = Buffer::create(
            context, 
            CL_MEM_READ_WRITE, 
            width * height * 2, 
            null_mut()
        ).expect("unable to create seed buffer");
    }
    seed_buffer
}

fn populate_seed_buffer(seed_buffer: &mut Buffer<u32>, command_queue: &CommandQueue, width: usize, height: usize) {
    let mut seed_buffer_host = vec![0; width*height*2];
    for i in 0..width*height*2 {
        seed_buffer_host[i] = rand::random();
        unsafe {
            command_queue.enqueue_write_buffer(
                seed_buffer, 
                CL_TRUE, 
                0, 
                &seed_buffer_host, 
                &[]
            ).expect("failed to write seed buffer");
        }
    }
}

fn create_image_buffer(context: &Context, width: usize, height: usize) -> Buffer<f32> {
    let image_buffer;
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

fn output_image_buffer(buffer: &Buffer<f32>, command_queue: &CommandQueue, width: usize, height: usize, filename: Option<&str>) {
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
        format!("output/{}", filename.unwrap_or("output.png")), 
        buf.as_slice(),
        width as u32, 
        height as u32, 
        image::ColorType::Rgb8, 
        image::ImageFormat::Png
    ).expect("failed to write png");
}