#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <fstream>

#define cl_expect(code) cl_expect_debug((code), __FILE__, __LINE__) 
void cl_expect_debug(cl_int code, const char* const file, int const line) {
    if (code != CL_SUCCESS) {
        std::cout << "encountered CL error: " << code << " at " << file << ":" << line << std::endl;
        exit(-1);
    }
}

int main() {
    cl_int result;
    cl::Platform platform = cl::Platform::get(&result);
    cl_expect(result);
    cl::Platform p = cl::Platform::setDefault(platform);
    if (platform != p) {
        std::cout << "failed to set default platform\n";
        exit(-1);
    }

    cl::Context context = cl::Context::getDefault();
    cl::Device device = cl::Device::getDefault();


    cl::CommandQueue command_queue{context, device};

    // create frame buffer
    uint32_t frame_height = 1080;
    uint32_t frame_width = 1920;
    cl::Buffer frame_buffer{context, CL_MEM_READ_WRITE, frame_width * frame_height * sizeof(float) * 3, nullptr, &result};
    cl_expect(result);

    // load kernels from file
    std::ifstream kernel_file("kernels/kernel.cl");
    std::stringstream buffer;
    buffer << kernel_file.rdbuf();
    std::string kernel_source = buffer.str();
    
    std::vector<std::string> kernels{kernel_source};
    cl::Program::Sources sources{kernels};
    cl::Program program{context, sources};

    result = program.build("-cl-std=CL3.0");
    if (result != CL_SUCCESS) {
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(-1);
    }

    auto kernel = cl::KernelFunctor<cl::Buffer, int, int>(program, "render", &result);
    cl_expect(result);

    cl::NDRange range{frame_width, frame_height};
    cl::NDRange workgroup_size{8, 8};
    cl::EnqueueArgs enqueue_args{command_queue, range, workgroup_size};
    kernel(enqueue_args, frame_buffer, frame_width, frame_height).wait();

    
    float* result_buffer = (float*) malloc(sizeof(float) * frame_width * frame_height * 3);
    command_queue.enqueueReadBuffer(frame_buffer, CL_TRUE, 0, sizeof(float) * frame_width * frame_height * 3, result_buffer);

    std::cout << "P3\n" << frame_width << " " << frame_height << "\n255\n";
    for (uint32_t j = 0; j < frame_height; j--) {
        for (uint32_t i = 0; i < frame_width; i++) {
            size_t pixel_index = ((frame_height - 1 - j) * frame_width + i) * 3;
            float r = result_buffer[pixel_index + 0];
            float g = result_buffer[pixel_index + 1];
            float b = result_buffer[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            // std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    return 0;
}