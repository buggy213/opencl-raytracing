#include <cstdio>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>


__global__ void device_code() {
    printf("hi from cuda device thread %d\n", threadIdx.x);
}

__host__ void host_code() {
    cudaSetDevice(0);

    optixInit();

    OptixDeviceContext optix_ctx;
    optixDeviceContextCreate(0, nullptr, &optix_ctx);

    printf("hi from cuda 2\n");

    device_code<<<1, 10>>>();

    cudaDeviceSynchronize();
}