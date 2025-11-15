#include <stdio.h>

__global__ void device_code() {
    printf("hi from cuda device thread %d\n", threadIdx.x);
}

__host__ void host_code() {
    printf("hi from cuda 2\n");
    device_code<<<1, 10>>>();
    cudaDeviceSynchronize();
}