#include <cuda.h>
#include <optix_stubs.h>
#include <optix_types.h>

// TODO: can probably optimize by using separate streams for every IAS, and adding synchronization between them
__host__ void makeSphereGAS(
    OptixDeviceContext context,
    float *center,
    float radius
) {
    OptixAccelBuildOptions accelOptions;
    OptixBuildInput buildInput;

    memset(&accelOptions, 0, sizeof(accelOptions));
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    memset(&buildInput, 0, sizeof(buildInput));

    void* d_center;
    void* d_radius;
    cudaMalloc(&d_center, sizeof(float) * 3);
    cudaMalloc(&d_radius, sizeof(float));
    cudaMemcpy(d_center, center, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_radius, &radius, sizeof(float), cudaMemcpyHostToDevice);

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    OptixBuildInputSphereArray& sphereBuildInput = buildInput.sphereArray;
    sphereBuildInput.numSbtRecords = 1;

    // runtime / device APIs are interoperable, so this should "just work"
    sphereBuildInput.numVertices = 1;
    sphereBuildInput.vertexBuffers = (CUdeviceptr*)(&d_center);
    sphereBuildInput.radiusBuffers = (CUdeviceptr*)(&d_radius);

    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &bufferSizes);

    void *d_output;
    void *d_temp;
    cudaMalloc(&d_output, bufferSizes.outputSizeInBytes);
    cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixTraversableHandle output;
    optixAccelBuild(
        context,
        (CUstream)stream,
        &accelOptions,
        &buildInput,
        1,
        (CUdeviceptr)d_temp,
        bufferSizes.tempSizeInBytes,
        (CUdeviceptr)d_output,
        bufferSizes.outputSizeInBytes,
        &output,
        nullptr,
        0
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_center);
    cudaFree(d_radius);
    cudaFree(d_temp);

    // we need to return d_output and output together
    // d_output backs the TraversableHandle
}

__host__ void makeMeshGAS(
    OptixDeviceContext context,
    
) {

}

__host__ void makeIAS() {

}