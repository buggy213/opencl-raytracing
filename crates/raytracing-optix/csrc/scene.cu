#include "scene.h"

#include <cuda.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <vector>

// TODO: can probably optimize by using separate streams for every IAS, and adding synchronization between them
__host__ std::pair<OptixTraversableHandle, CUdeviceptr> makeSphereGAS(
    OptixDeviceContext ctx,
    const float *center,
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
    optixAccelComputeMemoryUsage(ctx, &accelOptions, &buildInput, 1, &bufferSizes);

    void *d_output;
    void *d_temp;
    cudaMalloc(&d_output, bufferSizes.outputSizeInBytes);
    cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixTraversableHandle output;
    optixAccelBuild(
        ctx,
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
    // since d_output backs the TraversableHandle
    return std::make_pair(
        output, (CUdeviceptr)d_output
    );
}

__host__ std::pair<OptixTraversableHandle, CUdeviceptr> makeMeshGAS(
    OptixDeviceContext ctx,
    const float* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const unsigned int* tris, /* packed */
    size_t trisLen, /* number of uint3's */
    const float* transform /* 4x4 row-major */
) {
    OptixAccelBuildOptions accelOptions;
    OptixBuildInput buildInput;

    memset(&accelOptions, 0, sizeof(accelOptions));
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    memset(&buildInput, 0, sizeof(buildInput));

    void* d_vertices;
    void* d_tris;
    // we only copy the first 12 elements (since it's assumed transform is affine)
    void* d_transform;

    cudaMalloc(&d_vertices, sizeof(float) * verticesLen * 3);
    cudaMalloc(&d_tris, sizeof(unsigned int) * trisLen * 3);
    cudaMalloc(&d_transform, sizeof(float) * 12);
    cudaMemcpy(d_vertices, vertices, sizeof(float) * verticesLen * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tris, tris, sizeof(unsigned int) * trisLen * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_transform, transform, sizeof(float) * 12, cudaMemcpyHostToDevice);

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray& triangleBuildInput = buildInput.triangleArray;
    triangleBuildInput.numSbtRecords = 1;

    // runtime / device APIs are interoperable, so this should "just work"
    triangleBuildInput.vertexBuffers = (CUdeviceptr*)(&d_vertices);
    triangleBuildInput.numVertices = verticesLen;
    triangleBuildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleBuildInput.vertexStrideInBytes = sizeof(float) * 3;

    triangleBuildInput.indexBuffer = (CUdeviceptr)d_tris;
    triangleBuildInput.numIndexTriplets = trisLen;
    triangleBuildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleBuildInput.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangleBuildInput.preTransform = (CUdeviceptr)d_transform;

    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(ctx, &accelOptions, &buildInput, 1, &bufferSizes);

    void *d_output;
    void *d_temp;
    cudaMalloc(&d_output, bufferSizes.outputSizeInBytes);
    cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixTraversableHandle output;
    optixAccelBuild(
        ctx,
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
    cudaFree(d_vertices);
    cudaFree(d_tris);
    cudaFree(d_transform);
    cudaFree(d_temp);

    // we need to return d_output and output together
    // since d_output backs the TraversableHandle
    return std::make_pair(
        output, (CUdeviceptr)d_output
    );
}

__host__ std::pair<OptixTraversableHandle, CUdeviceptr> makeIAS(
    OptixDeviceContext ctx,
    const OptixTraversableHandle* traversableHandles,
    size_t traversableHandlesLen
) {
    std::vector<OptixInstance> instances;
    instances.reserve(traversableHandlesLen);

    for (size_t i = 0; i < traversableHandlesLen; i += 1) {
        OptixInstance instance;
        instance.instanceId = 0; /* we don't use this field, even if there is instancing happening */
        instance.visibilityMask = 255;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = traversableHandles[i];
        instances.push_back(instance);
    }

    void* d_instances;
    cudaMalloc(&d_instances, sizeof(OptixInstance) * traversableHandlesLen);
    cudaMemcpy(d_instances, instances.data(), sizeof(OptixInstance) * traversableHandlesLen, cudaMemcpyHostToDevice);

    OptixAccelBuildOptions accelOptions;
    OptixBuildInput buildInput;

    memset(&accelOptions, 0, sizeof(accelOptions));
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    memset(&buildInput, 0, sizeof(buildInput));

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray& instanceBuildInput = buildInput.instanceArray;
    instanceBuildInput.instances = (CUdeviceptr)d_instances;
    instanceBuildInput.numInstances = traversableHandlesLen;

    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(ctx, &accelOptions, &buildInput, 1, &bufferSizes);

    void *d_output;
    void *d_temp;
    cudaMalloc(&d_output, bufferSizes.outputSizeInBytes);
    cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixTraversableHandle output;
    optixAccelBuild(
        ctx,
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
    // is this legal?
    // "The application is free to release this memory after the build without invalidating the acceleration structure.
    // However, instance-AS builds will continue to refer to other instance-AS and geometry-AS instances and transform nodes."
    cudaFree(d_instances);
    cudaFree(d_temp);

    // we need to return d_output and output together
    // since d_output backs the TraversableHandle
    return std::make_pair(
        output, (CUdeviceptr)d_output
    );
}