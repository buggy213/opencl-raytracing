#include "scene.hpp"

#include <cuda.h>
#include <optix_stubs.h>
#include <optix_types.h>

#include <vector>

#include "types.h"
#include "util.hpp"

// TODO: can probably optimize by using separate streams for every IAS, and adding synchronization between them
__host__ OptixAccelerationStructure makeSphereGAS(
    OptixDeviceContext ctx,
    Vec3 center,
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
    cudaMemcpy(d_center, &center, sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_radius, &radius, sizeof(float), cudaMemcpyHostToDevice);

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    OptixBuildInputSphereArray& sphereBuildInput = buildInput.sphereArray;
    sphereBuildInput.numSbtRecords = 1;

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    sphereBuildInput.flags = &flags;

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
    return OptixAccelerationStructure {
        .data = (CUdeviceptr)d_output,
        .primitive_type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
        .handle = output
    };
}

__host__ OptixAccelerationStructure makeMeshGAS(
    OptixDeviceContext ctx,
    struct DeviceGeometryData geometry_data
) {
    OptixAccelBuildOptions accelOptions;
    OptixBuildInput buildInput;

    memset(&accelOptions, 0, sizeof(accelOptions));
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;

    memset(&buildInput, 0, sizeof(buildInput));

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray& triangleBuildInput = buildInput.triangleArray;
    triangleBuildInput.numSbtRecords = 1;

    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    triangleBuildInput.flags = &flags;

    // runtime / device APIs are interoperable, so this should "just work"
    triangleBuildInput.vertexBuffers = reinterpret_cast<const CUdeviceptr*>(&geometry_data.d_vertices);
    triangleBuildInput.numVertices = geometry_data.num_vertices;
    triangleBuildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleBuildInput.vertexStrideInBytes = sizeof(float) * 3;

    triangleBuildInput.indexBuffer = reinterpret_cast<CUdeviceptr>(geometry_data.d_tris);
    triangleBuildInput.numIndexTriplets = geometry_data.num_tris;
    triangleBuildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleBuildInput.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangleBuildInput.preTransform = 0;

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
    cudaFree(d_temp);

    // we need to return d_output and output together
    // since d_output backs the TraversableHandle
    return OptixAccelerationStructure {
        .data = (CUdeviceptr)d_output,
        .primitive_type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        .handle = output
    };
}

__host__ OptixAccelerationStructure makeIAS(
    OptixDeviceContext ctx,
    const OptixAccelerationStructure* instances,
    const Matrix4x4* transforms,
    const unsigned int* sbtOffsets,
    size_t len
) {
    std::vector<OptixInstance> instanceBuffer;
    instanceBuffer.reserve(len);

    for (size_t i = 0; i < len; i += 1) {
        OptixInstance instance;
        instance.instanceId = 0; /* we don't use this field, even if there is instancing happening */
        instance.visibilityMask = 255;

        instance.sbtOffset = sbtOffsets[i];

        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = instances[i].handle;
        memcpy(
            instance.transform,
            transforms[i].m,
            sizeof(float) * 12
        );
        instanceBuffer.push_back(instance);
    }

    void* d_instances;
    cudaMalloc(&d_instances, sizeof(OptixInstance) * len);
    cudaMemcpy(d_instances, instanceBuffer.data(), sizeof(OptixInstance) * len, cudaMemcpyHostToDevice);

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
    instanceBuildInput.numInstances = len;

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
    return OptixAccelerationStructure {
        .data = (CUdeviceptr)d_output,
        .primitive_type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        .handle = output
    };
}

// TODO: get rid of this nonsense
__host__ OptixAabb getAabb(OptixDeviceContext ctx, OptixTraversableHandle as) {
    // this is massively overkill / wasteful to have a stream and malloc just to get the AABB, but it should only happen once
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    OptixAabb *d_aabb;
    OptixAabb h_aabb;
    cudaMalloc(&d_aabb, sizeof(OptixAabb));
    OptixAccelEmitDesc emit_desc = {
        .result = reinterpret_cast<CUdeviceptr>(d_aabb),
        .type = OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_AABBS
    };
    optixAccelEmitProperty(ctx, stream, as, &emit_desc);
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_aabb, d_aabb, sizeof(OptixAabb), cudaMemcpyDeviceToHost);
    cudaFree(d_aabb);
    cudaStreamDestroy(stream);

    return h_aabb;
}
