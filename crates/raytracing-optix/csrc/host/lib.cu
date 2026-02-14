#include "util.hpp"

#include <array>
#include <cassert>

#include "lib_api.h"

#include <optix_stubs.h>

#include "scene.hpp"
#include "pipeline.hpp"
#include "sbt_host.hpp"
#include "texture.hpp"

__host__ void optixLogCallback(unsigned int level, const char* tag, const char* msg, void *cbdata) {
    printf("[%s] %s\n", tag, msg);
}

RT_API __host__ OptixDeviceContext initOptix(bool debug) {
    CUDA_CHECK(cudaSetDevice(0));

    OPTIX_CHECK(optixInit());

    OptixDeviceContext optix_ctx;
    OptixDeviceContextOptions options = {};
    if (debug) {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        // all messages
        options.logCallbackLevel = 4;
        options.logCallbackData = nullptr;
        options.logCallbackFunction = &optixLogCallback;

    }
    else {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        options.logCallbackLevel = 0;
        options.logCallbackData = nullptr;
        options.logCallbackFunction = nullptr;
    }

    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &optix_ctx));

    return optix_ctx;
}

RT_API __host__ void destroyOptix(OptixDeviceContext ctx) {
    OPTIX_CHECK(optixDeviceContextDestroy(ctx));
}

RT_API __host__ struct DeviceGeometryData uploadGeometryData(struct HostGeometryData data) {
    if (data.kind != GeometryKind::TRIANGLE) {
        assert(false); // don't use this for non-triangle data
    }

    void* d_vertices;
    void* d_tris;

    cudaMalloc(&d_vertices, sizeof(Vec3) * data.num_vertices);
    cudaMalloc(&d_tris, sizeof(Vec3u) * data.num_tris);
    cudaMemcpy(d_vertices, data.vertices, sizeof(Vec3) * data.num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tris, data.tris, sizeof(Vec3u) * data.num_tris, cudaMemcpyHostToDevice);

    void* d_normals = nullptr;
    if (data.normals) {
        cudaMalloc(&d_normals, data.num_vertices * sizeof(Vec3));
        cudaMemcpy(d_normals, data.normals, data.num_vertices * sizeof(Vec3), cudaMemcpyHostToDevice);
    }

    void* d_uvs = nullptr;
    if (data.uvs) {
        cudaMalloc(&d_uvs, data.num_vertices * sizeof(Vec2));
        cudaMemcpy(d_uvs, data.uvs, data.num_vertices * sizeof(Vec2), cudaMemcpyHostToDevice);
    }

    return DeviceGeometryData {
        data.kind,
        data.num_tris,
        d_tris,
        data.num_vertices,
        d_vertices,
        d_normals,
        d_uvs
    };
}

RT_API __host__ OptixAccelerationStructure makeSphereAccelerationStructure(
    const OptixDeviceContext ctx,
    const Vec3 center,
    const float radius
) {
    return makeSphereGAS(ctx, center, radius);
}

RT_API __host__ OptixAccelerationStructure makeMeshAccelerationStructure(
    OptixDeviceContext ctx,
    struct DeviceGeometryData geometry_data
) {
    return makeMeshGAS(
        ctx,
        geometry_data
    );
}

RT_API __host__ OptixAccelerationStructure makeInstanceAccelerationStructure(
    OptixDeviceContext ctx,
    const OptixAccelerationStructure* instances,
    const struct Matrix4x4* transforms,
    const unsigned int* sbtOffsets,
    size_t len
) {
    return makeIAS(
        ctx,
        instances,
        transforms,
        sbtOffsets,
        len
    );
}

RT_API __host__ AovPipelineWrapper makeAovPipeline(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize
) {
    AovPipeline aovPipeline = makeAovPipelineImpl(ctx, progData, progSize);
    return new AovPipeline(aovPipeline);
}

RT_API __host__ AovSbtWrapper makeAovSbt() {
    return new AovSbt();
}

RT_API __host__ size_t addHitRecordAovSbt(AovSbtWrapper sbt, DeviceGeometryData geometryData) {
    return sbt->addHitgroupRecord(geometryData);
}

RT_API __host__ void finalizeAovSbt(AovSbtWrapper sbt, AovPipelineWrapper pipeline) {
    sbt->finalize(*pipeline);
}

RT_API __host__ void releaseAovSbt(AovSbtWrapper sbt) {
    delete sbt;
}


RT_API __host__ void launchAovPipeline(
    AovPipelineWrapper pipeline,
    AovSbtWrapper sbt,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    struct Vec3* normals
) {
    launchAovPipelineImpl(
        *pipeline,
        *sbt,
        camera,
        rootHandle,
        normals
    );
}

RT_API __host__ void releaseAovPipeline(AovPipelineWrapper pipeline) {
    delete pipeline;
}

RT_API __host__ PathtracerPipelineWrapper makePathtracerPipeline(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize
) {
    PathtracerPipeline pathtracerPipeline = makePathtracerPipelineImpl(ctx, progData, progSize);
    return new PathtracerPipeline(pathtracerPipeline);
}

RT_API __host__ PathtracerSbtWrapper makePathtracerSbt() {
    return new PathtracerSbt;
}

RT_API __host__ size_t addHitRecordPathtracerSbt(
    PathtracerSbtWrapper sbt,
    DeviceGeometryData geometryData,
    Material material,
    // -1 = no light
    int area_light
) {
    std::optional<unsigned int> area_light_opt;
    if (area_light == -1) {
        area_light_opt = std::nullopt;
    } else {
        area_light_opt = area_light;
    }

    return sbt->addHitgroupRecord(geometryData, material, area_light_opt);
}

RT_API __host__ void finalizePathtracerSbt(PathtracerSbtWrapper sbt, PathtracerPipelineWrapper pipeline) {
    sbt->finalize(*pipeline);
}


RT_API __host__ void releasePathtracerSbt(PathtracerSbtWrapper sbt) {
    delete sbt;
}

RT_API __host__ void launchPathtracerPipeline(
    PathtracerPipelineWrapper pipeline,
    PathtracerSbtWrapper sbt,
    struct OptixRaytracerSettings settings,
    struct Scene scene,
    OptixTraversableHandle rootHandle,
    struct Vec4* radiance
) {
    launchPathtracerPipelineImpl(
        *pipeline,
        *sbt,
        settings,
        scene,
        rootHandle,
        radiance
    );
}

RT_API __host__ void releasePathtracerPipeline(PathtracerPipelineWrapper pipeline) {
    delete pipeline;
}

RT_API __host__ struct CudaArray makeCudaArray(const void *src, size_t pitch, size_t width, size_t height, enum TextureFormat fmt) {
    cudaArray_t cudaArray = createCudaArray(src, pitch, width, height, fmt);
    return CudaArray { .d_array = cudaArray };
}

RT_API __host__ struct CudaTextureObject makeCudaTexture(struct CudaArray backing_array, struct TextureSampler sampler) {
    cudaTextureObject_t cudaTexture = createCudaTexture(static_cast<cudaArray_t>(backing_array.d_array), sampler);
    return CudaTextureObject { .handle = cudaTexture };
}
