#include "util.h"

#include <array>

#include "lib_api.h"

#include <optix_stubs.h>

#include "scene.h"
#include "pipeline.h"

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

RT_API __host__ OptixAccelerationStructure makeSphereAccelerationStructure(
    const OptixDeviceContext ctx,
    const Vec3 center,
    const float radius
) {
    return makeSphereGAS(ctx, center, radius);
}

RT_API __host__ OptixAccelerationStructure makeMeshAccelerationStructure(
    OptixDeviceContext ctx,
    const struct Vec3* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const struct Vec3u* tris, /* packed */
    size_t trisLen /* number of uint3's */
) {
    return makeMeshGAS(
        ctx,
        vertices,
        verticesLen,
        tris,
        trisLen
    );
}

RT_API __host__ OptixAccelerationStructure makeInstanceAccelerationStructure(
    OptixDeviceContext ctx,
    const OptixAccelerationStructure* instances,
    const struct Matrix4x4* transforms,
    size_t len
) {
    return makeIAS(
        ctx,
        instances,
        transforms,
        len
    );
}

RT_API __host__ AovPipelineWrapper makeAovPipeline(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize
) {
    return makeAovPipelineImpl(ctx, progData, progSize);
}

RT_API __host__ void launchAovPipeline(
    AovPipelineWrapper pipelineWrapper,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    struct Vec3* normals
) {
    launchAovPipelineImpl(
        pipelineWrapper,
        camera,
        rootHandle,
        normals
    );
}