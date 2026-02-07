#include "util.h"

#include <array>

#include "lib_api.h"

#include <optix_stubs.h>

#include "scene.h"
#include "pipeline.h"
#include "sbt_host.h"

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
    AovPipeline aovPipeline = makeAovPipelineImpl(ctx, progData, progSize);
    return new AovPipeline(aovPipeline);
}

RT_API __host__ AovSbtWrapper makeAovSbt() {
    return new AovSbt();
}

RT_API __host__ void addHitRecordAovSbt(AovSbtWrapper sbt, GeometryData geometryData) {
    sbt->addHitgroupRecord(geometryData);
}

RT_API __host__ void finalizeAovSbt(AovSbtWrapper sbt, AovPipelineWrapper pipeline) {
    sbt->finalize(*pipeline);
}

RT_API __host__ void releaseAovSbt(AovSbtWrapper sbt) {
    delete sbt;
}


RT_API __host__ void launchAovPipeline(
    AovPipelineWrapper pipeline,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    struct Vec3* normals
) {
    launchAovPipelineImpl(
        *pipeline,
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
    size_t progSize,
    unsigned int maxRayDepth
) {
    PathtracerPipeline pathtracerPipeline = makePathtracerPipelineImpl(ctx, progData, progSize, maxRayDepth);
    return new PathtracerPipeline(pathtracerPipeline);
}

PathtracerSbtWrapper makePathtracerSbt() {
    return new PathtracerSbt;
}

void addHitRecordPathtracerSbt(PathtracerSbtWrapper sbt, GeometryData geometryData) {

}

void finalizePathtracerSbt(PathtracerSbtWrapper sbt, PathtracerPipelineWrapper pipeline) {
    sbt->finalize(*pipeline);
}


void releasePathtracerSbt(PathtracerSbtWrapper sbt) {
    delete sbt;
}

RT_API __host__ void launchPathtracerPipeline(
    PathtracerPipelineWrapper pipeline,
    const Scene* scene,
    OptixTraversableHandle rootHandle,
    struct Vec3* radiance
) {
    launchPathtracerPipelineImpl(
        *pipeline,
        scene,
        rootHandle,
        radiance
    );
}

RT_API __host__ void releasePathtracerPipeline(PathtracerPipelineWrapper pipeline) {
    delete pipeline;
}
