#include "util.h"

#include <array>

#include "lib_api.h"

#include <optix_stubs.h>

#include "scene.h"
#include "pipeline.h"

RT_API __host__ OptixDeviceContext initOptix() {
    CUDA_CHECK(cudaSetDevice(0));

    OPTIX_CHECK(optixInit());

    OptixDeviceContext optix_ctx;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &optix_ctx));

    int driverVersion = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    fprintf(stderr, "Driver version: %d\n", driverVersion);

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

RT_API __host__ OptixPipeline makeBasicPipeline(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize
) {
    return makeBasicPipelineImpl(ctx, progData, progSize);
}