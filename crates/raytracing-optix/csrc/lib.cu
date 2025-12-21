#include "lib.h"

#include <array>

#include "lib_api.h"

#include <optix_stubs.h>

#include "scene.h"

RT_API __host__ OptixDeviceContext initOptix() {
    CUDA_CHECK(cudaSetDevice(0));

    OPTIX_CHECK(optixInit());

    OptixDeviceContext optix_ctx;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &optix_ctx));

    return optix_ctx;
}

RT_API __host__ void destroyOptix(OptixDeviceContext ctx) {
    OPTIX_CHECK(optixDeviceContextDestroy(ctx));
}

RT_API __host__ OptixAccelerationStructure makeSphereAccelerationStructure(
    OptixDeviceContext ctx,
    Vec3 center,
    float radius
) {
    std::array<float, 3> centerArray {center.x, center.y, center.z};
    auto sphereGAS = makeSphereGAS(ctx, centerArray.data(), radius);
    return OptixAccelerationStructure {
        .data = sphereGAS.second,
        .handle = sphereGAS.first,
    };
}

RT_API __host__ OptixAccelerationStructure makeMeshAccelerationStructure(
    OptixDeviceContext ctx,
    const float* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const unsigned int* tris, /* packed */
    size_t trisLen, /* number of uint3's */
    const float* transform /* 4x4 row-major */
) {
    auto meshGAS = makeMeshGAS(
        ctx,
        vertices,
        verticesLen,
        tris,
        trisLen,
        transform
    );

    return OptixAccelerationStructure {
        .data = meshGAS.second,
        .handle = meshGAS.first,
    };
}

RT_API __host__ OptixAccelerationStructure makeInstanceAccelerationStructure(
    OptixDeviceContext ctx,
    const OptixTraversableHandle* traversableHandles,
    size_t traversableHandlesLen
) {
    auto IAS = makeIAS(
        ctx,
        traversableHandles,
        traversableHandlesLen
    );

    return OptixAccelerationStructure {
        .data = IAS.second,
        .handle = IAS.first
    };
}