#pragma once

#include "shared_lib.h"
#include <optix_types.h>

/* Vocabulary types */
struct Vec3 {
    float x;
    float y;
    float z;
};

/* Context management */
RT_API OptixDeviceContext initOptix();
RT_API void destroyOptix(OptixDeviceContext ctx);

/* Scene conversion functions */
struct OptixAccelerationStructure {
    CUdeviceptr data;
    OptixTraversableHandle handle;
};

RT_API struct OptixAccelerationStructure makeSphereAccelerationStructure(
    OptixDeviceContext ctx,
    struct Vec3 center,
    float radius
);

RT_API struct OptixAccelerationStructure makeMeshAccelerationStructure(
    OptixDeviceContext context,
    const float* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const unsigned int* tris, /* packed */
    size_t trisLen, /* number of uint3's */
    const float* transform /* 4x4 row-major */
);


RT_API struct OptixAccelerationStructure makeInstanceAccelerationStructure(
    OptixDeviceContext context,
    const OptixTraversableHandle* traversableHandles,
    size_t traversableHandlesLen
);