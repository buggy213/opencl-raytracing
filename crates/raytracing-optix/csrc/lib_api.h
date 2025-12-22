#pragma once

#include "shared_lib.h"
#include "lib_types.h"
#include <optix_types.h>

/* Context management */
RT_API OptixDeviceContext initOptix();
RT_API void destroyOptix(OptixDeviceContext ctx);

/* Scene conversion functions */
RT_API struct OptixAccelerationStructure makeSphereAccelerationStructure(
    OptixDeviceContext ctx,
    struct Vec3 center,
    float radius
);

RT_API struct OptixAccelerationStructure makeMeshAccelerationStructure(
    OptixDeviceContext ctx,
    const struct Vec3* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const struct Vec3u* tris, /* packed */
    size_t trisLen /* number of uint3's */
);

RT_API struct OptixAccelerationStructure makeInstanceAccelerationStructure(
    OptixDeviceContext ctx,
    const struct OptixAccelerationStructure* instances,
    const struct Matrix4x4* transforms,
    size_t len
);
