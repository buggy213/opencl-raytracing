#pragma once

#include "types.h"
#include "lib_optix_types.h"
#include "optix_types.h"

OptixAccelerationStructure makeSphereGAS(
    OptixDeviceContext ctx,
    Vec3 center,
    float radius
);

OptixAccelerationStructure makeMeshGAS(
    OptixDeviceContext ctx,
    const Vec3* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const Vec3u* tris, /* packed */
    size_t trisLen /* number of uint3's */
);

OptixAccelerationStructure makeIAS(
    OptixDeviceContext ctx,
    const OptixAccelerationStructure* instances,
    const Matrix4x4* transforms,
    const unsigned int* sbtOffsets,
    size_t len
);
