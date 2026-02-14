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
    DeviceGeometryData geometryData
);

OptixAccelerationStructure makeIAS(
    OptixDeviceContext ctx,
    const OptixAccelerationStructure* instances,
    const Matrix4x4* transforms,
    const unsigned int* sbtOffsets,
    size_t len
);

OptixAabb getAabb(OptixDeviceContext ctx, OptixTraversableHandle as);