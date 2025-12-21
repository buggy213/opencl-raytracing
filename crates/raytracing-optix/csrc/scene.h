#pragma once

#include "optix_types.h"

std::pair<OptixTraversableHandle, CUdeviceptr> makeSphereGAS(
    OptixDeviceContext ctx,
    const float *center,
    float radius
);

std::pair<OptixTraversableHandle, CUdeviceptr> makeMeshGAS(
    OptixDeviceContext ctx,
    const float* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const unsigned int* tris, /* packed */
    size_t trisLen, /* number of uint3's */
    const float* transform /* 4x4 row-major */
);

std::pair<OptixTraversableHandle, CUdeviceptr> makeIAS(
    OptixDeviceContext ctx,
    const OptixTraversableHandle* traversableHandles,
    size_t traversableHandlesLen
);
