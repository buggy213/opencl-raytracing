#pragma once

#define CCCL_DISABLE_INT128_SUPPORT
#include <cuda/std/utility>
#include "kernel_math.h"

namespace geometry
{

// @raytracing_cpu::geometry::make_orthonormal_basis
inline __device__ cuda::std::pair<float3, float3> make_orthonormal_basis(float3 z)
{
    float3 a = fabs(z.z) < 0.8f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, 1.0f, 0.0f);
    float3 x = normalize(cross(a, z));
    float3 y = cross(z, x);

    return cuda::std::make_pair(x, y);
}

}