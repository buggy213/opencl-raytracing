#pragma once

#define CCCL_DISABLE_INT128_SUPPORT
#include <cuda/std/utility>
#include "kernel_math.hpp"

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

inline __device__ float tri_area(float3 p0, float3 p1, float3 p2)
{
    return length(cross(p1 - p0, p2 - p0)) / 2.0f;
}

}