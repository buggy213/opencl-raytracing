#pragma once

#include "kernel_math.hpp"

struct Ray {
    float3 origin;
    float3 direction;

    __device__ float3 at(float t) const {
        return origin + t * direction;
    }
};

// sue me
typedef unsigned int u32;
typedef unsigned long long u64;
typedef int i32;
typedef long long i64;