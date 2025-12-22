#pragma once

#include <optix_types.h>

/* Vocabulary types */
struct Vec3 {
    float x;
    float y;
    float z;
};

struct Vec3u {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct Matrix4x4 {
    float m[16];
};

struct OptixAccelerationStructure {
    CUdeviceptr data;
    OptixTraversableHandle handle;
};