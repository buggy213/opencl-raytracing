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

struct Vec2u {
    unsigned int x;
    unsigned int y;
};

struct Matrix4x4 {
    float m[16];
};

struct OptixAccelerationStructure {
    CUdeviceptr data;
    OptixTraversableHandle handle;
};

struct OptixPipelineWrapper {
    OptixPipeline pipeline;
    OptixProgramGroup raygenProgram;
    OptixProgramGroup missProgram;
};