#pragma once

/*
 * Types shared between Rust/C++ to manage OptiX runtime state and configuration
 */

#include <optix_types.h>

struct OptixAccelerationStructure {
    CUdeviceptr data;
    OptixBuildInputType primitive_type;
    OptixTraversableHandle handle;
};

// Opaque pointers to C++ types
typedef struct AovPipeline* AovPipelineWrapper;
typedef struct PathtracerPipeline* PathtracerPipelineWrapper;
typedef struct AovSbt* AovSbtWrapper;
typedef struct PathtracerSbt* PathtracerSbtWrapper;

struct GeometryData {
    enum GeometryKind { TRIANGLE, SPHERE } kind;

    size_t num_tris;
    const struct Vec3u *tris;

    size_t num_vertices;
    const struct Vec3 *normals;
    const struct Vec2 *uvs;
};