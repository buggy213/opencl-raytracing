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
    struct Vec3 *tris;
    size_t num_vertices;
    struct Vec3 *normals;
    struct Vec2 *uvs;
};