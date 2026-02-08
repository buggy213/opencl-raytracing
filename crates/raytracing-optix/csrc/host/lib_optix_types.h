#pragma once

/*
 * Types shared between Rust/C++ to manage CUDA/OptiX runtime state and configuration
 */

#include "types.h"
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


#ifdef __cplusplus
enum class GeometryKind { TRIANGLE, SPHERE };
using MaterialKind = Material::MaterialKind;
#else
enum GeometryKind { TRIANGLE, SPHERE };
#endif


struct GeometryData {
    enum GeometryKind kind;

    size_t num_tris;
    const struct Vec3u *tris;

    size_t num_vertices;
    const struct Vec3 *normals;
    const struct Vec2 *uvs;
};

struct CudaArray {
    void *d_array;
};

struct CudaTextureObject {
    unsigned long long handle;
};
