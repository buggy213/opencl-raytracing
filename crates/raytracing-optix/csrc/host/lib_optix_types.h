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

// TODO: this seems like a pretty bad api wart; it'd be nice to have triangles not be 
//       a special case, and to use float3 / uint3 types in DeviceGeometryData definition
struct HostGeometryData {
    enum GeometryKind kind;

    // only valid for triangle meshes
    size_t num_tris;
    const struct Vec3u *tris;

    size_t num_vertices;
    const struct Vec3 *vertices;
    const struct Vec3 *normals;
    const struct Vec2 *uvs;
};

struct DeviceGeometryData {
    enum GeometryKind kind;

    // only valid for triangle meshes
    size_t num_tris;
    const void *d_tris;

    size_t num_vertices;
    const void *d_vertices;
    const void *d_normals;
    const void *d_uvs;
};

struct CudaArray {
    void *d_array;
};

struct CudaTextureObject {
    unsigned long long handle;
};
