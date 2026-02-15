#pragma once

/*
 * SBT record definitions, shared between host/device C++ code
 */

#include <cuda/std/optional>
#include <optix_types.h>

struct RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct HitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    struct MeshData {
        size_t num_tris;
        float3* vertices;
        uint3* indices;
        float3* normals;
        float2* uvs;
    } mesh_data;

    Material material_data;
    cuda::std::optional<unsigned int> area_light;
};