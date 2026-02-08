#pragma once

/*
 * SBT record definitions, shared between host/device C++ code
 */

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
        uint3* indices;
        float3* normals;
        float2* uvs;
    } mesh_data;

    struct Material material_data;
};