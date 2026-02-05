#pragma once

#include <optix_types.h>
#include <vector>

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

    union MaterialData {
        struct DiffuseMaterialData {
            float3 albedo;
        } diffuse_material_data;
    } material_data;
};

struct HostSbt {
    RaygenRecord raygen_record;
    MissRecord miss_record;
    std::vector<HitgroupRecord> hitgroup_records;
};
