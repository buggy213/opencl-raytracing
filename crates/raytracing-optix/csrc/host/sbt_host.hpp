#pragma once

#include <optix_types.h>
#include <vector>

#include "sbt_host.hpp"
#include "types.h"
#include "lib_optix_types.h"

// AOV shader binding table layout: 1 raygen record, 1 miss record (sets all AOV to default / invalid)
// 1 hitgroup record per primitive. points to appropriate program group based on geometry type
struct AovSbt {
    using HitgroupRecordPayload = GeometryData;
    std::vector<HitgroupRecordPayload> payloads;

    OptixShaderBindingTable sbt;

    size_t addHitgroupRecord(GeometryData geometryData);
    void finalize(AovPipeline& pipeline);
    ~AovSbt();
};

// Pathtracer shader binding table layout: 1 raygen record, 2 miss records (one for radiance rays, one for shadow rays)
// 2 hitgroup records per primitive, one for shadow rays and one for radiance rays
struct PathtracerSbt {
    struct StagedHitgroupRecord {
        GeometryData geometry;
        Material material;
    };

    std::vector<StagedHitgroupRecord> payloads;

    OptixShaderBindingTable sbt;

    size_t addHitgroupRecord(GeometryData geometryData, Material material);
    void finalize(PathtracerPipeline& pipeline);
    ~PathtracerSbt();
};