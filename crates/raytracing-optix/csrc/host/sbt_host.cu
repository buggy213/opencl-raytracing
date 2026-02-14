#include "sbt_host.hpp"

#include <cuda.h>
#include <optix_stubs.h>

#include "pipeline.hpp"
#include "sbt.hpp"

__host__ size_t AovSbt::addHitgroupRecord(DeviceGeometryData geometryData) {
    payloads.push_back(geometryData);
    return 1;
}

// @perf: lots of allocations and copies, can definitely be optimized
__host__ void AovSbt::finalize(AovPipeline& pipeline) {
    std::vector<HitgroupRecord> hitgroupRecords;
    hitgroupRecords.reserve(payloads.size());

    for (HitgroupRecordPayload& payload : payloads) {
        HitgroupRecord hitgroupRecord = {};
        switch (payload.kind) {
            case GeometryKind::TRIANGLE:
                optixSbtRecordPackHeader(pipeline.triHitProgramGroup, &hitgroupRecord);
                break;
            case GeometryKind::SPHERE:
                optixSbtRecordPackHeader(pipeline.sphereHitProgramGroup, &hitgroupRecord);
                break;
        }
        hitgroupRecord.mesh_data = {
            .num_tris = payload.num_tris,
            .vertices = (float3*)payload.d_vertices,
            .indices = (uint3*)payload.d_tris,
            .normals = (float3*)payload.d_normals,
            .uvs = (float2*)payload.d_uvs
        };
        hitgroupRecord.area_light = cuda::std::nullopt;

        hitgroupRecords.push_back(hitgroupRecord);
    }

    void* d_hitgroupRecords;
    cudaMalloc(&d_hitgroupRecords, hitgroupRecords.size() * sizeof(HitgroupRecord));
    cudaMemcpy(d_hitgroupRecords, hitgroupRecords.data(), hitgroupRecords.size() * sizeof(HitgroupRecord), cudaMemcpyHostToDevice);

    RaygenRecord raygenRecord;
    optixSbtRecordPackHeader(pipeline.raygenProgram, &raygenRecord);

    void* d_raygenRecord;
    cudaMalloc(&d_raygenRecord, sizeof(RaygenRecord));
    cudaMemcpy(d_raygenRecord, &raygenRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    MissRecord missRecord;
    optixSbtRecordPackHeader(pipeline.missProgram, &missRecord);

    void* d_missRecord;
    cudaMalloc(&d_missRecord, sizeof(MissRecord));
    cudaMemcpy(d_missRecord, &missRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);

    sbt = {
        .raygenRecord = (CUdeviceptr)d_raygenRecord,
        .exceptionRecord = (CUdeviceptr)nullptr,
        .missRecordBase = (CUdeviceptr)d_missRecord,
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = 1,
        .hitgroupRecordBase = (CUdeviceptr)d_hitgroupRecords,
        .hitgroupRecordStrideInBytes = sizeof(HitgroupRecord),
        .hitgroupRecordCount = static_cast<unsigned int>(hitgroupRecords.size()),
        .callablesRecordBase = (CUdeviceptr)nullptr,
        .callablesRecordStrideInBytes = 0,
        .callablesRecordCount = 0
    };

    payloads.clear();
}

__host__ AovSbt::~AovSbt() {
    cudaFree((void*)sbt.raygenRecord);
    cudaFree((void*)sbt.callablesRecordBase);
    cudaFree((void*)sbt.exceptionRecord);
    cudaFree((void*)sbt.hitgroupRecordBase);
    cudaFree((void*)sbt.missRecordBase);
}

__host__ size_t PathtracerSbt::addHitgroupRecord(DeviceGeometryData geometryData, Material material, std::optional<unsigned int> area_light) {
    payloads.push_back(StagedHitgroupRecord {
        .geometry = geometryData,
        material,
        area_light
    });

    return 2;
}


__host__ void PathtracerSbt::finalize(PathtracerPipeline &pipeline) {
    std::vector<HitgroupRecord> hitgroupRecords;
    hitgroupRecords.reserve(2 * payloads.size());

    for (StagedHitgroupRecord& payload : payloads) {
        auto geometryType = static_cast<PathtracerPipeline::GeometryType>(payload.geometry.kind);
        auto materialType = static_cast<PathtracerPipeline::MaterialType>(payload.material.kind);

        HitgroupRecord radianceHitgroupRecord = {};
        OptixProgramGroup radianceHit = pipeline.radianceHitProgram(geometryType, materialType);
        radianceHitgroupRecord.mesh_data = {
            .num_tris = payload.geometry.num_tris,
            .vertices = (float3*)payload.geometry.d_vertices,
            .indices = (uint3*)payload.geometry.d_tris,
            .normals = (float3*)payload.geometry.d_normals,
            .uvs = (float2*)payload.geometry.d_uvs
        };
        radianceHitgroupRecord.material_data = payload.material;
        if (payload.area_light) {
            radianceHitgroupRecord.area_light = *payload.area_light;
        } else {
            radianceHitgroupRecord.area_light = cuda::std::nullopt;
        }
        optixSbtRecordPackHeader(radianceHit, &radianceHitgroupRecord);
        hitgroupRecords.push_back(radianceHitgroupRecord);

        HitgroupRecord shadowHitgroupRecord = {};
        OptixProgramGroup shadowHit = pipeline.shadowHitProgram(geometryType);
        shadowHitgroupRecord.mesh_data = radianceHitgroupRecord.mesh_data;
        shadowHitgroupRecord.material_data = radianceHitgroupRecord.material_data;
        shadowHitgroupRecord.area_light = cuda::std::nullopt;
        optixSbtRecordPackHeader(shadowHit, &shadowHitgroupRecord);
        hitgroupRecords.push_back(shadowHitgroupRecord);
    }

    void* d_hitgroupRecords;
    cudaMalloc(&d_hitgroupRecords, hitgroupRecords.size() * sizeof(HitgroupRecord));
    cudaMemcpy(d_hitgroupRecords, hitgroupRecords.data(), hitgroupRecords.size() * sizeof(HitgroupRecord), cudaMemcpyHostToDevice);

    RaygenRecord raygenRecord;
    optixSbtRecordPackHeader(pipeline.raygenProgram, &raygenRecord);

    void* d_raygenRecord;
    cudaMalloc(&d_raygenRecord, sizeof(RaygenRecord));
    cudaMemcpy(d_raygenRecord, &raygenRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    MissRecord radianceMissRecord;
    optixSbtRecordPackHeader(pipeline.missProgram(PathtracerPipeline::RADIANCE), &radianceMissRecord);

    MissRecord shadowMissRecord;
    optixSbtRecordPackHeader(pipeline.missProgram(PathtracerPipeline::SHADOW), &shadowMissRecord);

    MissRecord* d_missRecord;
    cudaMalloc(&d_missRecord, 2 * sizeof(MissRecord));
    cudaMemcpy(&d_missRecord[0], &radianceMissRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_missRecord[1], &shadowMissRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);

    sbt = {
        .raygenRecord = (CUdeviceptr)d_raygenRecord,
        .exceptionRecord = (CUdeviceptr)nullptr,
        .missRecordBase = (CUdeviceptr)d_missRecord,
        .missRecordStrideInBytes = sizeof(MissRecord),
        .missRecordCount = 2,
        .hitgroupRecordBase = (CUdeviceptr)d_hitgroupRecords,
        .hitgroupRecordStrideInBytes = sizeof(HitgroupRecord),
        .hitgroupRecordCount = static_cast<unsigned int>(hitgroupRecords.size()),
        .callablesRecordBase = (CUdeviceptr)nullptr,
        .callablesRecordStrideInBytes = 0,
        .callablesRecordCount = 0
    };

    payloads.clear();
}

__host__ PathtracerSbt::~PathtracerSbt() {
    cudaFree((void*)sbt.raygenRecord);
    cudaFree((void*)sbt.callablesRecordBase);
    cudaFree((void*)sbt.exceptionRecord);
    cudaFree((void*)sbt.hitgroupRecordBase);
    cudaFree((void*)sbt.missRecordBase);
}




