#include "sbt_host.h"

#include <cuda.h>
#include <optix_stubs.h>

#include "pipeline.h"
#include "sbt.h"

__host__ size_t AovSbt::addHitgroupRecord(GeometryData geometryData) {
    payloads.push_back(geometryData);
    return 1;
}

// @perf: lots of allocations and copies, can definitely be optimized
__host__ void AovSbt::finalize(AovPipeline& pipeline) {
    std::vector<HitgroupRecord> hitgroupRecords;
    hitgroupRecords.reserve(payloads.size());

    for (HitgroupRecordPayload& payload : payloads) {
        void* d_tris;
        cudaMalloc(&d_tris, payload.num_tris * sizeof(uint3));
        cudaMemcpy(d_tris, payload.tris, payload.num_tris * sizeof(uint3), cudaMemcpyHostToDevice);

        void* d_normals = nullptr;
        if (payload.normals) {
            cudaMalloc(&d_normals, payload.num_vertices * sizeof(float3));
            cudaMemcpy(d_normals, payload.normals, payload.num_vertices * sizeof(float3), cudaMemcpyHostToDevice);
        }

        void* d_uvs = nullptr;
        if (payload.uvs) {
            cudaMalloc(&d_uvs, payload.num_vertices * sizeof(float2));
            cudaMemcpy(d_uvs, payload.uvs, payload.num_vertices * sizeof(float2), cudaMemcpyHostToDevice);
        }

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
            .indices = (uint3*)d_tris,
            .normals = (float3*)d_normals,
            .uvs = (float2*)d_uvs
        };

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

__host__ size_t PathtracerSbt::addHitgroupRecord(GeometryData geometryData, Material material) {
    payloads.push_back(StagedHitgroupRecord {
        .geometry = geometryData,
        .material = material
    });

    return 2;
}


__host__ void PathtracerSbt::finalize(PathtracerPipeline &pipeline) {
    std::vector<HitgroupRecord> hitgroupRecords;
    hitgroupRecords.reserve(2 * payloads.size());

    for (StagedHitgroupRecord& payload : payloads) {
        void* d_tris;
        cudaMalloc(&d_tris, payload.geometry.num_tris * sizeof(uint3));
        cudaMemcpy(d_tris, payload.geometry.tris, payload.geometry.num_tris * sizeof(uint3), cudaMemcpyHostToDevice);

        void* d_normals = nullptr;
        if (payload.geometry.normals) {
            cudaMalloc(&d_normals, payload.geometry.num_vertices * sizeof(float3));
            cudaMemcpy(d_normals, payload.geometry.normals, payload.geometry.num_vertices * sizeof(float3), cudaMemcpyHostToDevice);
        }

        void* d_uvs = nullptr;
        if (payload.geometry.uvs) {
            cudaMalloc(&d_uvs, payload.geometry.num_vertices * sizeof(float2));
            cudaMemcpy(d_uvs, payload.geometry.uvs, payload.geometry.num_vertices * sizeof(float2), cudaMemcpyHostToDevice);
        }

        auto geometryType = static_cast<PathtracerPipeline::GeometryType>(payload.geometry.kind);
        auto materialType = static_cast<PathtracerPipeline::MaterialType>(payload.material.kind);

        HitgroupRecord radianceHitgroupRecord = {};
        OptixProgramGroup radianceHit = pipeline.radianceHitProgram(geometryType, materialType);
        radianceHitgroupRecord.mesh_data = {
            .indices = (uint3*)d_tris,
            .normals = (float3*)d_normals,
            .uvs = (float2*)d_uvs
        };
        radianceHitgroupRecord.material_data = payload.material;
        optixSbtRecordPackHeader(radianceHit, &radianceHitgroupRecord);
        hitgroupRecords.push_back(radianceHitgroupRecord);

        HitgroupRecord shadowHitgroupRecord = {};
        OptixProgramGroup shadowHit = pipeline.shadowHitProgram(geometryType);
        shadowHitgroupRecord.mesh_data = {
            .indices = (uint3*)d_tris,
            .normals = (float3*)d_normals,
            .uvs = (float2*)d_uvs
        };
        shadowHitgroupRecord.material_data = payload.material;
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

    void* d_missRecord;
    cudaMalloc(&d_missRecord, 2 * sizeof(MissRecord));
    cudaMemcpy(d_missRecord, &radianceMissRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missRecord + sizeof(MissRecord), &shadowMissRecord, sizeof(MissRecord), cudaMemcpyHostToDevice);

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




