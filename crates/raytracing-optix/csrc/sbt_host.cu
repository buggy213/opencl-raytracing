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

        void* d_normals;
        cudaMalloc(&d_normals, payload.num_vertices * sizeof(float3));
        cudaMemcpy(d_normals, payload.normals, payload.num_vertices * sizeof(float3), cudaMemcpyHostToDevice);

        void* d_uvs;
        cudaMalloc(&d_uvs, payload.num_vertices * sizeof(float2));
        cudaMemcpy(d_uvs, payload.uvs, payload.num_vertices * sizeof(float2), cudaMemcpyHostToDevice);

        HitgroupRecord hitgroupRecord;
        switch (payload.kind) {
            case GeometryData::TRIANGLE:
                optixSbtRecordPackHeader(pipeline.triHitProgramGroup, &hitgroupRecord);
                break;
            case GeometryData::SPHERE:
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



