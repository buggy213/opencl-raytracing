#include "pipeline.h"

#include <cstdint>
#include <vector>

#include "lib_types.h"
#include "optix.h"
#include "util.h"

__host__ OptixPipelineWrapper makeBasicPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    moduleCompileOptions.numPayloadTypes = 0;
    moduleCompileOptions.payloadTypes = 0;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "pipeline_params";
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM |
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    OptixModule module = nullptr;
    OptixResult res = optixModuleCreate(
        ctx,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        (const char*)progData,
        progSize,
        nullptr,
        nullptr,
        &module
    );
    OPTIX_CHECK(res);

    OptixModule builtinModule = nullptr;
    OptixBuiltinISOptions builtinModuleOptions = {};
    builtinModuleOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    // need this flag in order to get the center in closest-hit program
    builtinModuleOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    builtinModuleOptions.curveEndcapFlags = 0;
    builtinModuleOptions.usesMotionBlur = false;

    res = optixBuiltinISModuleGet(
        ctx,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        &builtinModuleOptions,
        &builtinModule
    );
    OPTIX_CHECK(res);

    OptixProgramGroupDesc raygenGroupDesc = {};
    raygenGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenGroupDesc.raygen.module = module;
    raygenGroupDesc.raygen.entryFunctionName = "__raygen__debug";

    OptixProgramGroupOptions raygenGroupOptions = {};
    OptixProgramGroup raygenGroup = nullptr;
    res = optixProgramGroupCreate(
        ctx,
        &raygenGroupDesc,
        1,
        &raygenGroupOptions,
        nullptr,
        nullptr,
        &raygenGroup
    );
    OPTIX_CHECK(res);

    OptixProgramGroupDesc missGroupDesc = {};
    missGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missGroupDesc.miss.module = module;
    missGroupDesc.miss.entryFunctionName = "__miss__nop";

    OptixProgramGroupOptions missGroupOptions = {};
    OptixProgramGroup missGroup = nullptr;
    res = optixProgramGroupCreate(
        ctx,
        &missGroupDesc,
        1,
        &missGroupOptions,
        nullptr,
        nullptr,
        &missGroup
    );
    OPTIX_CHECK(res);

    OptixProgramGroupDesc hitGroupDesc = {};

    hitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitGroupDesc.hitgroup.moduleCH = module;
    hitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__normal";
    hitGroupDesc.hitgroup.moduleIS = builtinModule;
    hitGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
    hitGroupDesc.hitgroup.moduleAH = nullptr;
    hitGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

    OptixProgramGroupOptions hitGroupOptions = {};
    OptixProgramGroup hitGroup = nullptr;
    res = optixProgramGroupCreate(
        ctx,
        &hitGroupDesc,
        1,
        &hitGroupOptions,
        nullptr,
        nullptr,
        &hitGroup
    );
    OPTIX_CHECK(res);

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 4;

    OptixProgramGroup programGroups[3] = { raygenGroup, missGroup, hitGroup };
    res = optixPipelineCreate(
        ctx,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        3,
        nullptr,
        nullptr,
        &pipeline
    );
    OPTIX_CHECK(res);

    // note: we are leaking module / program group / pipeline at the moment
    // for one-off renders, this is fine
    return OptixPipelineWrapper {
        .pipeline = pipeline,
        .raygenProgram = raygenGroup,
        .missProgram = missGroup,
        .hitProgramGroup = hitGroup
    };
}

__host__ void launchBasicPipelineImpl(
    OptixPipelineWrapper pipelineWrapper
) {
    struct EmptyRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    EmptyRecord raygenRecord;
    EmptyRecord missRecord;
    EmptyRecord hitGroupRecord;
    optixSbtRecordPackHeader(pipelineWrapper.raygenProgram, &raygenRecord);
    optixSbtRecordPackHeader(pipelineWrapper.missProgram, &missRecord);
    optixSbtRecordPackHeader(pipelineWrapper.hitProgramGroup, &hitGroupRecord);

    void* d_raygenRecord;
    cudaMalloc(&d_raygenRecord, sizeof(EmptyRecord));
    cudaMemcpy(d_raygenRecord, &raygenRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    void* d_missRecord;
    cudaMalloc(&d_missRecord, sizeof(EmptyRecord));
    cudaMemcpy(d_missRecord, &missRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    void* d_hitGroupRecord;
    cudaMalloc(&d_hitGroupRecord, sizeof(EmptyRecord));
    cudaMemcpy(d_hitGroupRecord, &hitGroupRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = (CUdeviceptr)d_raygenRecord;

    sbt.missRecordBase = (CUdeviceptr)d_missRecord;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);

    sbt.hitgroupRecordBase = (CUdeviceptr)d_hitGroupRecord;
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);

    sbt.callablesRecordCount = 0;

    struct Params {
        CUdeviceptr debug;
    };

    void* d_debug;
    cudaMalloc(&d_debug, sizeof(uint2) * 5 * 5);

    Params pipelineParams = {};
    pipelineParams.debug = (CUdeviceptr)d_debug;

    void* d_pipelineParams;
    cudaMalloc(&d_pipelineParams, sizeof(Params));
    cudaMemcpy(d_pipelineParams, &pipelineParams, sizeof(Params), cudaMemcpyHostToDevice);

    OptixResult res = optixLaunch(
        pipelineWrapper.pipeline,
        stream,
        (CUdeviceptr)d_pipelineParams,
        sizeof(Params),
        &sbt,
        5,
        5,
        1
    );
    OPTIX_CHECK(res);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<uint2> h_debug(5 * 5);
    cudaMemcpy(h_debug.data(), d_debug, 5 * 5 * sizeof(uint2), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i += 1) {
        for (int j = 0; j < 5; j += 1) {
            uint2 ij = h_debug[i * 5 + j];
            printf("(%d, %d) ", ij.x, ij.y);
        }
        printf("\n");
    }
}