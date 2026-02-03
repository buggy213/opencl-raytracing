#include "pipeline.h"

#include <cstdint>
#include <vector>

#include "lib_types.h"
#include "optix.h"
#include "util.h"

__host__ OptixPipelineWrapper makeAovPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize) {
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    moduleCompileOptions.numPayloadTypes = 0;
    moduleCompileOptions.payloadTypes = 0;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 4;
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
    builtinModuleOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
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

    OptixProgramGroupDesc sphereHitGroupDesc = {};
    OptixProgramGroupDesc triHitGroupDesc = {};

    sphereHitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphereHitGroupDesc.hitgroup.moduleCH = module;
    sphereHitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__normal_sphere";
    sphereHitGroupDesc.hitgroup.moduleIS = builtinModule;
    sphereHitGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
    sphereHitGroupDesc.hitgroup.moduleAH = nullptr;
    sphereHitGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

    triHitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    triHitGroupDesc.hitgroup.moduleCH = module;
    triHitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__normal_tri";
    triHitGroupDesc.hitgroup.moduleIS = nullptr;
    triHitGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
    triHitGroupDesc.hitgroup.moduleAH = nullptr;
    triHitGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

    OptixProgramGroupOptions hitGroupOptions = {};
    OptixProgramGroup sphereHitGroup = nullptr;
    OptixProgramGroup triHitGroup = nullptr;

    OptixProgramGroupDesc hitGroupDescs[2] = { sphereHitGroupDesc, triHitGroupDesc };
    OptixProgramGroup hitGroups[2];
    res = optixProgramGroupCreate(
        ctx,
        hitGroupDescs,
        2,
        &hitGroupOptions,
        nullptr,
        nullptr,
        hitGroups
    );
    OPTIX_CHECK(res);

    sphereHitGroup = hitGroups[0];
    triHitGroup = hitGroups[1];

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 4;

    OptixProgramGroup programGroups[4] = { raygenGroup, missGroup, sphereHitGroup, triHitGroup };
    res = optixPipelineCreate(
        ctx,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        4,
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
        .sphereHitProgramGroup = sphereHitGroup,
        .triHitProgramGroup = triHitGroup,
    };
}

__host__ void launchAovPipelineImpl(
    OptixPipelineWrapper pipelineWrapper,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    Vec3* normals
) {
    struct EmptyRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    EmptyRecord raygenRecord;
    EmptyRecord missRecord;
    EmptyRecord sphereHitGroupRecord;
    EmptyRecord triHitGroupRecord;

    optixSbtRecordPackHeader(pipelineWrapper.raygenProgram, &raygenRecord);
    optixSbtRecordPackHeader(pipelineWrapper.missProgram, &missRecord);
    optixSbtRecordPackHeader(pipelineWrapper.sphereHitProgramGroup, &sphereHitGroupRecord);
    optixSbtRecordPackHeader(pipelineWrapper.triHitProgramGroup, &triHitGroupRecord);

    void* d_raygenRecord;
    cudaMalloc(&d_raygenRecord, sizeof(EmptyRecord));
    cudaMemcpy(d_raygenRecord, &raygenRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    void* d_missRecord;
    cudaMalloc(&d_missRecord, sizeof(EmptyRecord));
    cudaMemcpy(d_missRecord, &missRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    EmptyRecord* d_hitGroupRecords;
    cudaMalloc(&d_hitGroupRecords, 2 * sizeof(EmptyRecord));
    cudaMemcpy(&d_hitGroupRecords[0], &sphereHitGroupRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_hitGroupRecords[1], &triHitGroupRecord, sizeof(EmptyRecord), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = (CUdeviceptr)d_raygenRecord;

    sbt.missRecordBase = (CUdeviceptr)d_missRecord;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);

    sbt.hitgroupRecordBase = (CUdeviceptr)d_hitGroupRecords;
    sbt.hitgroupRecordCount = 2;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);

    sbt.callablesRecordCount = 0;

    struct Params {
        CUdeviceptr normals;
        CUdeviceptr camera;
        OptixTraversableHandle root_handle;
    };

    size_t width = camera->raster_width;
    size_t height = camera->raster_height;

    void* d_normals;
    cudaMalloc(&d_normals, sizeof(float3) * width * height);

    void* d_camera;
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);

    Params pipelineParams = {};
    pipelineParams.normals = (CUdeviceptr)d_normals;
    pipelineParams.camera = (CUdeviceptr)d_camera;
    pipelineParams.root_handle = rootHandle;

    void* d_pipelineParams;
    cudaMalloc(&d_pipelineParams, sizeof(Params));
    cudaMemcpy(d_pipelineParams, &pipelineParams, sizeof(Params), cudaMemcpyHostToDevice);

    OptixResult res = optixLaunch(
        pipelineWrapper.pipeline,
        stream,
        (CUdeviceptr)d_pipelineParams,
        sizeof(Params),
        &sbt,
        width,
        height,
        1
    );
    OPTIX_CHECK(res);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    std::vector<float3> h_normals(width * height);
    cudaMemcpy(h_normals.data(), d_normals, sizeof(float3) * width * height, cudaMemcpyDeviceToHost);

    // float3 is 16 bytes (aligned), Vec3 is 12 bytes (packed)
    for (size_t i = 0; i < width * height; i++) {
        normals[i].x = h_normals[i].x;
        normals[i].y = h_normals[i].y;
        normals[i].z = h_normals[i].z;
    }

    cudaFree(d_normals);
    cudaFree(d_camera);
    cudaFree(d_pipelineParams);
    cudaFree(d_raygenRecord);
    cudaFree(d_missRecord);
    cudaFree(d_hitGroupRecords);
}

OptixPipelineWrapper makePathtracerPipelineImpl(OptixDeviceContext ctx, const uint8_t *progData, size_t progSize) {
    // TODO
}


