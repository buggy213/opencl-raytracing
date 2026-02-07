#include "pipeline.h"

#include <cstdint>
#include <vector>

#include "types.h"
#include "optix.h"
#include "util.h"
#include "sbt.h"
#include "kernel_params.h"
#include "sbt_host.h"

__host__ AovPipeline makeAovPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize) {
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

    // AOV only needs ray depth of 1
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

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
    return AovPipeline {
        .pipeline = pipeline,
        .module = module,
        .raygenProgram = raygenGroup,
        .missProgram = missGroup,
        .sphereHitProgramGroup = sphereHitGroup,
        .triHitProgramGroup = triHitGroup,
    };
}

__host__ void launchAovPipelineImpl(
    const AovPipeline& pipeline,
    const AovSbt& sbt,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    Vec3* normals
) {
    size_t width = camera->raster_width;
    size_t height = camera->raster_height;

    void* d_normals;
    cudaMalloc(&d_normals, sizeof(float3) * width * height);

    void* d_camera;
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);

    AovPipelineParams pipelineParams = {};
    pipelineParams.normals = (float3*)d_normals;
    pipelineParams.camera = (Camera*)d_camera;
    pipelineParams.root_handle = rootHandle;

    void* d_pipelineParams;
    cudaMalloc(&d_pipelineParams, sizeof(AovPipelineParams));
    cudaMemcpy(d_pipelineParams, &pipelineParams, sizeof(AovPipelineParams), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    OptixResult res = optixLaunch(
        pipeline.pipeline,
        stream,
        (CUdeviceptr)d_pipelineParams,
        sizeof(AovPipelineParams),
        &sbt.sbt,
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
}

__host__ void releaseAovPipelineImpl(AovPipeline& pipeline) {
    OPTIX_CHECK(optixPipelineDestroy(pipeline.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(pipeline.raygenProgram));
    OPTIX_CHECK(optixProgramGroupDestroy(pipeline.missProgram));
    OPTIX_CHECK(optixProgramGroupDestroy(pipeline.sphereHitProgramGroup));
    OPTIX_CHECK(optixProgramGroupDestroy(pipeline.triHitProgramGroup));
    OPTIX_CHECK(optixModuleDestroy(pipeline.module));
}

__host__ PathtracerPipeline makePathtracerPipelineImpl(
    OptixDeviceContext ctx,
    const uint8_t *progData,
    size_t progSize,
    unsigned int maxRayDepth
) {
    PathtracerPipeline pathtracerPipeline;
    std::vector<OptixProgramGroup> programGroups;

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    moduleCompileOptions.numPayloadTypes = 2;
    moduleCompileOptions.payloadTypes = PathtracerPipeline::payloadTypes;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 0; // use payload types instead
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "pipeline_params";
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM |
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    // -- create OptixModules --
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
    pathtracerPipeline.module = module;

    pathtracerPipeline.intersectionModule(PathtracerPipeline::GeometryType::TRIANGLE) = nullptr;

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
    pathtracerPipeline.intersectionModule(PathtracerPipeline::GeometryType::SPHERE) = builtinModule;

    // -- create raygen program --
    OptixProgramGroupDesc raygenGroupDesc = {};
    raygenGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenGroupDesc.raygen.module = module;
    raygenGroupDesc.raygen.entryFunctionName = "__raygen__main";

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
    pathtracerPipeline.raygenProgram = raygenGroup;
    programGroups.push_back(raygenGroup);

    // -- create miss programs --
    OptixProgramGroupDesc missGroupRadianceDesc = {};
    missGroupRadianceDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missGroupRadianceDesc.miss.module = module;
    missGroupRadianceDesc.miss.entryFunctionName = "__miss__radiance";

    OptixProgramGroupOptions missGroupRadianceOptions = { .payloadType = &PathtracerPipeline::radiancePayloadType };
    OptixProgramGroup missGroupRadiance = nullptr;
    res = optixProgramGroupCreate(
        ctx,
        &missGroupRadianceDesc,
        1,
        &missGroupRadianceOptions,
        nullptr,
        nullptr,
        &missGroupRadiance
    );
    OPTIX_CHECK(res);
    pathtracerPipeline.missProgram(PathtracerPipeline::RayType::RADIANCE) = missGroupRadiance;
    programGroups.push_back(missGroupRadiance);

    OptixProgramGroupDesc missGroupShadowDesc = {};
    missGroupShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missGroupShadowDesc.miss.module = module;
    missGroupShadowDesc.miss.entryFunctionName = "__miss__shadow";

    OptixProgramGroupOptions missGroupShadowOptions = { .payloadType = &PathtracerPipeline::shadowPayloadType };
    OptixProgramGroup missGroupShadow = nullptr;
    res = optixProgramGroupCreate(
        ctx,
        &missGroupShadowDesc,
        1,
        &missGroupShadowOptions,
        nullptr,
        nullptr,
        &missGroupShadow
    );
    OPTIX_CHECK(res);
    pathtracerPipeline.missProgram(PathtracerPipeline::RayType::SHADOW) = missGroupShadow;
    programGroups.push_back(missGroupShadow);

    // -- create closest-hit programs for radiance rays --
    for (int geometry_type = 0; geometry_type < static_cast<int>(PathtracerPipeline::GeometryType::GEOMETRY_TYPE_COUNT); geometry_type++) {
        for (int material_type = 0; material_type < static_cast<int>(PathtracerPipeline::MaterialType::MATERIAL_TYPE_COUNT); material_type++) {
            auto geometryType = static_cast<PathtracerPipeline::GeometryType>(geometry_type);
            auto materialType = static_cast<PathtracerPipeline::MaterialType>(material_type);

            OptixProgramGroupDesc desc = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc.hitgroup.moduleIS = pathtracerPipeline.intersectionModule(geometryType);
            desc.hitgroup.entryFunctionNameIS = nullptr;
            desc.hitgroup.moduleCH = pathtracerPipeline.module;
            desc.hitgroup.entryFunctionNameCH = PathtracerPipeline::hitProgramNamesRadiance[material_type].data();
            desc.hitgroup.moduleAH = nullptr;
            desc.hitgroup.entryFunctionNameAH = nullptr;

            OptixProgramGroupOptions options = { .payloadType = &PathtracerPipeline::radiancePayloadType };
            OptixProgramGroup group = nullptr;
            res = optixProgramGroupCreate(
                ctx,
                &desc,
                1,
                &options,
                nullptr,
                nullptr,
                &group
            );
            OPTIX_CHECK(res);
            pathtracerPipeline.radianceHitProgram(geometryType, materialType) = group;
            programGroups.push_back(group);
        }
    }

    // -- create closest-hit programs for shadow rays --
    for (int geometry_type = 0; geometry_type < static_cast<int>(PathtracerPipeline::GeometryType::GEOMETRY_TYPE_COUNT); geometry_type++) {
        auto geometryType = static_cast<PathtracerPipeline::GeometryType>(geometry_type);

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = pathtracerPipeline.intersectionModule(geometryType);
        desc.hitgroup.entryFunctionNameIS = nullptr;
        desc.hitgroup.moduleCH = pathtracerPipeline.module;
        desc.hitgroup.entryFunctionNameCH = PathtracerPipeline::hitProgramNameShadow.data();
        desc.hitgroup.moduleAH = nullptr;
        desc.hitgroup.entryFunctionNameAH = nullptr;

        OptixProgramGroupOptions options = { .payloadType = &PathtracerPipeline::shadowPayloadType };
        OptixProgramGroup group = nullptr;
        res = optixProgramGroupCreate(
            ctx,
            &desc,
            1,
            &options,
            nullptr,
            nullptr,
            &group
        );
        OPTIX_CHECK(res);
        pathtracerPipeline.shadowHitProgram(geometryType) = group;
        programGroups.push_back(group);
    }

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = maxRayDepth;

    res = optixPipelineCreate(
        ctx,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        programGroups.size(),
        nullptr,
        nullptr,
        &pipeline
    );
    OPTIX_CHECK(res);
    pathtracerPipeline.pipeline = pipeline;

    return pathtracerPipeline;
}

void launchPathtracerPipelineImpl(
    const PathtracerPipeline& pipeline,
    const Scene* scene,
    OptixTraversableHandle rootHandle,
    Vec3* radiance
) {
    // TODO
}

void releasePathtracerPipelineImpl(PathtracerPipeline &pipeline) {
    // TODO
}

