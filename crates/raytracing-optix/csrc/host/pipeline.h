#pragma once

#include <cstdint>
#include <optix_types.h>
#include <string_view>

#include "lib_optix_types.h"
#include "types.h"

struct AovPipeline {
    OptixPipeline pipeline;
    OptixModule module;
    OptixProgramGroup raygenProgram;
    OptixProgramGroup missProgram;
    OptixProgramGroup sphereHitProgramGroup;
    OptixProgramGroup triHitProgramGroup;
};

AovPipeline makeAovPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize);

void launchAovPipelineImpl(
    const AovPipeline& pipeline,
    const AovSbt& sbt,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    Vec3* normals
);

void releaseAovPipelineImpl(AovPipeline& pipeline);

struct PathtracerPipeline {
    OptixPipeline pipeline;

    enum RayType {
        RADIANCE,
        SHADOW,
        RAY_TYPE_COUNT
    };

    enum GeometryType {
        TRIANGLE,
        SPHERE,
        GEOMETRY_TYPE_COUNT
    };

    enum MaterialType {
        DIFFUSE,
        MATERIAL_TYPE_COUNT
    };

    static constexpr std::string_view missProgramNames[RayType::RAY_TYPE_COUNT] = {
        "__miss__radiance",
        "__miss__shadow"
    };

    static constexpr std::string_view hitProgramNamesRadiance[MaterialType::MATERIAL_TYPE_COUNT] = {
        "__closesthit__radiance_diffuse"
    };
    static constexpr std::string_view hitProgramNameShadow = "__closesthit__shadow";

    static constexpr unsigned int radiancePayloadTypeSemantics[4] = {
        // radiance components all have the same semantics: readable by the caller of trace after being written by miss or closest-hit
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        // specular bounce flag is writable by caller of trace, read from closest-hit / miss in order to determine if light contribution
        // should be added
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ
    };
    static constexpr OptixPayloadType radiancePayloadType = { .numPayloadValues = 4, .payloadSemantics = radiancePayloadTypeSemantics };

    static constexpr unsigned int shadowPayloadTypeSemantics[1] = {
        // hit flag has semantics: readable by caller of trace after being written by miss (i.e. not in shadow) or closest-hit (i.e. in shadow)
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
    };
    static constexpr OptixPayloadType shadowPayloadType = { .numPayloadValues = 1, .payloadSemantics = shadowPayloadTypeSemantics };
    static constexpr OptixPayloadType payloadTypes[2] = { radiancePayloadType, shadowPayloadType };


    // primary module
    OptixModule module;

    // builtin intersection modules
    OptixModule intersectionModules[GeometryType::GEOMETRY_TYPE_COUNT];

    // 1 raygen program at top level
    OptixProgramGroup raygenProgram;

    // 1 miss program per ray-type
    OptixProgramGroup missPrograms[RayType::RAY_TYPE_COUNT];

    // 1 hitgroup per geometry for shadow rays (material doesn't matter for shadow rays)
    OptixProgramGroup hitProgramGroupsShadow[GeometryType::GEOMETRY_TYPE_COUNT];

    // 1 per hitgroup (geometry, material) for radiance rays
    static constexpr size_t radianceHitProgramCount =
        GeometryType::GEOMETRY_TYPE_COUNT * MaterialType::MATERIAL_TYPE_COUNT;

    OptixProgramGroup hitProgramGroupsRadiance[radianceHitProgramCount];

    OptixModule& intersectionModule(GeometryType geometryType) {
        return intersectionModules[geometryType];
    }

    OptixProgramGroup& missProgram(RayType rayType) {
        return missPrograms[rayType];
    }

    OptixProgramGroup& shadowHitProgram(GeometryType geometryType) {
        return hitProgramGroupsShadow[geometryType];
    }

    OptixProgramGroup& radianceHitProgram(GeometryType geometryType, MaterialType materialType) {
        size_t index = geometryType * MaterialType::MATERIAL_TYPE_COUNT + materialType;
        return hitProgramGroupsRadiance[index];
    }
};

PathtracerPipeline makePathtracerPipelineImpl(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize,
    unsigned int maxRayDepth
);

void launchPathtracerPipelineImpl(
    const PathtracerPipeline& pipeline,
    const PathtracerSbt& sbt,
    OptixRaytracerSettings settings,
    Scene scene,
    OptixTraversableHandle rootHandle,
    Vec4* radiance
);

void releasePathtracerPipelineImpl(PathtracerPipeline& pipeline);