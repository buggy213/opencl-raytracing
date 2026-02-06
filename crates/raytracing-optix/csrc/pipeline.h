#pragma once

#include <cstdint>
#include <optix_types.h>
#include <string_view>

#include "lib_optix_types.h"
#include "lib_types.h"

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
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    Vec3* normals
);

void releaseAovPipelineImpl(AovPipeline& pipeline);

struct PathtracerPipeline {
    OptixPipeline pipeline;

    enum class RayType {
        RADIANCE,
        SHADOW,
        RAY_TYPE_COUNT
    };

    enum class GeometryType {
        TRIANGLE,
        SPHERE,
        GEOMETRY_TYPE_COUNT
    };

    enum class MaterialType {
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

    static constexpr unsigned int radiancePayloadTypeSemantics[3] = {
        // RGB components all have the same semantics: readable by the caller of trace after being written by miss or closest-hit
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
    };
    static constexpr OptixPayloadType radiancePayloadType = { .numPayloadValues = 3, .payloadSemantics = radiancePayloadTypeSemantics };

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
        static_cast<size_t>(GeometryType::GEOMETRY_TYPE_COUNT) * static_cast<size_t>(MaterialType::MATERIAL_TYPE_COUNT);

    OptixProgramGroup hitProgramGroupsRadiance[radianceHitProgramCount];

    OptixModule& intersectionModule(GeometryType geometryType) {
        return intersectionModules[static_cast<size_t>(geometryType)];
    }

    OptixProgramGroup& missProgram(RayType rayType) {
        return missPrograms[static_cast<size_t>(rayType)];
    }

    OptixProgramGroup& shadowHitProgram(GeometryType geometryType) {
        return hitProgramGroupsShadow[static_cast<size_t>(geometryType)];
    }

    OptixProgramGroup& radianceHitProgram(GeometryType geometryType, MaterialType materialType) {
        size_t index = static_cast<size_t>(geometryType) * static_cast<size_t>(MaterialType::MATERIAL_TYPE_COUNT) + static_cast<size_t>(materialType);
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
    const Scene* scene,
    OptixTraversableHandle rootHandle,
    Vec3* radiance
);

void releasePathtracerPipelineImpl(PathtracerPipeline& pipeline);