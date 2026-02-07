#pragma once

#include "shared_lib.h"
#include "types.h"
#include "lib_optix_types.h"

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#include <stdbool.h>
#endif

#include <optix_types.h>

/* Context management */
RT_API OptixDeviceContext initOptix(bool debug);
RT_API void destroyOptix(OptixDeviceContext ctx);

/* Scene conversion functions */
RT_API struct OptixAccelerationStructure makeSphereAccelerationStructure(
    OptixDeviceContext ctx,
    struct Vec3 center,
    float radius
);

RT_API struct OptixAccelerationStructure makeMeshAccelerationStructure(
    OptixDeviceContext ctx,
    const struct Vec3* vertices, /* packed */
    size_t verticesLen, /* number of float3's */
    const struct Vec3u* tris, /* packed */
    size_t trisLen /* number of uint3's */
);

RT_API struct OptixAccelerationStructure makeInstanceAccelerationStructure(
    OptixDeviceContext ctx,
    const struct OptixAccelerationStructure* instances,
    const struct Matrix4x4* transforms,
    const unsigned int* sbtOffsets,
    size_t len
);

// Rust side takes ownership, and should call `releaseAovPipeline` when finished
RT_API AovPipelineWrapper makeAovPipeline(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize
);

RT_API AovSbtWrapper makeAovSbt();
RT_API void addHitRecordAovSbt(AovSbtWrapper sbt, struct GeometryData geometryData);
RT_API void finalizeAovSbt(AovSbtWrapper sbt, AovPipelineWrapper pipeline);
RT_API void releaseAovSbt(AovSbtWrapper sbt);

RT_API void launchAovPipeline(
    AovPipelineWrapper pipeline,
    AovSbtWrapper sbt,
    const struct Camera* camera,
    OptixTraversableHandle rootHandle,
    struct Vec3* normals
);

RT_API void releaseAovPipeline(AovPipelineWrapper pipeline);

// Rust side takes ownership, and should call `releasePathtracerPipeline` when finished
RT_API PathtracerPipelineWrapper makePathtracerPipeline(
    OptixDeviceContext ctx,
    const uint8_t* progData,
    size_t progSize,
    unsigned int maxRayDepth
);

RT_API PathtracerSbtWrapper makePathtracerSbt();
RT_API void addHitRecordPathtracerSbt(PathtracerSbtWrapper, struct GeometryData geometryData);
RT_API void finalizePathtracerSbt(PathtracerSbtWrapper sbt, PathtracerPipelineWrapper pipeline);
RT_API void releasePathtracerSbt(PathtracerSbtWrapper sbt);

RT_API void launchPathtracerPipeline(
    PathtracerPipelineWrapper pipeline,
    const struct Scene* scene,
    OptixTraversableHandle rootHandle,
    struct Vec3* radiance
);

RT_API void releasePathtracerPipeline(PathtracerPipelineWrapper pipeline);
