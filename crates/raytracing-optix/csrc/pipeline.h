#pragma once

#include <cstdint>
#include <optix_types.h>

#include "lib_optix_types.h"
#include "lib_types.h"

AovPipelineWrapper makeAovPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize);
void launchAovPipelineImpl(
    AovPipelineWrapper pipeline,
    const Camera* camera,
    OptixTraversableHandle rootHandle,
    Vec3* normals
);

PathtracerPipelineWrapper makePathtracerPipelineImpl(OptixDeviceContext ctx, const uint8_t* progData, size_t progSize);