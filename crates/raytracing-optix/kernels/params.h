#pragma once

#include <optix_device.h>

#include "lib_types.h"

struct AovPipelineParams {
    float3* normals;
    Camera* camera;
    OptixTraversableHandle root_handle;
};

struct PathtracerPipelineParams
{
    float3* radiance;
    Camera* camera;
    OptixTraversableHandle root_handle;
};

#ifdef USE_AOV_PIPELINE_PARAMS
extern "C" __constant__ AovPipelineParams pipeline_params;
#endif

#ifdef USE_PATHTRACER_PIPELINE_PARAMS
extern "C" __constant__ PathtracerPipelineParams pipeline_params;
#endif