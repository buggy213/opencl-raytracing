#pragma once

/*
 * Pipeline parameters for the different pipelines; shared with host C++ code
 */

#include <optix_device.h>

#include "types.h"

struct AovPipelineParams {
    float3* normals;
    Camera* camera;
    OptixTraversableHandle root_handle;
};

struct PathtracerPipelineParams
{
    float3* radiance;
    Scene* scene;
    OptixTraversableHandle root_handle;
};

#ifdef USE_AOV_PIPELINE_PARAMS
extern "C" __constant__ AovPipelineParams pipeline_params;
#endif

#ifdef USE_PATHTRACER_PIPELINE_PARAMS
extern "C" __constant__ PathtracerPipelineParams pipeline_params;
#endif