#pragma once

#include <optix_device.h>

#include "lib_types.h"

struct PipelineParams {
    float3* normals;
    Camera* camera;
    OptixTraversableHandle root_handle;
};
extern "C" __constant__ PipelineParams pipeline_params;