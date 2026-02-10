#pragma once

/*
 * Pipeline parameters for the different pipelines; shared between host/device C++ code
 */

#include <optix_device.h>

#include "types.h"

struct AovPipelineParams {
    float3* normals;
    Camera* camera;
    OptixTraversableHandle root_handle;
};

// this field is deliberately opaque to the host C++ code, which is only responsible for allocating it
// a static assert within the kernel code ensures that it is sufficiently large for actual contents
struct PathtracerPerRayData {
    __align__(8) char data[64];
};

struct PathtracerPipelineParams
{
    float4* radiance;
    Scene scene;
    OptixAabb scene_aabb;
    float scene_diameter;
    OptixTraversableHandle root_handle;
    Texture* textures;
    PathtracerPerRayData* ray_datas;
    OptixRaytracerSettings settings;
};
