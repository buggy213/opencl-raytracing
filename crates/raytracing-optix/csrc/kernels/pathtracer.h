#pragma once

#include "sample.h"

#ifdef USE_AOV_PIPELINE_PARAMS
#error mixing aov and pathtracer code not allowed
#endif

#define USE_PATHTRACER_PIPELINE_PARAMS
#include "kernel_params.h"
extern "C" __constant__ PathtracerPipelineParams pipeline_params;

enum RayType
{
    RADIANCE_RAY,
    SHADOW_RAY,
    RAY_TYPE_COUNT
};

struct PerRayData {
    sample::OptixSampler sampler;
};

static_assert(sizeof(PerRayData) == sizeof(PathtracerPerRayData), "pathtracer per-ray data size mismatch");
static_assert(alignof(PerRayData) == alignof(PathtracerPerRayData), "pathtracer per-ray data alignment mismatch");

inline __device__ PerRayData& get_ray_data() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    return (reinterpret_cast<PerRayData*>(pipeline_params.ray_datas))[tid.y * dim.x + tid.x];
}