#pragma once

#include "sample.h"
#include "types.h"

namespace lights
{

// @raytracing::lights::Light::is_delta_light
inline __device__ bool is_delta_light(const Light& light)
{
    switch (light.kind) {
    case Light::PointLight:
        return true;
    case Light::DirectionLight:
        return true;
    case Light::DiffuseAreaLight:
        return false;
    }
    return true;
}

// @raytracing_cpu::lights::LightSample
struct LightSample
{
    float3 radiance;
    Ray shadow_ray;
    float distance;
    float pdf;
};

// @raytracing_cpu::lights::sample_light
inline __device__ LightSample sample_light(
    const Light& light,
    float3 point,
    sample::OptixSampler& sampler
) {
    switch (light.kind)
    {
    case Light::PointLight:
        {
            Light::LightVariant::PointLight point_light = light.variant.point_light;
            float3 dir = point - point_light.position;
            float d = length(dir);
            float d2 = d * d;

            return LightSample {
                .radiance = point_light.intensity / d2,
                .shadow_ray = Ray {
                    .origin = point_light.position,
                    .direction = dir
                },
                .distance = d,
                .pdf = 1.0f
            };
        }
    case Light::DirectionLight:
        {
            float scene_diameter = pipeline_params.
        }
    case Light::DiffuseAreaLight:
        break;
    }
}

} // namespace lights