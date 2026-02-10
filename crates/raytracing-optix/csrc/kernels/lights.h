#pragma once

// lighting calculation only should be occurring in pathtracer kernel, not aov kernel
#include "pathtracer.h"
#include "kernel_params.h"
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
            float3 dir = point - vec3_to_float3(point_light.position);
            float d = length(dir);
            float d2 = d * d;

            return LightSample {
                .radiance = vec3_to_float3(point_light.intensity) / d2,
                .shadow_ray = Ray {
                    .origin = vec3_to_float3(point_light.position),
                    .direction = dir
                },
                .distance = d,
                .pdf = 1.0f
            };
        }
    case Light::DirectionLight:
        {
            Light::LightVariant::DirectionLight direction_light = light.variant.direction_light;
            float scene_diameter = pipeline_params.scene_diameter;
            float3 light_origin = point - vec3_to_float3(direction_light.direction) * scene_diameter;

            return LightSample {
                .radiance = vec3_to_float3(direction_light.radiance),
                .shadow_ray = Ray {
                    .origin = light_origin,
                    .direction = normalize(vec3_to_float3(direction_light.direction)),
                },
                .distance = scene_diameter,
                .pdf = 1.0f
            };
        }
    case Light::DiffuseAreaLight:
        break;
    }
}

// @raytracing_cpu::lights::occluded
inline __device__ bool occluded(const LightSample& light_sample)
{
    unsigned int hit;
    // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT corresponds to early exit
    optixTrace(
        OPTIX_PAYLOAD_TYPE_ID_1,
        pipeline_params.root_handle,
        light_sample.shadow_ray.origin,
        light_sample.shadow_ray.direction,
        0.001f,
        light_sample.distance - 0.001f,
        0.0f,
        (OptixVisibilityMask)(-1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RayType::SHADOW_RAY,
        RayType::RAY_TYPE_COUNT,
        RayType::SHADOW_RAY,
        hit
    );

    return hit != 0;
}

} // namespace lights