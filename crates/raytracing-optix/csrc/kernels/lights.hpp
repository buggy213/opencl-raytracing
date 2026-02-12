#pragma once

// lighting calculation only should be occurring in pathtracer kernel, not aov kernel
#include "pathtracer.hpp"
#include "kernel_params.hpp"
#include "sample.hpp"
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
        // TODO: area light sampling not yet implemented
        return LightSample {
            .radiance = float3_zero,
            .shadow_ray = Ray {
                .origin = point,
                .direction = make_float3(0.0f, 0.0f, 1.0f)
            },
            .distance = 1.0f,
            .pdf = 1.0f
        };
    }

    return LightSample {
        .radiance = float3_zero,
        .shadow_ray = Ray {
            .origin = point,
            .direction = make_float3(0.0f, 0.0f, 1.0f)
        },
        .distance = 1.0f,
        .pdf = 1.0f
    };
}

// @raytracing_cpu::lights::light_radiance
inline __device__ float3 light_radiance(const Light& light)
{
    switch (light.kind)
    {
    case Light::PointLight:
    case Light::DirectionLight:
        return float3_zero;
    case Light::DiffuseAreaLight:
        {
            const Light::LightVariant::DiffuseAreaLight& area_light = light.variant.area_light;
            return vec3_to_float3(area_light.radiance);
        }
    }

    return float3_zero;
}

// @raytracing_cpu::lights::occluded
inline __device__ bool occluded(const LightSample& light_sample)
{
    ShadowRayPayload res = traceShadowRay(
        light_sample.shadow_ray.origin,
        light_sample.shadow_ray.direction,
        0.0001f,
        light_sample.distance - 0.0001f
    );

    return res.hit != 0;
}

} // namespace lights