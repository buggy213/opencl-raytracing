#pragma once

// lighting calculation only should be occurring in pathtracer kernel, not aov kernel
#include "geometry.hpp"
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
        {
            const Light::LightVariant::DiffuseAreaLight& diffuse_area_light = light.variant.area_light;
            const HitgroupRecord& emitter = pipeline_params.sbt_hitgroup_records[diffuse_area_light.prim_id];

            // if it's not a triangle mesh, this will break horribly
            float pdf = 1.0f;
            pdf /= emitter.mesh_data.num_tris;

            u32 random_tri_idx = sampler.sample_u32(0, emitter.mesh_data.num_tris);
            float2 sample = sampler.sample_uniform2();
            float3 bary;
            if (sample.x < sample.y)
            {
                float b0 = sample.x / 2.0f;
                float b1 = sample.y - sample.x / 2.0f;
                float b2 = 1.0f - b0 - b1;
                bary = make_float3(b0, b1, b2);
            }
            else
            {
                float b0 = sample.x - sample.y / 2.0f;
                float b1 = sample.y / 2.0f;
                float b2 = 1.0f - b0 - b1;
                bary = make_float3(b0, b1, b2);
            }

            uint3 tri = emitter.mesh_data.indices[random_tri_idx];
            float3 p0, p1, p2;
            p0 = emitter.mesh_data.vertices[tri.x];
            p1 = emitter.mesh_data.vertices[tri.y];
            p2 = emitter.mesh_data.vertices[tri.z];

            pdf /= geometry::tri_area(p0, p1, p2);

            float3 p_local = bary.x * p0 + bary.y * p1 + bary.z * p2;
            float3 p_world = matrix4x4_apply_point(diffuse_area_light.light_to_world, p_local);
            float3 dir_world = point - p_world;
            float d = length(dir_world);

            float3 n;
            if (!emitter.mesh_data.normals)
            {
                n = normalize(cross(p2 - p0, p1 - p0));
            }
            else
            {
                float3 n0 = emitter.mesh_data.normals[tri.x];
                float3 n1 = emitter.mesh_data.normals[tri.y];
                float3 n2 = emitter.mesh_data.normals[tri.z];

                n = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
            }

            float3 radiance;
            if (dot(dir_world, n) < 0.0f)
            {
                radiance = float3_zero;
            }
            else
            {
                radiance = vec3_to_float3(diffuse_area_light.radiance);
            }

            pdf *= (d * d) / fabs(dot(dir_world, n));
            return LightSample {
                radiance,
                .shadow_ray = Ray {
                    .origin = p_world,
                    .direction = dir_world / d
                },
                .distance = d,
                pdf
            };
        }
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