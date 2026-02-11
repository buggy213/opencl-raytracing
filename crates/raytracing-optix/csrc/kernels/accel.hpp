#pragma once

#include "kernel_math.hpp"
#include "kernel_types.hpp"
#include "sbt.hpp"

#include <optix_device.h>

// the name "accel.h" is a little misleading, since the actual acceleration structure traversal is handled elsewhere
// this file just captures the functionality of constructing a HitInfo

struct HitInfo {
    float2 uv;
    float3 point;
    float3 normal;
};

// following functions must be only be called from closest-hit programs
// also, they return HitInfo in world-space. note that this is different
// from convention in cpu references

// @raytracing_cpu::geometry::ray_sphere_intersect
inline __device__ HitInfo get_hit_info_sphere() {
    float4 sphere_data[1];
    optixGetSphereData(sphere_data);

    // w component is the radius
    float3 sphere_center_o = make_float3(sphere_data->x, sphere_data->y, sphere_data->z);
    float3 sphere_center_w = optixTransformPointFromObjectToWorldSpace(sphere_center_o);
    float sphere_radius = sphere_data->w;

    float3 ray_origin_w = optixGetWorldRayOrigin();
    float3 ray_direction_w = optixGetWorldRayDirection();
    float t = optixGetRayTmax();

    float3 intersection_point = ray_origin_w + ray_direction_w * t;
    float3 normal = (intersection_point - sphere_center_w) / sphere_radius;

    float3 local = intersection_point - sphere_center_w;
    float r_cos_theta = local.z;
    float theta = acosf(r_cos_theta / sphere_radius);
    float r_sin_theta_cos_phi = local.x;
    float cos_phi = r_sin_theta_cos_phi / (sphere_radius * sinf(theta));
    float r_sin_theta_sin_phi = local.y;
    float sin_phi = r_sin_theta_sin_phi / (sphere_radius * sinf(theta));

    float phi = local.y > 0.0f ? acosf(cos_phi) : M_2_PIf - acosf(cos_phi);
    float2 uv = make_float2(phi / (2.0f * M_PIf), theta * M_1_PIf);

    return HitInfo { .uv = uv, .point = intersection_point, .normal = normal };
}

// @raytracing_cpu::geometry::ray_mesh_intersect
inline __device__ HitInfo get_hit_info_tri() {
    auto mesh_data = reinterpret_cast<HitgroupRecord::MeshData*>(optixGetSbtDataPointer());
    u32 tri_index = optixGetPrimitiveIndex();

    uint3 tri = mesh_data->indices[tri_index];

    float2 barycentric = optixGetTriangleBarycentrics();
    float t = optixGetRayTmax();
    float u = barycentric.x;
    float v = barycentric.y;
    float w = 1.0f - barycentric.x - barycentric.y;

    float3 normal_o;
    if (!mesh_data->normals) {
        float3 p[3];
        optixGetTriangleVertexData(p);

        normal_o = normalize(cross(p[2] - p[0], p[1] - p[0]));
    }
    else {
        float3 n0 = mesh_data->normals[tri.x];
        float3 n1 = mesh_data->normals[tri.y];
        float3 n2 = mesh_data->normals[tri.z];

        normal_o = normalize(w * n0 + u * n1 + v * n2);
    }

    float2 uv0, uv1, uv2;
    if (!mesh_data->uvs) {
        uv0 = make_float2(0.0f, 0.0f);
        uv1 = make_float2(1.0f, 0.0f);
        uv2 = make_float2(0.0f, 1.0f);
    }
    else {
        uv0 = mesh_data->uvs[tri.x];
        uv1 = mesh_data->uvs[tri.y];
        uv2 = mesh_data->uvs[tri.z];
    }

    float2 uv = w * uv0 + u * uv1 + v * uv2;

    float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 normal_w = optixTransformNormalFromObjectToWorldSpace(normal_o);

    return HitInfo { .uv = uv, .point = ray_origin + ray_direction * t, .normal = normal_w };
}