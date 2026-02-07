// Ray-generation w/ pinhole camera; only performs primary visibility and only calculates geometric normals
#define USE_AOV_PIPELINE_PARAMS
#include "kernel_params.h"
#include "types.h"

#include "kernel_math.h"
#include "kernel_types.h"

#include "camera.h"

#include <optix_device.h>

extern "C" __global__ void __raygen__debug() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    Ray ray = generate_ray(*pipeline_params.camera, tid.x, tid.y);

    uint x, y, z, w;
    optixTrace(
        pipeline_params.root_handle,
        ray.origin,
        ray.direction,
        pipeline_params.camera->near_clip,
        pipeline_params.camera->far_clip,
        0.0f,
        (OptixVisibilityMask)(-1),
        OPTIX_RAY_FLAG_NONE,
        0,
        0,
        0,
        x,
        y,
        z,
        w
    );

    float3& target = pipeline_params.normals[tid.y * dim.x + tid.x];
    target = make_float3(
        __uint_as_float(x),
        __uint_as_float(y),
        __uint_as_float(z)
    );
}

// set payload values to zeros
extern "C" __global__ void __miss__nop() {
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
    optixSetPayload_3(__float_as_uint(0.0f));
}

extern "C" __global__ void __closesthit__normal_sphere() {
    float4 sphere_data[1];
    optixGetSphereData(sphere_data);

    // w is the radius
    float3 sphere_center = make_float3(sphere_data->x, sphere_data->y, sphere_data->z);
    float sphere_radius = sphere_data->w;

    float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_direction = optixGetWorldRayDirection();
    float t = optixGetRayTmax();

    float3 intersection_point = ray_origin + ray_direction * t;

    float3 normal = (intersection_point - sphere_center) / sphere_radius;
    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
}

extern "C" __global__ void __closesthit__normal_tri() {
    float3 tri_data[3];
    optixGetTriangleVertexData(tri_data);

    // OptiX has front-face CCW winding order by default
    float3 e01 = tri_data[1] - tri_data[0];
    float3 e02 = tri_data[2] - tri_data[0];

    float3 normal = normalize(cross(e01, e02));
    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
}