// Ray-generation w/ pinhole camera; only performs primary visibility and only calculates geometric normals for now (more aovs later)
#include "aov.hpp"

#include "types.h"

#include "kernel_math.hpp"
#include "kernel_types.hpp"

#include "accel.hpp"
#include "camera.hpp"

#include <optix_device.h>

extern "C" __global__ void __raygen__debug() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    Ray ray = generate_ray(*pipeline_params.camera, tid.x, tid.y, nullptr);

    uint x, y, z, w;
    optixTrace(
        pipeline_params.root_handle,
        ray.origin,
        ray.direction,
        pipeline_params.camera->near_clip,
        pipeline_params.camera->far_clip,
        0.0f,
        OptixVisibilityMask(0xFF),
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
    HitInfo hit = get_hit_info_sphere();

    float3 normal = hit.normal;
    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
}

extern "C" __global__ void __closesthit__normal_tri() {
    HitInfo hit = get_hit_info_tri();

    float3 normal = hit.normal;
    optixSetPayload_0(__float_as_uint(normal.x));
    optixSetPayload_1(__float_as_uint(normal.y));
    optixSetPayload_2(__float_as_uint(normal.z));
}