// Ray-generation w/ pinhole camera; only performs primary visibility and only calculates geometric normals
#include "params.h"
#include "lib_types.h"
#include "kernel_types.h"

#include <optix_device.h>

__device__ Ray generate_ray(
    const Camera& camera,
    unsigned int x,
    unsigned int y
) {
    switch (camera.camera_type.kind) {
        case Orthographic:
            return {};
        case PinholePerspective: {
            float xf = __uint_as_float(x);
            float yf = __uint_as_float(y);
            float3 raster_loc = make_float3(xf, yf, 0.0f);
            float3 cam_point = camera.raster_to_camera.forward * raster_loc;

            float3 world_origin = camera.camera_to_world.forward * make_float3(0.0f, 0.0f, 0.0f);
            float3 world_dir = camera.camera_to_world.forward * cam_point;

            return Ray { .origin = world_origin, .direction = world_dir };
        }
        case ThinLensPerspective:
            return {};
    }

    return {};
}

extern "C" __global__ void __raygen__debug() {
    uint3 dim = optixGetLaunchDimensions();
    uint3 tid = optixGetLaunchIndex();

    Ray ray = generate_ray(*pipeline_params.camera, tid.x, tid.y);

    uint x, y, z, w;
    optixTrace(
        pipeline_params.root_handle,
        ray.origin,
        ray.direction,
        0.0f,
        10000.0f,
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
}

// set payload values to zeros
extern "C" __global__ void __miss__nop() {
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
    optixSetPayload_3(__float_as_uint(0.0f));
}

extern "C" __global__ void __closesthit__normal() {
    float4 sphere_data[1];
    optixGetSphereData(sphere_data);

    // w is the radius
    optixSetPayload_0(__float_as_uint(sphere_data[0].x));
    optixSetPayload_1(__float_as_uint(sphere_data[0].y));
    optixSetPayload_2(__float_as_uint(sphere_data[0].z));
    optixSetPayload_3(__float_as_uint(sphere_data[0].w));
}