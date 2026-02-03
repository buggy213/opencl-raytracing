#pragma once

#include "lib_types.h"
#include "kernel_types.h"
#include "kernel_math.h"

__device__ Ray generate_ray(
    const Camera& camera,
    unsigned int x,
    unsigned int y
) {
    switch (camera.camera_type.kind) {
        case Orthographic:
            return {};
        case PinholePerspective: {
            float xf = (float)x;
            float yf = (float)y;
            float3 raster_loc = make_float3(xf, yf, 0.0f);
            float3 cam_point = camera.raster_to_camera.forward * raster_loc;

            float3 world_origin = camera.camera_to_world.forward * make_float3(0.0f, 0.0f, 0.0f);
            float3 world_point = camera.camera_to_world.forward * cam_point;
            float3 world_dir = normalize(world_point - world_origin);

            return Ray { .origin = world_origin, .direction = world_dir };
        }
        case ThinLensPerspective:
            return {};
    }

    return {};
}