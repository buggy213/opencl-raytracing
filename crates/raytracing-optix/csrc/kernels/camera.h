#pragma once

#include "types.h"
#include "kernel_types.h"
#include "kernel_math.h"
#include "sample.h"

// TODO: temporary while AOV pass is still lacking sampler
inline __device__ Ray generate_ray(
    const Camera& camera,
    u32 x,
    u32 y,
    sample::OptixSampler* sampler
) {
    float xf = static_cast<float>(x);
    float yf = static_cast<float>(y);
    if (sampler)
    {
        auto [x_disp, y_disp] = sampler->sample_uniform2();
        xf += x_disp;
        yf += y_disp;
    }

    float3 raster_loc = make_float3(xf, yf, 0.0f);

    switch (camera.camera_type.kind) {
        case Orthographic: {
            float3 camera_space_o = matrix4x4_apply_point(camera.raster_to_camera.forward, raster_loc);
            float3 camera_space_d = make_float3(0.0f, 0.0f, 1.0f);

            float3 ray_o = matrix4x4_apply_point(camera.camera_to_world.forward, camera_space_o);
            float3 ray_d = normalize(matrix4x4_apply_vector(camera.camera_to_world.forward, camera_space_d));

            return Ray { .origin = ray_o, .direction = ray_d };
        }
        case PinholePerspective: {
            float3 cam_point = matrix4x4_apply_point(camera.raster_to_camera.forward, raster_loc);
            float3 cam_ray_dir = normalize(cam_point);

            float3 world_origin = matrix4x4_apply_point(camera.camera_to_world.forward, float3_zero);
            float3 world_dir = normalize(matrix4x4_apply_vector(camera.camera_to_world.forward, cam_ray_dir));

            return Ray { .origin = world_origin, .direction = world_dir };
        }
        case ThinLensPerspective:
            return {};
    }

    return {};
}