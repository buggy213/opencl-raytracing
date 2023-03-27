#include "cl/geometry.cl"
#ifndef RT_CAMERA
#define RT_CAMERA
ray_t generate_ray(uint2 *rng_state, int x, int y, transform raster_to_camera) {
    float x_disp = (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float y_disp = (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float3 raster_loc = (float3) ((float)x + x_disp, (float)y + y_disp, 0.0f);

    float3 camera_loc = apply_transform_point(raster_to_camera, raster_loc);
    float3 camera_dir = camera_loc / length(camera_loc);

    ray_t r = {
        .origin = (float3) (0.0f, 0.0f, 0.0f),
        .direction = camera_dir
    };
    
    return r;
}


#endif