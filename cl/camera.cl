#include "cl/geometry.cl"
#ifndef RT_CAMERA
#define RT_CAMERA
ray_t generate_ray(uint2 *rng_state, int x, int y, transform raster_to_world, float3 camera_position) {
    float x_disp = 0.0f; // (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float y_disp = 0.0f; // (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float3 raster_loc = (float3) ((float)x + x_disp, (float)y + y_disp, 0.0f);

    float3 camera_loc = apply_transform_point(raster_to_world, raster_loc);
    float3 camera_dir = (camera_loc - camera_position);
    camera_dir = camera_dir / length(camera_dir);
    
    if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
        printf("%v3f\n", raster_loc);
        printf("%v3f\n", camera_position);
        printf("%v3f\n", camera_loc);
        printf("%v3f\n", camera_dir);
    }

    ray_t r = {
        .origin = camera_position,
        .direction = camera_dir
    };
    
    return r;
}


#endif