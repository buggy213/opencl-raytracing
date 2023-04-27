#include "cl/geometry.cl"
#ifndef RT_CAMERA
#define RT_CAMERA
ray_t generate_ray(uint2 *rng_state, int x, int y, transform raster_to_world, float3 camera_position, bool perspective) {
    float x_disp = 0.0f; // (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float y_disp = 0.0f; // (float) (MWC64X(rng_state) % 1024) / 1024.0f;
    float3 raster_loc = (float3) ((float)x + x_disp, (float)y + y_disp, 0.0f);
    float3 ray_origin;
    float3 ray_dir;
    if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
        printf("is_perspective=%d", perspective);
    }
    if (perspective) {
        ray_origin = camera_position;
        float3 camera_loc = apply_transform_point(raster_to_world, raster_loc);
        ray_dir = (camera_loc - ray_origin);
    }
    else {
        ray_origin = apply_transform_point(raster_to_world, raster_loc);
        float3 raster_loc_offset = (float3) ((float)x + x_disp, (float)y + y_disp, 1.0f);
        float3 camera_loc = apply_transform_point(raster_to_world, raster_loc_offset);
        ray_dir = (camera_loc - ray_origin);
    }
    ray_dir = ray_dir / length(ray_dir);
    
    /*if (get_global_id(0) == 0 && get_global_id(1) == get_global_size(1) - 1) {
        printf("%v3f\n", raster_loc);
        printf("%v3f\n", camera_position);
        printf("%v3f\n", ray_origin);
        printf("%v3f\n", ray_dir);
    }*/

    ray_t r = {
        .origin = ray_origin,
        .direction = ray_dir
    };
    
    return r;
}


#endif