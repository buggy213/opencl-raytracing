#include "cl/geometry.cl"
#include "cl/utils.cl"

#ifndef RT_CAMERA
#define RT_CAMERA

// only projective cameras supported
typedef struct {
    float3 camera_position;
    transform raster_to_world_transform;
    int is_perspective;
    float near_clip;
    float far_clip;
} camera_t;

ray_t generate_ray(uint2 *rng_state, int x, int y, camera_t camera) {
    float x_disp = 0.0f; // rand_float(rng_state);
    float y_disp = 0.0f; // rand_float(rng_state);
    float3 raster_loc = (float3) ((float)x + x_disp, (float)y + y_disp, 0.0f);
    float3 ray_origin;
    float3 ray_dir;
    if (camera.is_perspective) {
        ray_origin = camera.camera_position;
        float3 camera_loc = apply_transform_point(camera.raster_to_world_transform, raster_loc);
        ray_dir = (camera_loc - ray_origin);
    }
    else {
        ray_origin = apply_transform_point(camera.raster_to_world_transform, raster_loc);
        float3 raster_loc_offset = (float3) ((float)x + x_disp, (float)y + y_disp, 1.0f);
        float3 camera_loc = apply_transform_point(camera.raster_to_world_transform, raster_loc_offset);
        ray_dir = (camera_loc - ray_origin);
    }
    ray_dir = ray_dir / length(ray_dir);
    
    #ifdef DEBUG
    IF_LEADER {
        printf("%v3f\n", raster_loc);
        printf("%v3f\n", camera_position);
        printf("%v3f\n", ray_origin);
        printf("%v3f\n", ray_dir);
    }
    #endif

    ray_t r = {
        .origin = ray_origin,
        .direction = ray_dir
    };
    
    return r;
}


#endif