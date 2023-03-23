#include "cl/geometry.cl"
#include "cl/utils.cl"
#include "cl/accel.cl"

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

float3 ray_color(ray_t ray) {

    float3 v0 = (float3) (-0.5f, -0.5f, 10.0f);
    float3 v1 = (float3) (0.5f, -0.5f, 4.0f);
    float3 v2 = (float3) (-0.5f, 0.5f, 4.0f);
    float3 tuv;
    if (ray_triangle_intersect(v0, v1, v2, ray, 0.01, INFINITY, &tuv)) {
        return tuv;
    }

    float3 normalized_direction = normalize(ray.direction);
    float t = 0.5f * (normalized_direction.y + 1.0f);
    return (1.0f - t) * (float3) (1.0f, 1.0f, 1.0f) + t * (float3) (0.5f, 0.7f, 1.0f);
}

void __kernel render(
    __global float* frame_buffer, 
    int frame_width,
     int frame_height,
    int num_samples, 
    __global uint* seeds_buffer,
    __global float* raster_to_camera_buf,
    __global float* mesh_vertices,
    __global uint* mesh_tris
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= frame_width || j >= frame_height) {
        return;
    }

    int pixel_index = (j * frame_width + i) * 3;
    float u = (float) i / frame_width;
    float v = (float) j / frame_height;

    uint2 seed = (uint2) (seeds_buffer[(j * frame_width + i) * 2], seeds_buffer[(j * frame_width + i) * 2 + 1]);

    __local transform raster_to_camera_transform;
    int k = get_local_id(0);
    int l = get_local_id(0);
    if (k == 0 && l == 0) {
        raster_to_camera_transform = (transform) {
            .m = vload16(0, raster_to_camera_buf + 16),
            .inverse = vload16(0, raster_to_camera_buf)
        };
        // if (i == 0 && j == 0) {
        //     for (int i = 0; i < 32; i += 1) {
        //         printf("%f\n", raster_to_camera_buf[i]);
        //     }

        //     printf("%v16f\n", raster_to_camera_transform.m);
        //     printf("%v16f\n", raster_to_camera_transform.inverse);
        // }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    ray_t ray = generate_ray(&seed, i, j, raster_to_camera_transform);
    float3 color = ray_color(ray);

    frame_buffer[pixel_index] = color.r;
    frame_buffer[pixel_index + 1] = color.g;
    frame_buffer[pixel_index + 2] = color.b;    
}