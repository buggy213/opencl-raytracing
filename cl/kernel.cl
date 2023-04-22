#include "cl/geometry.cl"
#include "cl/utils.cl"
#include "cl/accel.cl"
#include "cl/camera.cl"

float3 ray_color(
    ray_t ray, 
    __global bvh_node_t* bvh,
    __global uint* triangles, 
    __global float* vertices
) {
    // printf("began traversal\n");
    /*float3 v1 = (float3)(0.258717f, 1.015191f, 8.968245f);
    float3 v2 = (float3)(-1.29084f, 0.534367f, 10.137722f);
    float3 v3 = (float3)(1.290841f,-0.534367f, 9.698717f);
    float3 v4 = (float3)(-0.25871f, -1.015191f, 10.868195f);
    float3 tuv;
    bool hit = ray_triangle_intersect(v2, v3, v1, ray, 0.01f, 1000.0f, &tuv);
    if (hit) {
        return tuv;
    }
    hit = ray_triangle_intersect(v2, v4, v3, ray, 0.01f, 1000.0f, &tuv);
    if (hit) {
        return tuv;
    }
    */
    hit_info_t hit_info;
    hit_info.hit = false;
    traverse_bvh(
        ray,
        0.01f,
        1000.0f,
        &hit_info,
        bvh, 
        triangles, 
        vertices
    );

    // printf("finished traversal\n");
    if (hit_info.hit) {
        return hit_info.tuv;
    }
    
    // printf("finished traversal\n");
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
    __global float* raster_to_world_buf,

    float camera_position_x,
    float camera_position_y,
    float camera_position_z,

    __global float* vertices,
    __global uint* tris,
    __global bvh_node_t* bvh
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

    __local transform raster_to_world_transform;
    int k = get_local_id(0);
    int l = get_local_id(0);
    if (k == 0 && l == 0) {
        raster_to_world_transform = (transform) {
            .m = vload16(0, raster_to_world_buf + 16),
            .inverse = vload16(0, raster_to_world_buf)
        };
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    float3 camera_position = (float3) (camera_position_x, camera_position_y, camera_position_z);
    ray_t ray = generate_ray(&seed, i, j, raster_to_world_transform, camera_position);
    float3 color = ray_color(ray, bvh, tris, vertices);

    frame_buffer[pixel_index] = color.r;
    frame_buffer[pixel_index + 1] = color.g;
    frame_buffer[pixel_index + 2] = color.b;    
}