#include "cl/geometry.cl"
#include "cl/utils.cl"
#include "cl/accel.cl"
#include "cl/camera.cl"
#include "cl/lights.cl"
#include "cl/materials.cl"

// #define DEBUG

float3 ray_color(
    ray_t ray, 
    bvh_data_t bvh,
    camera_t camera,

    uint2* rng_state,
    __global light_t* lights,
    int num_lights,

    bool debug
) {
    bsdf_t matte;
    matte.tag = BSDF_LAMBERTIAN;
    matte.value.lambertian = (lambertian_t) { .albedo = (float3) (1.0f, 1.0f, 1.0f) }; // 100% reflective test material

    // printf("began traversal\n");
    hit_info_t hit_info;
    hit_info.hit = false;
    traverse_bvh(
        ray,
        camera.near_clip,
        camera.far_clip,
        &hit_info,
        bvh,
        false
    );

    // printf("finished traversal\n");
    if (hit_info.hit) {
        if (debug) {
            return hit_info.tuv;
        }
        // calculate direct illumination
        int light_index = rand_int(rng_state, 0, num_lights);
        float3 direction;
        visibility_check_t visibility;
        float3 incident_radiance = sample_light(lights[light_index], hit_info.point, &direction, &visibility);
        bool light_occluded = occluded(bvh, visibility);
        if (!light_occluded) {
            float3 bsdf_value = evaluate_bsdf(matte, direction, -ray.direction);
            float cos_theta = fabs(dot(hit_info.normal, direction));
            return bsdf_value * incident_radiance * num_lights * cos_theta;
        }
        else {
            return (float3) (0.0f, 0.0f, 0.0f);
        }
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
    
    __global float* raster_to_world_buf,
    float camera_position_x,
    float camera_position_y,
    float camera_position_z,
    int is_perspective,

    __global float* vertices,
    __global uint* tris,
    __global bvh_node_t* bvh_tree,

    __global light_t* lights,
    int num_lights
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

    __local camera_t camera;

    #ifdef DEBUG
    IF_LEADER {
        for (int i = 0; i < num_lights; i += 1) {
            light_t light = lights[i];
            debug_print_light(light);
        }
    }
    #endif    

    int k = get_local_id(0);
    int l = get_local_id(0);
    if (k == 0 && l == 0) {
        // TODO: see if this can be improved, shouldn't be a major bottleneck
        transform raster_to_world_transform = (transform) {
            .m = vload16(0, raster_to_world_buf + 16),
            .inverse = vload16(0, raster_to_world_buf)
        };

        camera = (camera_t) {
            .raster_to_world_transform = raster_to_world_transform,
            .camera_position = (float3) (camera_position_x, camera_position_y, camera_position_z),
            .is_perspective = is_perspective,
            .near_clip = 0.01f, // TODO: parameterize
            .far_clip = 1000.0f
        };
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    bvh_data_t bvh = { .bvh_tree = bvh_tree, .triangles = tris, .vertices = vertices };
    ray_t ray = generate_ray(&seed, i, j, camera);

    bool debug_lighting = (num_lights == 0);
    float3 color = (float3) (0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num_samples; i += 1) {
        color += ray_color(ray, bvh, camera, &seed, lights, num_lights, debug_lighting);
    }
    color /= num_samples;
    
    frame_buffer[pixel_index] = clamp(color.s0, 0.0f, 1.0f);
    frame_buffer[pixel_index + 1] = clamp(color.s1, 0.0f, 1.0f);
    frame_buffer[pixel_index + 2] = clamp(color.s2, 0.0f, 1.0f);    
}