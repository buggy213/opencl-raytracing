#include "cl/accel.cl"
#include "cl/geometry.cl"

#ifndef RT_LIGHTS
#define RT_LIGHTS

typedef struct {
    float position[3];
    float intensity[3]; // W / steradian
} point_light_t;


__constant int POINT_LIGHT = 0;
typedef struct {
    int tag;
    union {
        point_light_t point;
    } value;
} light_t;

typedef struct {
    ray_t ray;
    float dist;
} visibility_check_t;

void debug_print_light(light_t light) {
    switch (light.tag) {
        case POINT_LIGHT:
            printf("point light (%d)\n", sizeof(light_t));
            printf("position=%f %f %f\n", light.value.point.position[0], light.value.point.position[1], light.value.point.position[2]);
            printf("intensity=%f %f %f\n", light.value.point.intensity[0], light.value.point.intensity[1], light.value.point.intensity[2]);
            break;
        default:
            printf("unknown light type: %d\n", light.tag);
            break;
    }
}

// returns radiance at point 
float3 sample_light(light_t light, float3 point, float3* direction, visibility_check_t* visibility_closure) {
    switch (light.tag) {
        case POINT_LIGHT:;
            float3 pos = (float3) (light.value.point.position[0], light.value.point.position[1], light.value.point.position[2]);
            float3 intensity = (float3) (light.value.point.intensity[0], light.value.point.intensity[1], light.value.point.intensity[2]);
            float3 dir = point - pos;
            float d = length(dir);
            float square_distance = dot(dir, dir);
            dir = dir / length(dir);
            *direction = dir;
            ray_t light_to_point = { .origin = pos, .direction = dir };
            *visibility_closure = (visibility_check_t) {
                .ray = light_to_point,
                .dist = d
            };
            return intensity / square_distance; // divide by square distance to convert from intensity to radiance (?)
        default:
            printf("unsupported light type");
            return (float3) (0.0f, 0.0f, 0.0f);
    }
}

bool occluded(bvh_data_t bvh, visibility_check_t visibility_closure) {
    hit_info_t hit_info;
    hit_info.hit = false;
    traverse_bvh(
        visibility_closure.ray,
        0.0f,
        visibility_closure.dist - 0.01f,
        &hit_info,
        bvh,
        true
    );

    return hit_info.hit;
}

#endif