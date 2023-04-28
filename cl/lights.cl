#include "cl/geometry.cl"

#ifndef RT_LIGHTS
#define RT_LIGHTS

typedef struct {
    float3 position;
    float3 intensity; // W / steradian
} point_light_t;


const int POINT_LIGHT = 0;
typedef struct {
    int tag;
    union {
        point_light_t point;
    } value;
} light_t;

// returns radiance at point 
float3 sample_light(light_t light, float3 point, float3* direction) {
    switch (light.tag) {
        case POINT_LIGHT:;
            float3 dir = point - light.value.point.position;
            dir = dir / length(dir);
            // test visibility

            float square_distance = dot(dir, dir);
            *direction = dir;
            return light.value.point.intensity / square_distance; // divide by square distance to convert from intensity to radiance (?)
        default:
            printf("unsupported light type");
            return (float3) (0.0f, 0.0f, 0.0f);
    }
}

// bool occluded(ray_t ray, )

#endif