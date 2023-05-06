#include "cl/utils.cl"

#ifndef RT_SAMPLING
#define RT_SAMPLING

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
float2 uniform_sample_disk(float2 uv) {
    // doesn't work as well for stratified random sampling apparently; TODO: look into this
    float r = sqrt(uv.s0);
    float theta = M_2_PI_F * uv.s1;
    return (float2) (r * cos(theta), r * sin(theta));
}

// "Malley's method"
float3 sample_cosine_hemisphere(float2 uv, float* pdf) {
    float2 disk_sample = uniform_sample_disk(uv);
    float z = sqrt(max(0.0f, 1.0f - disk_sample.x * disk_sample.x - disk_sample.y * disk_sample.y));
    *pdf = z * M_1_PI_F; // z / pi since z is cos_theta
    return (float3) (disk_sample.x, disk_sample.y, z);
}

#endif