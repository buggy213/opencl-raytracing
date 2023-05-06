#include "cl/geometry.cl"
#include "cl/sampling.cl"

#ifndef RT_MATERIALS
#define RT_MATERIALS

// Lambertian reflection: BSDF is uniform over hemisphere
typedef struct {
    float3 albedo; // TODO: add texture sampling later
} lambertian_t;


__constant int BSDF_LAMBERTIAN = 0;
typedef struct {
    int tag;
    union {
        lambertian_t lambertian;
    } value;
} bsdf_t;


// convention: both in_dir and out_dir point away from the point of intersection (towards source and destination of radiance, respectively)
// and normal is pointing towards outside of object
float3 evaluate_bsdf(bsdf_t bsdf, float3 in_dir, float3 out_dir) {
    switch (bsdf.tag) {
        case BSDF_LAMBERTIAN:
            return bsdf.value.lambertian.albedo * M_1_PI_F; // multiply by 1 / pi since radiance is conserved
        default:
            printf("invalid BSDF tag value");
            return (float3) (0.0f, 0.0f, 0.0f);
    }
}

float3 sample_bsdf(bsdf_t bsdf, float3 out_dir, float3* sampled_in_dir, float* pdf, float2 uv) {
    switch (bsdf.tag) {
        case BSDF_LAMBERTIAN:
            // cosine weighted importance sampling: preferentially choose directions not close to glancing angles as they will contribute more to 
            // indirect lighting
            // need a transform of some sort...
            // *sampled_in_dir = sample_cosine_hemisphere(uv, pdf);
            return bsdf.value.lambertian.albedo * M_1_PI_F;
        default:
            printf("invalid BSDF tag value");
            return (float3) (0.0f, 0.0f, 0.0f);
    }
}

// Specular reflection / transmission, describes interaction of light with perfectly flat surfaces

float dielectric_reflectance(float cos_theta_incident, float eta_outside, float eta_inside) {
    cos_theta_incident = clamp(cos_theta_incident, -1.0f, 1.0f);
    if (cos_theta_incident > 0.0f) {
        // incident direction is originating outside medium and coming in, no action needed
    }
    else {
        // incident direction is originating within medium and going out, need to swap 
        // indices of refraction and flip cos_theta_incident for equations to work out
        float tmp = eta_outside;
        eta_outside = eta_inside;
        eta_inside = tmp;
        cos_theta_incident = fabs(cos_theta_incident);
    }
    // need to calculate angle of transmitted direction using snell's law: (eta_i) sin(theta_i) = (eta_t) sin(theta_t)
    float sin_theta_incident = sqrt(max(0.0f, 1.0f - cos_theta_incident * cos_theta_incident));
    float sin_theta_transmitted = eta_outside * sin_theta_incident / eta_inside;
    if (sin_theta_transmitted >= 1.0f) {
        // total internal reflection
        return 1.0f;
    }
    float cos_theta_transmitted = sqrt(max(0.0f, 1.0f - sin_theta_incident * sin_theta_incident));

    // assuming light is unpolarized
    float r_parallel = 
        (eta_inside * cos_theta_incident - eta_outside * cos_theta_transmitted) / (eta_inside * cos_theta_incident + eta_outside * cos_theta_transmitted);

    float r_perp = 
        (eta_outside * cos_theta_incident - eta_inside * cos_theta_transmitted) / (eta_outside * cos_theta_incident + eta_inside * cos_theta_transmitted);
    
    return 0.5f * (r_parallel * r_parallel + r_perp * r_perp);
}


#endif