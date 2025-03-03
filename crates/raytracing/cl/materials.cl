#include "cl/geometry.cl"
#include "cl/sampling.cl"

#ifndef RT_MATERIALS
#define RT_MATERIALS

// Lambertian reflection: BSDF is uniform over hemisphere
typedef struct {
    float3 albedo; // TODO: add texture sampling later
} lambertian_t;

typedef struct {
    float3 color;
    float ior; 
} fresnel_specular_t;

typedef fresnel_specular_t specular_reflection_t;
typedef fresnel_specular_t specular_transmission_t;

float3 world_to_local(float3 world_direction, surface_interaction_t interaction) {
    float x = dot(interaction.tangent, world_direction);
    float y = dot(interaction.bitangent, world_direction);
    float z = dot(interaction.normal, world_direction);
    return (float3) (x, y, z);
}

float3 local_to_world(float3 local_direction, surface_interaction_t interaction) {
    return local_direction.x * interaction.tangent + local_direction.y * interaction.bitangent + local_direction.z * interaction.normal;
}

__constant int BSDF_LAMBERTIAN = 0; // perfectly matte material
__constant int BSDF_SPECULAR_REFLECTION = 1; // reflection only
__constant int BSDF_SPECULAR_TRANSMISSION = 2; // transmission only
__constant int BSDF_FRESNEL_SPECULAR = 3; // fresnel-modulated specular reflection and transmission
typedef struct {
    int tag;
    union {
        lambertian_t lambertian;
        specular_reflection_t specular_reflection;
        specular_transmission_t specular_transmission;
        fresnel_specular_t fresnel_specular;
    } value;
} bsdf_t;

// Specular reflection / transmission, describes interaction of light with perfectly flat surfaces
// in the case that index of refraction is real valued (e.g. glass, diamonds, etc.)
// uncolored reflectance
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

// ior_ratio is eta_incident / eta_transmitted
// derivation in PBR: https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#SpecularTransmission
bool refract(float3 out_dir, surface_interaction_t interaction, float3* refract_dir, float ior_ratio) {
    // snell's law: (eta_i) sin(theta_i) = (eta_t) sin(theta_t)
    float cos_theta_incident = dot(out_dir, interaction.normal);
    float sin2_theta_incident = max(0.0f, 1.0f - cos_theta_incident * cos_theta_incident);
    float sin2_theta_transmitted = ior_ratio * ior_ratio * sin2_theta_incident;
    if (sin2_theta_transmitted >= 1.0f) {
        return false;
    }
    float cos_theta_transmitted = sqrt(1.0f - sin2_theta_transmitted);

    *refract_dir = -out_dir * ior_ratio + (ior_ratio * cos_theta_incident - cos_theta_transmitted) * interaction.normal;
    return true;
}

// convention: both in_dir and out_dir point away from the point of intersection (towards source and destination of radiance, respectively)
// and normal is pointing towards outside of object
float3 evaluate_bsdf(bsdf_t bsdf, float3 in_dir, float3 out_dir) {
    switch (bsdf.tag) {
        case BSDF_LAMBERTIAN:
            return bsdf.value.lambertian.albedo * M_1_PI_F; // multiply by 1 / pi since radiance is conserved
        case BSDF_SPECULAR_REFLECTION:
        case BSDF_SPECULAR_TRANSMISSION:
        case BSDF_FRESNEL_SPECULAR:
            return (float3) (0.0f, 0.0f, 0.0f); // 0 since these BSDF's include a delta distribution term
        default:
            printf("invalid BSDF tag value");
            return (float3) (0.0f, 0.0f, 0.0f);
    }
}

// TODO: stratified random sampling for lower variance
float3 sample_bsdf(
    bsdf_t bsdf, 
    float3 out_dir, 
    float3* sampled_in_dir, 
    float* pdf, 
    uint2* rng_state, 
    surface_interaction_t interaction
) {
    switch (bsdf.tag) {
        case BSDF_LAMBERTIAN:;
            // cosine weighted importance sampling: preferentially choose directions not close to glancing angles as they will contribute more to 
            // indirect lighting
            float2 rand = rand_float2(rng_state);
            float3 direction = sample_cosine_hemisphere(rand, pdf);
            *sampled_in_dir = local_to_world(direction, interaction);
            return bsdf.value.lambertian.albedo * M_1_PI_F;
        case BSDF_SPECULAR_REFLECTION:;
            float3 in_dir = reflect(out_dir, interaction.normal);
            *sampled_in_dir = in_dir;
            *pdf = 1.0f; // implied delta distribution
            float cos_theta_incident = dot(in_dir, interaction.normal);
            float ior = bsdf.value.specular_reflection.ior;
            float r = dielectric_reflectance(cos_theta_incident, 1.0f, ior);
            return bsdf.value.specular_reflection.color * r / fabs(cos_theta_incident);
        case BSDF_SPECULAR_TRANSMISSION:;
            float cos_theta_incident = dot(in_dir, interaction.normal);
            float ior_ratio = (cos_theta_incident > 0.0f) ? bsdf.value.specular_transmission.ior : 1.0f / bsdf.value.specular_transmission.ior;
            bool valid_transmission = refract(out_dir, interaction, sampled_in_dir, ior_ratio);
            *pdf = 1.0f;
            if (!valid_transmission) {
                return (float3) (0.0f, 0.0f, 0.0f);
            }
            float r = 1.0f - dielectric_reflectance(cos_theta_incident, 1.0f, bsdf.value.specular_transmission.ior);
            return ior_ratio * ior_ratio * r * bsdf.value.specular_transmission.color / fabs(cos_theta_incident);
        default:
            printf("invalid BSDF tag value");
            return (float3) (0.0f, 0.0f, 0.0f);
    }
}

#endif