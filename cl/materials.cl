#include "cl/geometry.cl"

#ifndef RT_MATERIALS
#define RT_MATERIALS

// Specular reflection / transmission, describes interaction of light with perfectly flat surfaces

// Calculates Fresnel reflectance given the cosine of the angle between an incoming ray and the normal of the surface 
// as well as the indices of refraction for the media on either side of the dielectric
float fresnel_dielectric(float cos_theta_incident, float eta_incident, float eta_transmitted) {
    cos_theta_incident = clamp(cos_theta_incident, -1.0f, 1.0f);
     // if angle between incident direction and normal is acute then ray is coming from outside of material
    bool entering = cos_theta_incident > 0.0f;
    if (!entering) {
        float tmp = eta_incident;
        eta_incident = eta_transmitted;
        eta_transmitted = tmp;
        cos_theta_incident = fabs(cos_theta_incident);
    }

    float sin_theta_incident = sqrt(max(0.0f, 1.0f - cos_theta_incident * cos_theta_incident));
    float sin_theta_transmitted = eta_incident / eta_transmitted * sin_theta_incident; // snell's law
    if (sin_theta_incident >= 1.0f) {
        return 1.0f;
    }

    float cos_theta_transmitted = sqrt(max(0.0f, 1.0f - sin_theta_incident * sin_theta_incident));

    float r_parallel = (eta_transmitted * cos_theta_incident - eta_incident * cos_theta_transmitted) / (eta_transmitted * cos_theta_incident + eta_incident * cos_theta_transmitted);
    float r_perp = (eta_incident * cos_theta_incident - eta_transmitted * cos_theta_transmitted) / (eta_incident * cos_theta_incident + eta_transmitted * cos_theta_transmitted);
    return (r_parallel * r_parallel + r_perp * r_perp) / 2; // assuming that light is unpolarized
}



#endif