#pragma once

#include <cuda/std/optional>

#include "kernel_math.hpp"
#include "kernel_types.hpp"
#include "sample.hpp"
#include "texture.hpp"

namespace materials
{

/// In the common case, we don't need the std::variant of the various BSDFs, since material dispatch
/// is handled by different closest-hit programs in the SBT. This is good for register usage / scheduling
/// We still do need it for more specialized (i.e. mix, layered) materials.
struct OptixBsdfDiffuse {
    float3 albedo;

    __device__ static constexpr bool is_delta_bsdf() { return false; }
};

struct OptixBsdfSmoothDielectric {
    float eta;

    __device__ static constexpr bool is_delta_bsdf() { return true; }
};

struct OptixBsdfSmoothConductor {
    float3 eta;
    float3 kappa;

    __device__ static constexpr bool is_delta_bsdf() { return true; }
};

struct OptixBsdfRoughConductor {
    float3 eta;
    float3 kappa;
    float alpha_x;
    float alpha_y;

    __device__ static constexpr bool is_delta_bsdf() { return false; }
};

struct OptixBsdfRoughDielectric {
    float eta;
    float alpha_x;
    float alpha_y;

    __device__ static constexpr bool is_delta_bsdf() { return false; }
};

struct LayeredBsdf {
    // todo
};

// -- Material evaluation --
// these roughly correspond to the material evaluation routines of
// @<raytracing::materials::Material as raytracing_cpu::materials::CpuMaterial>::get_bsdf
// but don't need to match against the kind of material it is, since this is known from call site

inline __device__ OptixBsdfDiffuse get_diffuse_bsdf(const Material& material, float2 uv)
{
    const Material::MaterialVariant::Diffuse& diffuse = material.variant.diffuse;
    float4 albedo = texture::sample(diffuse.albedo, uv);
    return OptixBsdfDiffuse { .albedo = make_float3(albedo.x, albedo.y, albedo.z) };
}

inline __device__ OptixBsdfSmoothDielectric get_smooth_dielectric_bsdf(const Material& material, float2 uv)
{
    const Material::MaterialVariant::SmoothDielectric& smooth_dielectric = material.variant.smooth_dielectric;
    float4 eta = texture::sample(smooth_dielectric.eta, uv);
    return OptixBsdfSmoothDielectric { eta.x };
}

inline __device__ OptixBsdfSmoothConductor get_smooth_conductor_bsdf(const Material& material, float2 uv)
{
    const Material::MaterialVariant::SmoothConductor& smooth_conductor = material.variant.smooth_conductor;
    float4 eta = texture::sample(smooth_conductor.eta, uv);
    float4 kappa = texture::sample(smooth_conductor.kappa, uv);
    return OptixBsdfSmoothConductor {
        .eta = make_float3(eta.x, eta.y, eta.z),
        .kappa = make_float3(kappa.x, kappa.y, kappa.z)
    };
}

constexpr float MINIMUM_ROUGHNESS = 1.0e-3f;

// Rough dielectric / conductor both return a smooth fallback if roughness is below threshold
inline __device__ cuda::std::variant<OptixBsdfRoughDielectric, OptixBsdfSmoothDielectric> get_rough_dielectric_bsdf(const Material& material, float2 uv)
{
    const auto& rd = material.variant.rough_dielectric;
    float eta = texture::sample(rd.eta, uv).x;
    float4 roughness = texture::sample(rd.roughness, uv);
    float alpha_x = roughness.x;
    float alpha_y = roughness.y;

    if (rd.remap_roughness)
    {
        alpha_x = sqrtf(alpha_x);
        alpha_y = sqrtf(alpha_y);
    }

    if (fmaxf(alpha_x, alpha_y) < MINIMUM_ROUGHNESS)
    {
        return OptixBsdfSmoothDielectric { eta };
    }
    return OptixBsdfRoughDielectric { eta, alpha_x, alpha_y };
}

inline __device__ cuda::std::variant<OptixBsdfRoughConductor, OptixBsdfSmoothConductor> get_rough_conductor_bsdf(const Material& material, float2 uv)
{
    const auto& rc = material.variant.rough_conductor;
    float4 eta = texture::sample(rc.eta, uv);
    float4 kappa = texture::sample(rc.kappa, uv);
    float4 roughness = texture::sample(rc.roughness, uv);
    float alpha_x = roughness.x;
    float alpha_y = roughness.y;

    if (rc.remap_roughness)
    {
        alpha_x = sqrtf(alpha_x);
        alpha_y = sqrtf(alpha_y);
    }

    float3 eta3 = make_float3(eta.x, eta.y, eta.z);
    float3 kappa3 = make_float3(kappa.x, kappa.y, kappa.z);

    if (fmaxf(alpha_x, alpha_y) < MINIMUM_ROUGHNESS)
    {
        return OptixBsdfSmoothConductor { eta3, kappa3 };
    }
    return OptixBsdfRoughConductor { eta3, kappa3, alpha_x, alpha_y };
}

struct BsdfComponentFlags {
    u32 value;

    __device__ constexpr static BsdfComponentFlags EMPTY() { return { 0 }; }
    __device__ constexpr static BsdfComponentFlags NONSPECULAR_REFLECTION() { return { 1 << 0 }; }
    __device__ constexpr static BsdfComponentFlags SPECULAR_REFLECTION() { return { 1 << 1 }; }
    __device__ constexpr static BsdfComponentFlags NONSPECULAR_TRANSMISSION() { return { 1 << 2 }; }
    __device__ constexpr static BsdfComponentFlags SPECULAR_TRANSMISSION() { return { 1 << 3 }; }

    __device__ constexpr BsdfComponentFlags operator|(BsdfComponentFlags o) const {
        return BsdfComponentFlags { value | o.value };
    }

    __device__ constexpr BsdfComponentFlags operator&(BsdfComponentFlags o) const {
        return BsdfComponentFlags { value & o.value };
    }

    __device__ constexpr BsdfComponentFlags& operator|=(BsdfComponentFlags o) {
        value |= o.value;
        return *this;
    }

    __device__ constexpr BsdfComponentFlags& operator&=(BsdfComponentFlags o)
    {
        value &= o.value;
        return *this;
    }

    __device__ constexpr bool operator==(BsdfComponentFlags o) const
    {
        return value == o.value;
    }

    __device__ constexpr bool operator!=(BsdfComponentFlags o) const
    {
        return value != o.value;
    }

    __device__ constexpr static BsdfComponentFlags REFLECTION() { return NONSPECULAR_REFLECTION() | SPECULAR_REFLECTION(); }
    __device__ constexpr static BsdfComponentFlags TRANSMISSION() { return NONSPECULAR_TRANSMISSION() | SPECULAR_TRANSMISSION(); }
    __device__ constexpr static BsdfComponentFlags SPECULAR() { return SPECULAR_REFLECTION() | SPECULAR_TRANSMISSION(); }
    __device__ constexpr static BsdfComponentFlags NONSPECULAR() { return NONSPECULAR_REFLECTION() | NONSPECULAR_TRANSMISSION(); }
    __device__ constexpr static BsdfComponentFlags ALL() { return REFLECTION() | TRANSMISSION(); }

    __device__ constexpr bool is_specular() const
    {
        return (*this & SPECULAR()) != EMPTY();
    }

    __device__ constexpr bool contains(BsdfComponentFlags set) const
    {
        return (*this & set) == set;
    }

    __device__ constexpr bool intersects(BsdfComponentFlags set) const
    {
        return (*this & set) != EMPTY();
    }
};

struct BsdfSample
{
    float3 wi;
    float3 bsdf;
    float pdf;
    BsdfComponentFlags component;
    bool valid;
};

// Material evaluation helpers

// @raytracing::geometry::Vec3::reflect
inline __device__ float3 reflect(float3 v, float3 n)
{
    return -v + 2.0f * dot(v, n) * n;
}

// @raytracing_cpu::materials::refract
inline __device__ cuda::std::optional<float3> refract(float eta, float3 wo, float3 normal)
{
    float cos_theta_i = dot(wo, normal);
    if (cos_theta_i < 0.0f)
    {
        eta = 1.0f / eta;
        cos_theta_i = -cos_theta_i;
        normal = -normal;
    }

    float sin_theta_2_i = 1.0f - cos_theta_i * cos_theta_i;
    float sin_theta_2_t = sin_theta_2_i / (eta * eta);
    if (sin_theta_2_t >= 1.0f)
    {
        return cuda::std::nullopt;
    }

    float cos_theta_t = sqrtf(1.0f - sin_theta_2_t);
    return -wo / eta + (cos_theta_i / eta - cos_theta_t) * normal;
}

// @raytracing_cpu::materials::fresnel_dielectric
inline __device__ float fresnel_dielectric(float cos_theta_i, float eta)
{
    if (cos_theta_i < 0.0f)
    {
        eta = 1.0f / eta;
        cos_theta_i = -cos_theta_i;
    }

    float sin_theta_2_i = 1.0f - cos_theta_i * cos_theta_i;
    float sin_theta_2_t = sin_theta_2_i / (eta * eta);
    if (sin_theta_2_t >= 1.0f)
    {
        return 1.0f;
    }

    float cos_theta_t = sqrtf(1.0f - sin_theta_2_t);
    float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    return (r_parl * r_parl + r_perp * r_perp) / 2.0f;
}

// @raytracing_cpu::materials::fresnel_complex
inline __device__ float fresnel_complex(float cos_theta_i, complex eta)
{
    float sin_theta_2_i = 1.0f - cos_theta_i * cos_theta_i;
    complex sin_theta_2_t = sin_theta_2_i / (eta * eta);
    complex cos_theta_t = (complex(1.0f, 0.0f) - sin_theta_2_t).sqrt();

    complex r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    complex r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    return (r_parl.norm() + r_perp.norm()) / 2.0f;
}

// BSDF Evaluation
// @raytracing_cpu::materials::CpuBsdf::evaluate_bsdf
inline __device__ float3 evaluate_bsdf(OptixBsdfDiffuse bsdf, float3 wo, float3 wi)
{
    if (wo.z * wi.z < 0.0f)
    {
        return float3_zero;
    }
    else
    {
        return bsdf.albedo / M_PIf;
    }
}

inline __device__ float3 evaluate_bsdf(OptixBsdfSmoothDielectric bsdf, float3 wo, float3 wi)
{
    return float3_zero;
}

inline __device__ float3 evaluate_bsdf(OptixBsdfSmoothConductor bsdf, float3 wo, float3 wi)
{
    return float3_zero;
}

// PDF evaluation
// @raytracing_cpu::materials::CpuBsdf::evaluate_pdf
inline __device__ float evaluate_pdf(OptixBsdfDiffuse bsdf, float3 wo, float3 wi, BsdfComponentFlags component)
{
    if (!component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION()))
    {
        return 0.0f;
    }

    if (wo.z * wi.z > 0.0f)
    {
        return 1.0f / (2.0 * M_PIf);
    }
    else
    {
        return 0.0f;
    }
}

inline __device__ float evaluate_pdf(OptixBsdfSmoothDielectric bsdf, float3 wo, float3 wi, BsdfComponentFlags component)
{
    return 0.0f;
}

inline __device__ float evaluate_pdf(OptixBsdfSmoothConductor bsdf, float3 wo, float3 wi, BsdfComponentFlags component)
{
    return 0.0f;
}

// BSDF sampling
// @raytracing_cpu::materials::CpuBsdf::sample_bsdf
inline __device__ BsdfSample sample_bsdf(
    OptixBsdfDiffuse bsdf, float3 wo, BsdfComponentFlags component, sample::OptixSampler& sampler
) {
    if (!component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION()))
    {
        return {};
    }

    float2 u = sampler.sample_uniform2();
    float3 wi = sample::sample_cosine_hemisphere(u);
    float pdf = wi.z / M_PIf;

    return BsdfSample {
        .wi = wi,
        .bsdf = bsdf.albedo / M_PIf,
        .pdf = pdf,
        .component = BsdfComponentFlags::NONSPECULAR_REFLECTION(),
        .valid = true
    };
}

inline __device__ BsdfSample sample_bsdf(
    OptixBsdfSmoothDielectric bsdf, float3 wo, BsdfComponentFlags component, sample::OptixSampler& sampler
) {
    if (!component.intersects(BsdfComponentFlags::SPECULAR_REFLECTION() | BsdfComponentFlags::SPECULAR_TRANSMISSION()))
    {
        return {};
    }

    float3 normal = make_float3(0.0f, 0.0f, 1.0f);
    float R = fresnel_dielectric(wo.z, bsdf.eta);
    float T = 1.0f - R;

    float p_reflect = component.contains(BsdfComponentFlags::SPECULAR_REFLECTION()) ? R : 0.0f;
    float p_transmit = component.contains(BsdfComponentFlags::SPECULAR_TRANSMISSION()) ? T : 0.0f;
    float p_total = p_reflect + p_transmit;

    float u = sampler.sample_uniform();
    if (u * p_total < p_reflect)
    {
        float3 wi = reflect(wo, normal);
        float cos_theta = fabsf(wi.z);
        float f = R / cos_theta;
        float pdf = R / p_total;
        return BsdfSample {
            .wi = wi,
            .bsdf = make_float3(f, f, f),
            .pdf = pdf,
            .component = BsdfComponentFlags::SPECULAR_REFLECTION(),
            .valid = true
        };
    }
    else
    {
        auto refract_dir = refract(bsdf.eta, wo, normal);
        if (!refract_dir)
        {
            return {};
        }

        float3 wi = *refract_dir;
        float eta = wo.z < 0.0f ? 1.0f / bsdf.eta : bsdf.eta;
        float cos_theta = fabsf(wi.z);
        float f = (T / cos_theta) / (eta * eta);
        float pdf = T / p_total;
        return BsdfSample {
            .wi = wi,
            .bsdf = make_float3(f, f, f),
            .pdf = pdf,
            .component = BsdfComponentFlags::SPECULAR_TRANSMISSION(),
            .valid = true
        };
    }
}

inline __device__ BsdfSample sample_bsdf(
    OptixBsdfSmoothConductor bsdf, float3 wo, BsdfComponentFlags component, sample::OptixSampler& sampler
) {
    if (!component.contains(BsdfComponentFlags::SPECULAR_REFLECTION()))
    {
        return {};
    }

    float3 normal = make_float3(0.0f, 0.0f, 1.0f);
    float3 wi = reflect(wo, normal);

    float f_r = fresnel_complex(wo.z, complex(bsdf.eta.x, bsdf.kappa.x)) / wo.z;
    float f_g = fresnel_complex(wo.z, complex(bsdf.eta.y, bsdf.kappa.y)) / wo.z;
    float f_b = fresnel_complex(wo.z, complex(bsdf.eta.z, bsdf.kappa.z)) / wo.z;

    return BsdfSample {
        .wi = wi,
        .bsdf = make_float3(f_r, f_g, f_b),
        .pdf = 1.0f,
        .component = BsdfComponentFlags::SPECULAR_REFLECTION(),
        .valid = true
    };
}

// -- Trowbridge-Reitz microfacet distribution --
// @raytracing_cpu::materials::microfacet
namespace microfacet {

inline __device__ float distribution(float3 wm, float alpha_x, float alpha_y)
{
    float cos_theta_2 = wm.z * wm.z;
    float sin_theta_2 = 1.0f - cos_theta_2;
    float e = (wm.x * wm.x) / (alpha_x * alpha_x) + (wm.y * wm.y) / (alpha_y * alpha_y);
    float t = 1.0f + (sin_theta_2 / cos_theta_2) * e;
    return 1.0f / (M_PIf * alpha_x * alpha_y * cos_theta_2 * cos_theta_2 * t * t);
}

inline __device__ float lambda(float3 w, float alpha_x, float alpha_y)
{
    float cos_theta_2 = w.z * w.z;
    float sin_theta_2 = 1.0f - cos_theta_2;
    float tan_theta_2 = sin_theta_2 / cos_theta_2;
    float alpha_2 = alpha_x * alpha_x * w.x * w.x + alpha_y * alpha_y * w.y * w.y;
    return (sqrtf(1.0f + alpha_2 * tan_theta_2) - 1.0f) / 2.0f;
}

inline __device__ float G1(float3 w, float alpha_x, float alpha_y)
{
    return 1.0f / (1.0f + lambda(w, alpha_x, alpha_y));
}

inline __device__ float G(float3 wo, float3 wi, float alpha_x, float alpha_y)
{
    return 1.0f / (1.0f + lambda(wo, alpha_x, alpha_y) + lambda(wi, alpha_x, alpha_y));
}

inline __device__ float visible_distribution(float3 w, float3 wm, float alpha_x, float alpha_y)
{
    float cos_theta = fabsf(w.z);
    return (G1(w, alpha_x, alpha_y) / cos_theta)
        * distribution(wm, alpha_x, alpha_y)
        * fabsf(dot(w, wm));
}

inline __device__ float3 sample_wm(float3 w, float alpha_x, float alpha_y, float2 u)
{
    float3 wh = normalize(make_float3(alpha_x * w.x, alpha_y * w.y, w.z));
    if (wh.z < 0.0f) { wh = -wh; }

    float2 p = sample::sample_unit_disk(u);
    float3 t1 = (wh.z < 0.9999f)
        ? normalize(cross(make_float3(0.0f, 0.0f, 1.0f), wh))
        : make_float3(1.0f, 0.0f, 0.0f);
    float3 t2 = cross(wh, t1);

    float h = sqrtf(1.0f - p.x * p.x);
    float offset = 0.5f * h * (1.0f - wh.z);
    float scale = 0.5f * (1.0f + wh.z);
    p = make_float2(p.x, offset + scale * p.y);

    float pz = fmaxf(0.0f, 1.0f - dot(p, p));
    pz = sqrtf(pz);
    float3 nh = p.x * t1 + p.y * t2 + pz * wh;

    return normalize(make_float3(
        alpha_x * nh.x,
        alpha_y * nh.y,
        fmaxf(1.0e-6f, nh.z)
    ));
}

} // namespace microfacet

inline __device__ float3 evaluate_bsdf(OptixBsdfRoughConductor bsdf, float3 wo, float3 wi)
{
    if ((wo + wi) == float3_zero) { return float3_zero; }
    float3 wm = normalize(wo + wi);
    float cos_theta = fabsf(dot(wm, wi));

    float3 fresnel = make_float3(
        fresnel_complex(cos_theta, complex(bsdf.eta.x, bsdf.kappa.x)),
        fresnel_complex(cos_theta, complex(bsdf.eta.y, bsdf.kappa.y)),
        fresnel_complex(cos_theta, complex(bsdf.eta.z, bsdf.kappa.z))
    );

    float D = microfacet::distribution(wm, bsdf.alpha_x, bsdf.alpha_y);
    float Gv = microfacet::G(wo, wi, bsdf.alpha_x, bsdf.alpha_y);
    return fresnel * D * Gv / (4.0f * wo.z * wi.z);
}

inline __device__ float evaluate_pdf(OptixBsdfRoughConductor bsdf, float3 wo, float3 wi, BsdfComponentFlags component)
{
    if (!component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION()))
    {
        return 0.0f;
    }

    if ((wo + wi) == float3_zero) { return 0.0f; }
    float3 wm = normalize(wo + wi);
    if (wm.z < 0.0f) { wm = -wm; }

    return microfacet::visible_distribution(wo, wm, bsdf.alpha_x, bsdf.alpha_y)
        / (4.0f * fabsf(dot(wo, wm)));
}

inline __device__ BsdfSample sample_bsdf(
    OptixBsdfRoughConductor bsdf, float3 wo, BsdfComponentFlags component, sample::OptixSampler& sampler
) {
    if (!component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION()))
    {
        return {};
    }

    float2 u = sampler.sample_uniform2();
    float3 wm = microfacet::sample_wm(wo, bsdf.alpha_x, bsdf.alpha_y, u);
    float3 wi = reflect(wo, wm);

    if (wo.z * wi.z < 0.0f)
    {
        return {};
    }

    float pdf = evaluate_pdf(bsdf, wo, wi, component);
    float3 f = evaluate_bsdf(bsdf, wo, wi);

    return BsdfSample {
        .wi = wi,
        .bsdf = f,
        .pdf = pdf,
        .component = BsdfComponentFlags::NONSPECULAR_REFLECTION(),
        .valid = true
    };
}

inline __device__ float3 evaluate_bsdf(OptixBsdfRoughDielectric bsdf, float3 wo, float3 wi)
{
    using namespace microfacet;

    bool reflect = wo.z * wi.z > 0.0f;
    float eta_wm = reflect ? 1.0f : (wo.z > 0.0f ? bsdf.eta : 1.0f / bsdf.eta);

    float3 wm = wi * eta_wm + wo;
    wm = normalize(wm);
    if (wm.z < 0.0f) { wm = -wm; }

    if (wi.z == 0.0f || wo.z == 0.0f || wm == float3_zero)
    {
        return float3_zero;
    }

    if (dot(wm, wi) * wi.z < 0.0f || dot(wm, wo) * wo.z < 0.0f)
    {
        return float3_zero;
    }

    float F = fresnel_dielectric(
        dot(wo, wm),
        bsdf.eta
    );

    if (reflect)
    {
        float brdf = distribution(wm, bsdf.alpha_x, bsdf.alpha_y)
        * F
        * G(wo, wi, bsdf.alpha_x, bsdf.alpha_y)
        / fabsf(4.0f * wo.z * wi.z);

        return make_float3(brdf, brdf, brdf);
    }
    else
    {
        float x = dot(wi, wm) + dot(wo, wm) / eta_wm;
        float denom = wi.z * wo.z * x * x;
        float btdf = distribution(wm, bsdf.alpha_x, bsdf.alpha_y)
        * (1.0f - F)
        * G(wo, wi, bsdf.alpha_x, bsdf.alpha_y)
        * fabsf(dot(wi, wm) * dot(wo, wm) / denom)
        / (eta_wm * eta_wm);

        return make_float3(btdf, btdf, btdf);
    }
}

inline __device__ float evaluate_pdf(OptixBsdfRoughDielectric bsdf, float3 wo, float3 wi, BsdfComponentFlags component)
{
    using namespace microfacet;

    bool reflect = wo.z * wi.z > 0.0f;
    float eta_wm = reflect ? 1.0f : (wo.z > 0.0f ? bsdf.eta : 1.0f / bsdf.eta);

    float3 wm = wi * eta_wm + wo;
    wm = normalize(wm);
    if (wm.z < 0.0f) { wm = -wm; }

    if (wi.z == 0.0f || wo.z == 0.0f || wm == float3_zero)
    {
        return 0.0f;
    }

    if (dot(wm, wi) * wi.z < 0.0f || dot(wm, wo) * wo.z < 0.0f)
    {
        return 0.0f;
    }

    float R = fresnel_dielectric(dot(wo, wm), bsdf.eta);
    float T = 1.0f - R;

    float p_reflect = component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION()) ? R : 0.0f;
    float p_transmit = component.contains(BsdfComponentFlags::NONSPECULAR_TRANSMISSION()) ? T : 0.0f;
    float p_total = p_reflect + p_transmit;

    if (reflect)
    {
        return (p_reflect / p_total) * visible_distribution(wo, wm, bsdf.alpha_x, bsdf.alpha_y) / (4.0f * fabsf(dot(wo, wm)));
    }
    else
    {
        float x = dot(wi, wm) + dot(wo, wm) / eta_wm;
        float denom = x * x;
        float dwm_dwi = fabsf(dot(wi, wm)) / denom;

        return (p_transmit / p_total) * visible_distribution(wo, wm, bsdf.alpha_x, bsdf.alpha_y) * dwm_dwi;
    }
}

inline __device__ BsdfSample sample_bsdf(
    OptixBsdfRoughDielectric bsdf, float3 wo, BsdfComponentFlags component, sample::OptixSampler& sampler
) {
    using namespace microfacet;

    float2 u = sampler.sample_uniform2();
    float3 wm = sample_wm(wo, bsdf.alpha_x, bsdf.alpha_y, u);

    float R = fresnel_dielectric(dot(wo, wm), bsdf.eta);
    float T = 1.0f - R;

    float p_reflect = component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION()) ? R : 0.0f;
    float p_transmit = component.contains(BsdfComponentFlags::NONSPECULAR_TRANSMISSION()) ? T : 0.0f;
    float p_total = p_reflect + p_transmit;

    float3 wi;
    bool reflected;
    if (sampler.sample_uniform() * p_total < p_reflect)
    {
        wi = reflect(wo, wm);
        reflected = true;
        if (wo.z * wi.z < 0.0f)
        {
            return {};
        }
    }
    else
    {
        auto wi_opt = refract(bsdf.eta, wo, wm);
        if (!wi_opt)
        {
            return {};
        }

        wi = *wi_opt;
        reflected = false;
        if (wo.z * wi.z > 0.0f || wi.z == 0.0f)
        {
            return {};
        }
    }

    float pdf = evaluate_pdf(bsdf, wo, wi, component);
    float3 bsdf_value = evaluate_bsdf(bsdf, wo, wi);

    BsdfComponentFlags chosen_component = reflected ? BsdfComponentFlags::NONSPECULAR_REFLECTION() : BsdfComponentFlags::NONSPECULAR_TRANSMISSION();
    return BsdfSample {
        wi,
        .bsdf = bsdf_value,
        pdf,
        .component = chosen_component,
        .valid = true
    };
}

}
