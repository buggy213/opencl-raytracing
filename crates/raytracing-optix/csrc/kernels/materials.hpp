#pragma once

#define CCCL_DISABLE_INT128_SUPPORT
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
};

struct OptixBsdfRoughDielectric {
    float eta;
    float alpha_x;
    float alpha_y;
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
    if ((component & BsdfComponentFlags::NONSPECULAR_TRANSMISSION()) == BsdfComponentFlags::EMPTY())
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

// BSDF sampling
// @raytracing_cpu::materials::CpuBsdf::sample_bsdf
inline __device__ BsdfSample sample_bsdf(
    OptixBsdfDiffuse bsdf, float3 wo, BsdfComponentFlags component, sample::OptixSampler& sampler
) {
    if ((component & BsdfComponentFlags::NONSPECULAR_REFLECTION()) == BsdfComponentFlags::EMPTY())
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
}
