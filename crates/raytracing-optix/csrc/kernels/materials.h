#pragma once

#define CCCL_DISABLE_INT128_SUPPORT
#include <cuda/std/optional>

#include "kernel_math.h"
#include "kernel_types.h"
#include "sample.h"

/// In the common case, we don't need the std::variant of the various BSDFs, since material dispatch
/// is handled by different closest-hit programs in the SBT. This is good for register usage / scheduling
/// We still do need it for more specialized (i.e. mix, layered) materials.
namespace materials
{
struct OptixBsdfDiffuse {
    float3 albedo;
};

struct OptixBsdfSmoothDielectric {
    float eta;
};

struct OptixBsdfSmoothConductor {
    float3 eta;
    float3 kappa;
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

enum class BsdfComponentFlags : u32 {
    NONSPECULAR_REFLECTION = 1 << 0,
    SPECULAR_REFLECTION = 1 << 1,
    NONSPECULAR_TRANSMISSION = 1 << 2,
    SPECULAR_TRANSMISSION = 1 << 3
};

inline __device__ BsdfComponentFlags operator|(BsdfComponentFlags a, BsdfComponentFlags b) {
    return static_cast<BsdfComponentFlags>(static_cast<u32>(a) | static_cast<u32>(b));
}

inline __device__ BsdfComponentFlags operator&(BsdfComponentFlags a, BsdfComponentFlags b) {
    return static_cast<BsdfComponentFlags>(static_cast<u32>(a) & static_cast<u32>(b));
}

inline __device__ BsdfComponentFlags& operator|=(BsdfComponentFlags& a, BsdfComponentFlags b) {
    a = a | b;
    return a;
}

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
    if (component & BsdfComponentFlags::NONSPECULAR_TRANSMISSION == 0)
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
    if (component & BsdfComponentFlags::NONSPECULAR_REFLECTION == 0)
    {
        return BsdfSample();
    }

    float2 u = sampler.sample_uniform2();
    float3 wi = sample::sample_cosine_hemisphere(u);
    float pdf = wi.z / M_PIf;

    return BsdfSample {
        .wi = wi,
        .bsdf = bsdf.albedo / M_PIf,
        .pdf = pdf,
        .component = BsdfComponentFlags::NONSPECULAR_REFLECTION,
        .valid = true
    };
}
}
