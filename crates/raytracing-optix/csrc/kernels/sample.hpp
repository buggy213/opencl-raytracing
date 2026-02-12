#pragma once

// libcu++ and Optix-IR compilation don't play super nicely
// see https://github.com/NVIDIA/cccl/issues/1227
#define CCCL_DISABLE_INT128_SUPPORT
#include <cuda/std/optional>
#include <cuda/std/variant>

#include <curanddx.hpp>

#include "hash.hpp"
#include "kernel_math.hpp"
#include "types.h"

namespace sample {

// @raytracing_cpu::sample::permute
inline __device__ unsigned int permute(unsigned int index, unsigned int length, unsigned int seed) {
    unsigned int mask = 0;
    // unsigned int mask = cuda::next_power_of_two(length) - 1;

    // not great for thread divergence, but whatever
    while (true) {
        index ^= seed;
        index *= 0xe170893d;
        index ^= seed >> 16;
        index ^= (index & mask) >> 4;
        index ^= seed >> 8;
        index *= 0x0929eb3f;
        index ^= seed >> 23;
        index ^= (index & mask) >> 1;
        index *= (1 | seed >> 27);
        index *= 0x6935fa69;
        index ^= (index & mask) >> 11;
        index *= 0x74dcb303;
        index ^= (index & mask) >> 2;
        index *= 0x9e501cc3;
        index ^= (index & mask) >> 2;
        index *= 0xc860a3df;
        index &= mask;
        index ^= index >> 5;

        if (index < length) {
            return (index + seed) % length;
        }
    }
}

using RNG = decltype(curanddx::Generator<curanddx::pcg>() + curanddx::SM<860>() + curanddx::Thread());

struct OptixSamplerIndependent
{
    unsigned long long seed;
    RNG rng;
};

struct OptixSamplerStratified
{
    bool jitter;
    unsigned int x_strata;
    unsigned int y_strata;

    unsigned long long seed;
    RNG rng;

    unsigned int dimension;
    unsigned int sample_index;
};

struct OptixSampler
{
    using Variant = cuda::std::variant<OptixSamplerIndependent, OptixSamplerStratified>;
    Variant inner;

    // @raytracing_cpu::sample::CpuSampler::from_sampler
    __device__ static OptixSampler from_sampler(const Sampler& sampler, cuda::std::optional<unsigned long long> seed)
    {
        unsigned long long seed_unwrapped = seed.value_or(42);
        seed_unwrapped = hash_u32(seed_unwrapped);

        switch (sampler.kind)
        {
        case Sampler::Independent:
        default:
            return OptixSampler {
                OptixSamplerIndependent { .seed = seed_unwrapped, .rng = RNG(seed_unwrapped, 0, 0) }
            };
        case Sampler::Stratified:
            return OptixSampler {
                OptixSamplerStratified {
                    .jitter = sampler.variant.stratified.jitter,
                    .x_strata = sampler.variant.stratified.x_strata,
                    .y_strata = sampler.variant.stratified.y_strata,
                    .seed = seed_unwrapped,
                    .rng = RNG(seed_unwrapped, 0, 0),
                    .dimension = 0,
                    .sample_index = 0
                }
            };
        }
    }

    // @raytracing_cpu::sample::CpuSampler::one_off_sampler
    __device__ static OptixSampler one_off_sampler(uint64_t seed)
    {
        return OptixSampler {
            OptixSamplerIndependent {
                .seed = seed,
                .rng = RNG(seed, 0, 0)
            }
        };
    }

    // @raytracing_cpu::sample::CpuSampler::start_sample
    __device__ void start_sample(uint2 p, unsigned int sample_index)
    {
        unsigned long long hash = hash_u32(p.x);
        hash = hash_u32(p.y, hash);
        hash = hash_u32(sample_index, hash);

        unsigned long long stream = hash;
        struct Visitor
        {
            unsigned long long stream;
            unsigned int sample_index;

            __device__ void operator()(OptixSamplerIndependent& independent)
            {
                independent.rng = RNG(independent.seed, stream, 0);
            }
            __device__ void operator()(OptixSamplerStratified& stratified)
            {
                stratified.rng = RNG(stratified.seed, stream, 0);
                stratified.dimension = 0;
                stratified.sample_index = sample_index;
            }
        };
        cuda::std::visit(Visitor{ stream, sample_index }, inner);
    }

    __device__ float sample_uniform()
    {
        struct Visitor
        {
            __device__ float operator()(OptixSamplerIndependent& independent)
            {
                curanddx::uniform uniform {0.0f, 1.0f};
                return uniform.generate(independent.rng);
            }
            __device__ float operator()(OptixSamplerStratified& stratified)
            {
                unsigned int total_samples = stratified.x_strata * stratified.y_strata;

                curanddx::uniform uniform {0.0f, 1.0f};
                unsigned long long hash = hash_u32(stratified.dimension);
                hash = hash_u32(stratified.seed, hash);

                unsigned int strata = permute(stratified.sample_index, total_samples, hash);
                float delta = stratified.jitter ? uniform.generate(stratified.rng) : 0.5f;

                stratified.dimension += 1;
                return (strata + delta) / total_samples;
            }
        };
        return cuda::std::visit(Visitor{}, inner);
    }

    __device__ float2 sample_uniform2()
    {
        struct Visitor
        {
            __device__ float2 operator()(OptixSamplerIndependent& independent)
            {
                curanddx::uniform uniform {0.0f, 1.0f};
                float first = uniform.generate(independent.rng);
                float second = uniform.generate(independent.rng);
                return make_float2(first, second);
            }
            __device__ float2 operator()(OptixSamplerStratified& stratified)
            {
                curanddx::uniform uniform {0.0f, 1.0f};
                unsigned long long hash = hash_u32(stratified.dimension);
                hash = hash_u32(stratified.seed, hash);

                unsigned int total_samples = stratified.x_strata * stratified.y_strata;
                unsigned int strata = permute(stratified.sample_index, total_samples, hash);
                stratified.dimension += 2;

                unsigned int y = strata / stratified.x_strata;
                unsigned int x = strata % stratified.x_strata;
                float dx = stratified.jitter ? uniform.generate(stratified.rng) : 0.5f;
                float dy = stratified.jitter ? uniform.generate(stratified.rng) : 0.5f;

                return make_float2(
                    (x + dx) / stratified.x_strata,
                    (y + dy) / stratified.y_strata
                );
            }
        };
        return cuda::std::visit(Visitor{}, inner);
    }
};

// @raytracing_cpu::sample::sample_unit_disk
inline __device__ float2 sample_unit_disk(float2 u) {
    float r = sqrtf(u.x);
    float theta = 2.0f * M_PIf * u.y;
    return make_float2(r * cosf(theta), r * sinf(theta));
}

// @raytracing_cpu::sample::sample_unit_disk_concentric
inline __device__ float2 sample_unit_disk_concentric(float2 u) {
    float2 u_offset = 2.0 * u - make_float2(1.0f, 1.0f);
    if (u_offset == make_float2(0.0f, 0.0f)) {
        return float2_zero;
    }

    float theta;
    float r;
    if (abs(u_offset.x) > abs(u_offset.y)) {
        theta = M_PI_4f * (u_offset.y / u_offset.x);
        r = u_offset.x;
    }
    else {
        theta = M_PI_2f - M_PI_4f * (u_offset.x / u_offset.y);
        r = u_offset.y;
    }

    return r * make_float2(cosf(theta), sinf(theta));
}

// @raytracing_cpu::sample::sample_cosine_hemisphere
inline __device__ float3 sample_cosine_hemisphere(float2 u) {
    float2 d = sample_unit_disk(u);
    float z = sqrtf(fmaxf(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return make_float3(d.x, d.y, z);
}

// @raytracing_cpu::sample::sample_exponential
inline __device__ float sample_exponential(float u, float a) {
    return logf(1.0f - u) / a;
}
}
