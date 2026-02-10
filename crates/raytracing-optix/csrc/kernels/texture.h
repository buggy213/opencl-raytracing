#pragma once

#include "kernel_math.h"
#include "kernel_params.h"
#include "types.h"

namespace texture
{

// internal function; do not use
template<typename F>
inline __device__ float4 _sample(F f, TextureId texture_id, float2 uv)
{
    static_assert(std::is_invocable_r_v<float4, F, TextureId, float2>, "(TextureId, float2) -> float4");

    Texture& texture = pipeline_params.textures[texture_id];
    switch (texture.kind)
    {
    case Texture::ImageTexture: {
        cudaTextureObject_t texture_object = texture.variant.image_texture.texture_object;
        return tex2D<float4>(texture_object, uv.x, uv.y);
    }
    case Texture::ConstantTexture: {
        Vec4 v = texture.variant.constant_texture.value;
        return vec4_to_float4(v);
    }
    case Texture::CheckerTexture: {
        Texture::TextureVariant::CheckerTexture tex = texture.variant.checker_texture;
        float u = uv.x - floorf(uv.x);
        float v = uv.y - floorf(uv.y);
        if ((u > 0.5f) != (v > 0.5f))
        {
            return vec4_to_float4(tex.color1);
        }
        else
        {
            return vec4_to_float4(tex.color2);
        }
    }
    case Texture::ScaleTexture: {
        Texture::TextureVariant::ScaleTexture tex = texture.variant.scale_texture;
        float4 a_val = f(tex.a, uv);
        float4 b_val = f(tex.b, uv);
        return a_val * b_val;
    }
    case Texture::MixTexture: {
        Texture::TextureVariant::MixTexture tex = texture.variant.mix_texture;
        float4 c_val = f(tex.c, uv);
        float4 a_val;
        float4 b_val;
        float4 one = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        if (c_val == float4_zero)
        {
            b_val = float4_zero;
        }
        else
        {
            b_val = f(tex.b, uv);
        }

        if (c_val == one)
        {
            a_val = float4_zero;
        }
        else
        {
            a_val = f(tex.a, uv);
        }

        return (one - c_val) * a_val + c_val * b_val;
    }
    default:
        return float4_zero;
    }
}

// internal function; do not use
inline __device__ float4 _leaf_sample(TextureId texture_id, float2 uv)
{
    // TODO: can we report an error somehow? or should make it the host's responsibility to legalize textures?
    auto leaf_terminate = [](TextureId, float2) -> float4 { return float4_zero; };
    return _sample(leaf_terminate, texture_id, uv);
}

// unlike the cpu raytracer, which uses recursion to evaluate arbitrarily nested textures,
// we only sample at most one "level" of indirection deep - recursion on GPU is expensive
// and more complex textures are very unlikely to be needed in practice
// @raytracing_cpu::texture::CpuTextures::sample
inline __device__ float4 sample(TextureId texture_id, float2 uv)
{
    auto sample_leaf = [](TextureId texture_id, float2 uv) { return _leaf_sample(texture_id, uv); };
    return _sample(sample_leaf, texture_id, uv);
}

}
