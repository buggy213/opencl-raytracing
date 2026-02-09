#pragma once

/*
 * Types shared between host C++/Rust code and device OptiX kernels
 */

#ifdef __cplusplus
#include <cstddef>
#else
#include "stddef.h"
#include "stdbool.h"
#endif

/* Vocabulary types */
struct Vec2 {
    float x;
    float y;
};

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Vec4 {
    float x;
    float y;
    float z;
    float w;
};

struct Quaternion {
    float real;
    struct Vec3 pure;
};

struct Vec2u {
    unsigned int x;
    unsigned int y;
};

struct Vec3u {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct Matrix4x4 {
    float m[16];
};

struct Transform {
    struct Matrix4x4 forward;
    struct Matrix4x4 inverse;
};

/* Scene description types */

// @raytracing::scene::camera::CameraType
enum CameraTypeKind {
    Orthographic,
    PinholePerspective,
    ThinLensPerspective,
};

struct CameraType {
    enum CameraTypeKind kind;
    union CameraVariant {
        struct Orthographic {
            float screen_space_width;
            float screen_space_height;
        } orthographic;

        struct PinholePerspective {
            float yfov;
        } pinhole_perspective;

        struct ThinLensPerspective {
            float yfov;
            float aperture_radius;
            float focal_distance;
        } thin_lens_perspective;
    } variant;
};

// @raytracing::scene::camera::Camera
struct Camera {
    struct Vec3 camera_position;
    struct Quaternion camera_rotation;

    struct CameraType camera_type;
    size_t raster_width;
    size_t raster_height;
    float near_clip;
    float far_clip;

    struct Transform world_to_raster;

    struct Transform camera_to_world;
    struct Transform raster_to_camera;
};

// @raytracing::sampling::Sampler
struct Sampler
{
    enum SamplerKind { Independent, Stratified } kind;
    union SamplerVariant
    {
        struct Independent {} independent;
        struct Stratified
        {
            bool jitter;
            unsigned int x_strata;
            unsigned int y_strata;
        } stratified;
    } variant;
};

// @raytracing::lights::Light
struct Light
{
    enum LightKind { PointLight, DirectionLight, DiffuseAreaLight } kind;
    union LightVariant
    {
        struct PointLight
        {
            struct Vec3 position;
            struct Vec3 intensity;
        } point_light;
        struct DirectionLight
        {
            struct Vec3 direction;
            struct Vec3 radiance;
        } direction_light;
        struct DiffuseAreaLight
        {
            unsigned int prim_id;

            struct Vec3 radiance;
            struct Matrix4x4 light_to_world;
        } area_light;
    } variant;
};

// corresponds roughly to @raytracing::renderer::RaytracerSettings
// but with some fields handled by the host side (e.g. default seed, AovFlags, max_ray_depth)
struct OptixRaytracerSettings {
    bool accumulate_bounces;

    unsigned int light_sample_count;
    unsigned int samples_per_pixel;
    unsigned long long seed;
    struct Sampler sampler;
};

// corresponds to `image` crate's `ColorType` enum
// note that CUDA textures do *not* support 3 channel types (e.g. uchar3, ushort3, float3).
// caller must pad w/ alpha channel
enum TextureFormat {
    R8,
    RG8,
    RGBA8,

    R16,
    RG16,
    RGBA16,

    R32F,
    RG32F,
    RGBA32F,
};

// @raytracing::materials::texture::FilterMode
enum FilterMode {
    Nearest,
    Bilinear,
    Trilinear
};

// @raytracing::materials::texture::WrapMode
enum WrapMode {
    Repeat,
    Mirror,
    Clamp
};

// @raytracing::materials::texture::TextureSampler
struct TextureSampler {
    enum FilterMode filter;
    enum WrapMode wrap;
};

// @raytracing::materials::texture::Texture
struct Texture {
    enum TextureKind { ImageTexture, ConstantTexture, CheckerTexture, ScaleTexture, MixTexture } kind;
    union TextureVariant {
        struct ImageTexture {
            unsigned long long texture_object;
        } image_texture;
        struct ConstantTexture {
            struct Vec4 value;
        } constant_texture;
        struct CheckerTexture {
            struct Vec4 color1;
            struct Vec4 color2;
        } checker_texture;

        struct ScaleTexture {
            unsigned int a;
            unsigned int b;
        } scale_texture;
        struct MixTexture {
            unsigned int a;
            unsigned int b;
            unsigned int c;
        } mix_texture;
    } variant;
};

// @raytracing::materials::texture::TextureId
typedef unsigned int TextureId;

// @raytracing::scene::scene::Scene
// scene hierarchy is mapped onto OptiX AS-hierarchy
// materials are stored inline in SBT hitgroup record payload
struct Scene {
    const struct Camera* camera;

    size_t num_lights;
    const struct Light* lights;

    size_t num_textures;
    const struct Texture* textures;
};

// @raytracing::materials::Material
struct Material {
    enum MaterialKind {
        Diffuse,
        SmoothDielectric,
        SmoothConductor,
        RoughDielectric,
        RoughConductor,
        CoatedDiffuse
    } kind;

    union MaterialVariant {
        struct Diffuse {
            TextureId albedo;
        } diffuse;
        // TODO: add other variants
    } variant;
};
