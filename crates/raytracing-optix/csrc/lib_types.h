#pragma once

#ifdef __cplusplus
#include <cstddef>
#else
#include "stddef.h"
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
    union {
        struct {
            float screen_space_width;
            float screen_space_height;
        } orthographic;

        struct {
            float yfov;
        } pinhole_perspective;

        struct {
            float yfov;
            float aperture_radius;
            float focal_distance;
        } thin_lens_perspective;
    };
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