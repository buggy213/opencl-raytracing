#pragma once
#include "types.h"

// conversion between scene-description types and native CUDA types
inline __device__ float4 vec4_to_float4(Vec4 v)
{
    return make_float4(v.x, v.y, v.z, v.w);
}

inline __device__ float3 vec3_to_float3(Vec3 v)
{
    return make_float3(v.x, v.y, v.z);
}

inline __device__ float2 vec2_to_float2(Vec2 v)
{
    return make_float2(v.x, v.y);
}

inline __device__ uint3 vec3u_to_uint3(Vec3u v)
{
    return make_uint3(v.x, v.y, v.z);
}

inline __device__ uint2 vec2u_to_uint2(Vec2u v)
{
    return make_uint2(v.x, v.y);
}

inline __device__ Vec4 float4_to_vec4(float4 v)
{
    return Vec4 { v.x, v.y, v.z, v.w };
}

inline __device__ Vec3 float3_to_vec3(float3 v)
{
    return Vec3 { v.x, v.y, v.z };
}

inline __device__ Vec2 float2_to_vec2(float2 v)
{
    return Vec2 { v.x, v.y };
}

inline __device__ Vec3u uint3_to_vec3u(uint3 v)
{
    return Vec3u { v.x, v.y, v.z };
}

inline __device__ Vec2u uint2_to_vec2u(uint2 v)
{
    return Vec2u { v.x, v.y };
}

inline constexpr float4 float4_zero = float4 {0.0f, 0.0f, 0.0f, 0.0f };
inline constexpr float3 float3_zero = float3 {0.0f, 0.0f, 0.0f };
inline constexpr float2 float2_zero = float2 {0.0f, 0.0f };

inline __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __device__ float4& operator*=(float4& lhs, const float4& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    lhs.w *= rhs.w;
    return lhs;
}

inline __device__ float3& operator*=(float3& lhs, const float3& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}

inline __device__ float2& operator*=(float2& lhs, const float2& rhs)
{
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    return lhs;
}

inline __device__ float4& operator+=(float4& lhs, const float4& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

inline __device__ float3& operator+=(float3& lhs, const float3& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

inline __device__ float2& operator+=(float2& lhs, const float2& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

inline __device__ float4& operator-=(float4& lhs, const float4& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}

inline __device__ float3& operator-=(float3& lhs, const float3& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

inline __device__ float2& operator-=(float2& lhs, const float2& rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}

inline __device__ float4& operator/=(float4& lhs, float rhs)
{
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    lhs.w /= rhs;
    return lhs;
}

inline __device__ float3& operator/=(float3& lhs, float rhs)
{
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    return lhs;
}

inline __device__ float2& operator/=(float2& lhs, float rhs)
{
    lhs.x /= rhs;
    lhs.y /= rhs;
    return lhs;
}

inline __device__ float4& operator*=(float4& lhs, float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    lhs.w *= rhs;
    return lhs;
}

inline __device__ float3& operator*=(float3& lhs, float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

inline __device__ float2& operator*=(float2& lhs, float rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    return lhs;
}

inline __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator*(float a, float4 b) {
    return b * a;
}

inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator*(float a, float3 b) {
    return b * a;
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator*(float a, float2 b) {
    return b * a;
}

inline __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

inline __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float4 operator-(float4 a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ float3 operator-(float3 a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ float2 operator-(float2 a)
{
    return make_float2(a.x, -a.y);
}

inline __device__ bool operator==(float4 a, float4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __device__ bool operator!=(float4 a, float4 b) {
    return !(a == b);
}

inline __device__ bool operator==(float3 a, float3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __device__ bool operator!=(float3 a, float3 b) {
    return !(a == b);
}

inline __device__ bool operator==(float2 a, float2 b) {
    return a.x == b.x && a.y == b.y;
}

inline __device__ bool operator!=(float2 a, float2 b) {
    return !(a == b);
}

inline __device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

inline __device__ float length(float4 a)
{
    return sqrtf(dot(a, a));
}

inline __device__ float length(float3 a)
{
    return sqrtf(dot(a, a));
}

inline __device__ float length(float2 a)
{
    return sqrtf(dot(a, a));
}

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float3 normalize(float3 a) {
    float inv_len = rsqrtf(dot(a, a));
    return a * inv_len;
}

// @raytracing::geometry::Matrix4x4::apply_point
inline __device__ float3 matrix4x4_apply_point(const Matrix4x4& transform, float3 point)
{
    float4 v = make_float4(point.x, point.y, point.z, 1.0f);
    float4 a = make_float4(transform.m[0], transform.m[1], transform.m[2], transform.m[3]);
    float4 b = make_float4(transform.m[4], transform.m[5], transform.m[6], transform.m[7]);
    float4 c = make_float4(transform.m[8], transform.m[9], transform.m[10], transform.m[11]);
    float4 d = make_float4(transform.m[12], transform.m[13], transform.m[14], transform.m[15]);

    float va = dot(a, v);
    float vb = dot(b, v);
    float vc = dot(c, v);
    float vd = dot(d, v);

    return make_float3(va / vd, vb / vd, vc / vd);
}

// @raytracing::geometry::Matrix4x4::apply_vector
inline __device__ float3 matrix4x4_apply_vector(const Matrix4x4& transform, float3 vector)
{
    float3 v = make_float3(vector.x, vector.y, vector.z);
    float3 a = make_float3(transform.m[0], transform.m[1], transform.m[2]);
    float3 b = make_float3(transform.m[4], transform.m[5], transform.m[6]);
    float3 c = make_float3(transform.m[8], transform.m[9], transform.m[10]);

    float va = dot(a, v);
    float vb = dot(b, v);
    float vc = dot(c, v);

    return make_float3(va, vb, vc);
}

// cuda::std::complex is problematicaly slow, so we just roll our own
// https://github.com/NVIDIA/cccl/issues/1000
// @crates/raytracing/src/geometry/complex.rs
struct complex
{
    float real;
    float imag;

    __device__ complex() : real(0.0f), imag(0.0f) {}
    __device__ complex(float re, float im) : real(re), imag(im) {}
    // implicit conversion from float is "ok"
    __device__ complex(float re) : real(re), imag(0.0f) {}

    __device__ float square_magnitude() const { return real * real + imag * imag; }

    __device__ float modulus() const { return sqrtf(square_magnitude()); }

    static __device__ complex from_polar(float r, float theta) {
        return { r * cosf(theta), r * sinf(theta) };
    }
    __device__ complex sqrt() const
    {
        float r = modulus();
        float theta = atan2f(imag, real);
        return from_polar(sqrtf(r), theta * 0.5f);
    }

    // Squared magnitude, matching cuda::std::norm
    __device__ float norm() const { return square_magnitude(); }

    __device__ complex operator+(complex b) const { return { real + b.real, imag + b.imag }; }
    __device__ complex operator+(float b) const { return { real + b, imag }; }
    __device__ void operator+=(complex b) { real += b.real; imag += b.imag; }
    __device__ void operator+=(float b) { real += b; }

    __device__ complex operator-() const { return { -real, -imag }; }
    __device__ complex operator-(complex b) const { return { real - b.real, imag - b.imag }; }
    __device__ complex operator-(float b) const { return { real - b, imag }; }
    __device__ void operator-=(complex b) { real -= b.real; imag -= b.imag; }
    __device__ void operator-=(float b) { real -= b; }

    __device__ complex operator*(complex b) const {
        return { real * b.real - imag * b.imag, imag * b.real + real * b.imag };
    }
    __device__ complex operator*(float b) const { return { real * b, imag * b }; }
    __device__ void operator*=(complex b) { *this = *this * b; }
    __device__ void operator*=(float b) { real *= b; imag *= b; }

    __device__ complex operator/(complex b) const {
        float denom = b.real * b.real + b.imag * b.imag;
        return {
            (real * b.real + imag * b.imag) / denom,
            (imag * b.real - real * b.imag) / denom
        };
    }
    __device__ complex operator/(float b) const { return { real / b, imag / b }; }
    __device__ void operator/=(complex b) { *this = *this / b; }
    __device__ void operator/=(float b) { real /= b; imag /= b; }
};

__device__ inline complex operator+(float a, complex b) { return b + a; }
__device__ inline complex operator-(float a, complex b) { return complex(a, 0.0f) - b; }
__device__ inline complex operator*(float a, complex b) { return b * a; }
__device__ inline complex operator/(float a, complex b) { return complex(a, 0.0f) / b; }
