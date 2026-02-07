#pragma once
#include "types.h"

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