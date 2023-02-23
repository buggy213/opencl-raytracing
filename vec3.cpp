#include "vec3.h"

float vec3::operator[](int idx) {
    if (idx == 0) {
        return (*this).x;   
    }
    if (idx == 1) {
        return (*this).y;
    }
    if (idx == 2) {
        return (*this).z;
    }

    return 0.0f;
}

vec3 operator+(const vec3& a, const vec3& b) {
    return vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 operator-(const vec3& a, const vec3& b) {
    return vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 operator*(const vec3& a, const vec3& b) {
    return vec3{a.x * b.x, a.y * b.y, a.z * b.z};
}

vec3 operator/(const vec3& a, const vec3& b) {
    return vec3{a.x / b.x, a.y / b.y, a.z / b.z};
}

float vec3::dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 vec3::cross(const vec3& a, const vec3& b) {
    return vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}