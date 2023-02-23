#pragma once

class vec3 {
    public:
        vec3(float x, float y, float z) : x(x), y(y), z(z) {};
        float operator[](int idx); 
        
        static float dot(const vec3& a, const vec3& b);
        static vec3 cross(const vec3& a, const vec3& b);

        float x, y, z;
        
};

vec3 operator+(const vec3& a, const vec3& b);
vec3 operator*(const vec3& a, const vec3& b);
vec3 operator-(const vec3& a, const vec3& b);
vec3 operator/(const vec3& a, const vec3& b);