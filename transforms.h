#pragma once

#include <CL/opencl.hpp>
#include "vec3.h"

typedef cl_float16 cl_matrix4x4;

struct matrix4x4 {
    matrix4x4();
    matrix4x4(float m11, float m12, float m13, float m14,
              float m21, float m22, float m23, float m24,
              float m31, float m32, float m33, float m34,
              float m41, float m42, float m43, float m44);

    static cl_matrix4x4 to_cl(matrix4x4& m);
    float data[4][4];
};

matrix4x4 transpose(matrix4x4& m);
matrix4x4 multiply(matrix4x4& a, matrix4x4& b);

class transform {
    public:
        transform(matrix4x4 m, matrix4x4 inv) : m(m), m_inv(inv) {};
        static transform translate(const vec3& direction);
        matrix4x4 m, m_inv;
};

struct cl_transform {
    cl_matrix4x4 m;
    cl_matrix4x4 inverse;

    cl_transform(transform t) : m(matrix4x4::to_cl(t.m)), inverse(matrix4x4::to_cl(t.m_inv)) {}
};