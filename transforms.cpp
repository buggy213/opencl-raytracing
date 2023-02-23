#include <CL/opencl.hpp>
#include "transforms.h"

matrix4x4::matrix4x4() {
    data[0][0] = data[1][1] = data[2][2] = data[3][3] = 1.0f;
    data[1][0] = data[2][0] = data[3][0] = \
    data[0][1] = data[2][1] = data[3][1] = \
    data[0][2] = data[1][2] = data[3][2] = \
    data[0][3] = data[1][3] = data[2][3] = 0.0f;
}

matrix4x4::matrix4x4(float m11, float m12, float m13, float m14,
            float m21, float m22, float m23, float m24,
            float m31, float m32, float m33, float m34,
            float m41, float m42, float m43, float m44) {
    data[0][0] = m11;
    data[0][1] = m12;
    data[0][2] = m13;
    data[0][3] = m14;
    data[1][0] = m21;
    data[1][1] = m22;
    data[1][2] = m23;
    data[1][3] = m24;
    data[2][0] = m31;
    data[2][1] = m32;
    data[2][2] = m33;
    data[2][3] = m34;
    data[3][0] = m41;
    data[3][1] = m42;
    data[3][2] = m43;
    data[3][3] = m44;
}

cl_matrix4x4 matrix4x4::to_cl(matrix4x4& m) {
    cl_matrix4x4 cl_matrix;
    cl_matrix.s[0]  = m.data[0][0];
    cl_matrix.s[1]  = m.data[0][1];
    cl_matrix.s[2]  = m.data[0][2];
    cl_matrix.s[3]  = m.data[0][3];
    cl_matrix.s[4]  = m.data[1][0];
    cl_matrix.s[5]  = m.data[1][1];
    cl_matrix.s[6]  = m.data[1][2];
    cl_matrix.s[7]  = m.data[1][3];
    cl_matrix.s[8]  = m.data[2][0];
    cl_matrix.s[9]  = m.data[2][1];
    cl_matrix.s[10] = m.data[2][2];
    cl_matrix.s[11] = m.data[2][3];
    cl_matrix.s[12] = m.data[3][0];
    cl_matrix.s[13] = m.data[3][1];
    cl_matrix.s[14] = m.data[3][2];
    cl_matrix.s[15] = m.data[3][3];
    return cl_matrix;
}

matrix4x4 transpose(matrix4x4& m) {
    return matrix4x4(
        m.data[0][0], m.data[1][0], m.data[2][0], m.data[3][0],
        m.data[0][1], m.data[1][1], m.data[2][1], m.data[3][1],
        m.data[0][2], m.data[1][2], m.data[2][2], m.data[3][2],
        m.data[0][3], m.data[1][3], m.data[2][3], m.data[3][3]
    );
}

matrix4x4 multiply(matrix4x4& a, matrix4x4& b) {
    matrix4x4 c;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)  {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a.data[i][k] * b.data[k][j];
            }
            c.data[i][j] = sum;
        }
    }
    return c;
}

transform transform::translate(const vec3& direction) {
    matrix4x4 m {
        1.0f, 0.0f, 0.0f, direction.x,
        0.0f, 1.0f, 0.0f, direction.y,
        0.0f, 0.0f, 1.0f, direction.z,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    matrix4x4 m_inverse {
        1.0f, 0.0f, 0.0f, direction.x,
        0.0f, 1.0f, 0.0f, direction.y,
        0.0f, 0.0f, 1.0f, direction.z,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    return transform{m, m_inverse};
}