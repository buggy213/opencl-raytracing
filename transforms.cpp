struct matrix4x4 {
    matrix4x4() {
        data[0][0] = data[1][1] = data[2][2] = data[3][3] = 1.0f;
        data[1][0] = data[2][0] = data[3][0] = \
        data[0][1] = data[2][1] = data[3][1] = \
        data[0][2] = data[1][2] = data[3][2] = \
        data[0][3] = data[1][3] = data[2][3] = 0.0f;
    }

    matrix4x4(float m11, float m12,) {
        
    }

    float data[4][4];
};

matrix4x4 transpose(matrix4x4& m) {
    
}

matrix4x4 

class transform {

    private:
        matrix4x4 m, m_inv;
};