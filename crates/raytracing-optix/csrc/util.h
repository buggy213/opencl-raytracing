#pragma once

#include <cstdio>
#include <optix_stubs.h>
#include <optix_types.h>

inline void cudaAssert(cudaError code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorName(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#define CUDA_CHECK(code) { cudaAssert((code), __FILE__, __LINE__); }

inline void optixAssert(OptixResult code, const char *file, int line, bool abort = true) {
    if (code != OPTIX_SUCCESS) {
        fprintf(stderr, "OptiX error %s at %s:%d\n", optixGetErrorName(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

#define OPTIX_CHECK(code) { optixAssert((code), __FILE__, __LINE__); }