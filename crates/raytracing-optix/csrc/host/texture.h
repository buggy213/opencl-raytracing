#pragma once

#include <cuda.h>

#include "types.h"

cudaArray_t createCudaArray(const void* src, size_t pitch, size_t width, size_t height, TextureFormat fmt);
cudaTextureObject_t createCudaTexture(cudaArray_t backing_array, TextureSampler sampler);
CUdeviceptr uploadOptixTexturesImpl(const Texture* textures, size_t count);