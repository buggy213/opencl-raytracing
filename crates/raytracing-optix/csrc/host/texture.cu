#include "texture.hpp"
#include "lib_optix_types.h"
#include "types.h"
#include "util.hpp"

__host__ cudaChannelFormatDesc fromTextureFormat(TextureFormat fmt) {
    switch (fmt) {
        case R8:
            return cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        case RG8:
            return cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
        case RGBA8:
            return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        case R16:
            return cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
        case RG16:
            return cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned);
        case RGBA16:
            return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);

        case R32F:
            return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        case RG32F:
            return cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
        case RGBA32F:
            return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        default:
            return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
    }
}

__host__ cudaTextureDesc fromTextureSampler(TextureSampler sampler) {
    cudaTextureDesc desc;
    cudaTextureAddressMode address_mode {};

    switch (sampler.wrap) {
        case Repeat:
            address_mode = cudaAddressModeWrap;
            break;
        case Mirror:
            address_mode = cudaAddressModeMirror;
            break;
        case Clamp:
            address_mode = cudaAddressModeClamp;
            break;
    }

    desc.addressMode[0] = desc.addressMode[1] = desc.addressMode[2] = address_mode;

    cudaTextureFilterMode filter_mode {};
    switch (sampler.filter) {
        case Nearest:
            filter_mode = cudaFilterModePoint;
            break;
        case Bilinear:
        case Trilinear:
            filter_mode = cudaFilterModeLinear;
            break;
    }

    desc.filterMode = filter_mode;

    // always want tex2D to return floating-point values
    desc.readMode = cudaReadModeNormalizedFloat;

    // we assume the host has already done this
    desc.sRGB = false;
    desc.borderColor[0] = desc.borderColor[1] = desc.borderColor[2] = desc.borderColor[3] = 0.0f;

    // u, v in [0.0, 1.0)
    desc.normalizedCoords = true;

    // for now, we don't support mipmapping (the backing array doesn't support it either)
    desc.maxAnisotropy = 1;
    desc.mipmapFilterMode = cudaFilterModePoint;
    desc.mipmapLevelBias = 0;
    desc.minMipmapLevelClamp = 0;
    desc.maxMipmapLevelClamp = 0;
    desc.disableTrilinearOptimization = true;
    desc.seamlessCubemap = false;

    return desc;
}

// Requirement is that host-side memory is row-major w/ specified pitch and channels are packed
// @perf: very inefficient to be called in a loop to upload all textures. can definitely be smarter
__host__ cudaArray_t createCudaArray(const void* src, size_t pitch, size_t width, size_t height, TextureFormat fmt) {
    cudaArray_t backing_array;
    cudaChannelFormatDesc channel_format = fromTextureFormat(fmt);

    CUDA_CHECK(cudaMallocArray(&backing_array, &channel_format, width, height));

    cudaError_t err = cudaMemcpy2DToArray(
        backing_array,
        0,
        0,
        src,
        pitch,
        pitch,
        height,
        cudaMemcpyHostToDevice
    );

    CUDA_CHECK(err);

    return backing_array;
}

__host__ cudaTextureObject_t createCudaTexture(cudaArray_t backing_array, TextureSampler sampler) {
    cudaTextureObject_t texture;
    cudaResourceDesc resource_desc { .resType = cudaResourceTypeArray, .res = backing_array, .flags = 0 };
    cudaTextureDesc texture_desc = fromTextureSampler(sampler);

    CUDA_CHECK(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr));

    return texture;
}
