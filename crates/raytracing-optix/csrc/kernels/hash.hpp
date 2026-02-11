#pragma once

#include "kernel_types.hpp"

// dead simple xor / shift based hash function. maybe not the fastest or most optimal hash function but should be
// good enough for any rendering task
inline __device__ u64 hash_u32(
    u32 data,
    u64 hash_state = 0x9e3779b97f4a7c15ULL
) {
    u64 data_long = data;
    data_long *= 0xbf58476d1ce4e5b9ULL;
    data_long ^= data_long >> 32;

    hash_state ^= data;
    hash_state *= 0x94d049bb133111ebULL;

    // final avalanche
    hash_state ^= hash_state >> 33;
    hash_state *= 0xff51afd7ed558ccdULL;
    hash_state ^= hash_state >> 33;

    return hash_state;
}
