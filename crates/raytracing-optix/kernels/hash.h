#pragma once

// dead simple xor / shift based hash function. maybe not the fastest or most optimal hash function but should be
// good enough for any rendering task
inline __device__ unsigned long long hash_u32(
    unsigned int data,
    unsigned long long hash_state = 0x9e3779b97f4a7c15ULL
) {
    data *= 0xbf58476d1ce4e5b9ULL;
    data ^= data >> 32;

    hash_state ^= data;
    hash_state *= 0x94d049bb133111ebULL;

    // final avalanche
    hash_state ^= hash_state >> 33;
    hash_state *= 0xff51afd7ed558ccdULL;
    hash_state ^= hash_state >> 33;

    return hash_state;
}
