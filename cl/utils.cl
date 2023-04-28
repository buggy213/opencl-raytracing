#ifndef RT_UTILS
#define RT_UTILS
// copied from https://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
} // how does this work ???

// returns a random floating point number in the range [0, 1)
float rand_float(uint2* rng_state) {
    uint rand_uint = MWC64X(rng_state);
    // S01111111MMMMMMMMMMMMMMMMMMMMMMM. sign is 0, mantissa is random. this generates a number in the range [1, 2)
    union { uint u32; float f; } u = { .u32 = rand_uint >> 9 | 0x3f800000 }; 
    return u.f - 1.0f; // offset by -1.0 to get it in range [0, 1) 
}
#endif