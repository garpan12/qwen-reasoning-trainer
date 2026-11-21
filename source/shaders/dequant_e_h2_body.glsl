// Per-element dequantization of a whole weight to packed fp16 hi/lo parts for the
// pre-split tensor-core GEMMs (matmul_h2_body.glsl). W[n, kk] = hi + lo with hi
// the fp32 value truncated to fp16's 10 mantissa bits (integer masking: the
// NVIDIA compiler folds float16 round-trips) and lo = W - hi.
// Layout (uint = 2 halves): hi[e/2] for e = n*K + kk, then lo at total/2 + e/2.

layout(local_size_x = 256) in;

layout(binding = 0) readonly  buffer W { uint data[]; };
layout(binding = 1) writeonly buffer Y { uint y[]; };

layout(push_constant) uniform P {
    uint K;      // row length (elements, even)
    uint N;      // rows
    uint total;  // N * K
    uint pad;
} p;

#include "quant_decode.glsl"

void main() {
    uint half_ = p.total >> 1u;
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint e2 = gl_GlobalInvocationID.x; e2 < half_; e2 += stride) {
        uint e = e2 * 2u;
        uint n = e / p.K;
        uint kk = e - n * p.K;
        float v0 = QDECODE(n, kk);
        float v1 = QDECODE(n, kk + 1u);
        float h0 = uintBitsToFloat(floatBitsToUint(v0) & 0xFFFFE000u);
        float h1 = uintBitsToFloat(floatBitsToUint(v1) & 0xFFFFE000u);
        y[e2] = packHalf2x16(vec2(h0, h1));
        y[half_ + e2] = packHalf2x16(vec2(v0 - h0, v1 - h1));
    }
}
