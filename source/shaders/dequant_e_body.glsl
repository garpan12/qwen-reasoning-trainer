// Per-element dequantization of a whole weight to fp32, Y[n*K + kk] = W[n, kk].
// One invocation per element (grid-stride), so consecutive lanes write
// consecutive floats (coalesced) and read neighbouring bytes of the same block.
// Uses the fused GEMMs' decoders (quant_decode.glsl), so the output matches what
// matmul_wq_* / matmul_t_wq_* decode element-for-element.

layout(local_size_x = 256) in;

layout(binding = 0) readonly  buffer W { uint data[]; };
layout(binding = 1) writeonly buffer Y { float y[]; };

layout(push_constant) uniform P {
    uint K;      // row length (elements)
    uint N;      // rows
    uint total;  // N * K
    uint pad;
} p;

#include "quant_decode.glsl"

void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint e = gl_GlobalInvocationID.x; e < p.total; e += stride) {
        uint n = e / p.K;
        uint kk = e - n * p.K;
        y[e] = QDECODE(n, kk);
    }
}
