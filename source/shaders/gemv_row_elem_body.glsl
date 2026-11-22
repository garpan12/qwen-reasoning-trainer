// Subgroup row GEMV over a per-element decoder QDECODE(n, kk) (quant_decode.glsl):
// y[n] = sum_k dequant(W[n,k]) * x[k]. One 32-lane subgroup per output row (4
// rows per 128-invocation workgroup); lanes take consecutive elements, so the
// quant bytes of a block are read by neighbouring lanes.

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 128) in;

layout(binding = 0) readonly  buffer W { uint data[]; };
layout(binding = 1) readonly  buffer X { float x[]; };
layout(binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform P {
    uint K;
    uint N;
    uint rowBase;
} p;

#include "quant_decode.glsl"

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint n = p.rowBase + gl_WorkGroupID.x * 4u + gl_SubgroupID;
    if (n >= p.N) {
        return;
    }
    float a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    uint kk = lane;
    for (; kk + 96u < p.K; kk += 128u) {
        a0 += QDECODE(n, kk) * x[kk];
        a1 += QDECODE(n, kk + 32u) * x[kk + 32u];
        a2 += QDECODE(n, kk + 64u) * x[kk + 64u];
        a3 += QDECODE(n, kk + 96u) * x[kk + 96u];
    }
    for (; kk < p.K; kk += 32u) {
        a0 += QDECODE(n, kk) * x[kk];
    }
    float acc = subgroupAdd((a0 + a1) + (a2 + a3));
    if (lane == 0u) {
        y[n] = acc;
    }
}
