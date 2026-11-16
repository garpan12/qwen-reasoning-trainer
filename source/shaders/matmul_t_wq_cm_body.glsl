// Tensor-core (VK_KHR_cooperative_matrix) version of matmul_t_wq_body.glsl:
//   O[k,m] = sum_r dequant(W[n0+r, k]) * D[n0+r, m]     (stored o[k + m*K])
//
// Same bindings, push constants (incl. the n0/rows k-split and acc flag), 64x64
// output tile and 256-invocation workgroup as the fp32 body. Requires subgroup
// size 32 and 16x16x16 f16*f16+f32 cooperative matrices.
//
// The gradient tile D gets the same power-of-two scaling + hi/lo fp16 split as
// the activation tile in matmul_wq_cm_body.glsl (gradients are small, so plain
// fp16 would underflow); weights are dequantized to fp16; accumulation is fp32.

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) readonly buffer W { uint data[]; };
layout(binding = 1) readonly buffer D { float d[]; };
layout(binding = 2)          buffer O { float o[]; };

layout(push_constant) uniform P {
    uint K;
    uint N;
    uint M;
    uint n0;
    uint rows;
    uint acc;
} p;

#include "quant_decode.glsl"

#define TM 64            // output k per workgroup
#define TN 64            // output m per workgroup
#define TK 32            // reduction step over rows r
#define AST (TK + 8)     // f16 row stride of the gradient tile [m][r]
#define BST (TM + 8)     // f16 row stride of the weight tile [r][k]
#define CST (TM + 4)     // f32 row stride of the output staging tile [m][k]

shared float16_t Ahi[TN * AST];
shared float16_t Alo[TN * AST];
shared float16_t Bw[TK * BST];
shared float16_t Bwl[TK * BST];   // weight lo part (dequantized values aren't exact in fp16)
shared float Cs[TN * CST];
shared float smax[8];

#define ACC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MATA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MATB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>

void main() {
    uint tid = gl_LocalInvocationID.y * 16u + gl_LocalInvocationID.x;
    uint k0 = gl_WorkGroupID.x * TM;
    uint m0 = gl_WorkGroupID.y * TN;
    uint sg = gl_SubgroupID;
    uint sm = sg & 3u;
    uint sn = sg >> 2u;

    ACC acc0 = ACC(0.0);
    ACC acc1 = ACC(0.0);

    for (uint r0 = 0u; r0 < p.rows; r0 += TK) {
        // Gradient tile A[m][r] = D[n0+r0+r, m0+m]; consecutive lanes walk r (contiguous).
        float av[8];
        float lmax = 0.0;
        for (uint i = 0u; i < 8u; i++) {
            uint idx = tid + i * 256u;
            uint m = idx / TK;
            uint r = idx - m * TK;
            uint rr = r0 + r;
            uint mm = m0 + m;
            float v = (rr < p.rows && mm < p.M) ? d[(p.n0 + rr) + mm * p.N] : 0.0;
            av[i] = v;
            lmax = max(lmax, abs(v));
        }
        lmax = subgroupMax(lmax);
        if (subgroupElect()) {
            smax[sg] = lmax;
        }
        // Weight tile Bw[r][k] = dequant(W[n0+r0+r, k0+k]); decode walks k.
        for (uint i = 0u; i < 8u; i++) {
            uint idx = tid + i * 256u;
            uint r = idx / TM;
            uint k = idx - r * TM;
            uint rr = r0 + r;
            uint kk = k0 + k;
            float wv = (rr < p.rows && kk < p.K) ? QDECODE(p.n0 + rr, kk) : 0.0;
            float wh = uintBitsToFloat(floatBitsToUint(wv) & 0xFFFFE000u);
            Bw[r * BST + k] = float16_t(wh);
            Bwl[r * BST + k] = float16_t(wv - wh);
        }
        barrier();

        float tmax = max(max(max(smax[0], smax[1]), max(smax[2], smax[3])),
                         max(max(smax[4], smax[5]), max(smax[6], smax[7])));
        float e = tmax > 0.0 ? clamp(14.0 - ceil(log2(tmax)), -100.0, 100.0) : 0.0;
        float s = exp2(e);
        for (uint i = 0u; i < 8u; i++) {
            uint idx = tid + i * 256u;
            uint m = idx / TK;
            uint r = idx - m * TK;
            // Split by truncating the fp32 mantissa to fp16's 10 bits with integer ops.
            // Float round-trips (float16_t(x), packHalf2x16) get folded by the NVIDIA
            // compiler, which silently zeroes the lo part. The scaled tile max is 2^14,
            // so hf is a normal fp16 value (or a tiny value whose loss is negligible).
            float x = av[i] * s;
            float hf = uintBitsToFloat(floatBitsToUint(x) & 0xFFFFE000u);
            Ahi[m * AST + r] = float16_t(hf);
            Alo[m * AST + r] = float16_t(x - hf);
        }
        barrier();

        ACC t0 = ACC(0.0);
        ACC t1 = ACC(0.0);
        for (uint kk = 0u; kk < TK; kk += 16u) {
            MATA ah;
            MATA al;
            MATB b0;
            MATB b1;
            MATB l0;
            MATB l1;
            coopMatLoad(ah, Ahi, (sm * 16u) * AST + kk, AST, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(al, Alo, (sm * 16u) * AST + kk, AST, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(b0, Bw, kk * BST + sn * 32u, BST, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(b1, Bw, kk * BST + sn * 32u + 16u, BST, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(l0, Bwl, kk * BST + sn * 32u, BST, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(l1, Bwl, kk * BST + sn * 32u + 16u, BST, gl_CooperativeMatrixLayoutRowMajor);
            // (hi + lo) * (whi + wlo) without the negligible lo * wlo term.
            t0 = coopMatMulAdd(ah, b0, t0);
            t0 = coopMatMulAdd(al, b0, t0);
            t0 = coopMatMulAdd(ah, l0, t0);
            t1 = coopMatMulAdd(ah, b1, t1);
            t1 = coopMatMulAdd(al, b1, t1);
            t1 = coopMatMulAdd(ah, l1, t1);
        }
        float inv = exp2(-e);
        acc0 = acc0 + t0 * inv;
        acc1 = acc1 + t1 * inv;
        barrier();
    }

    coopMatStore(acc0, Cs, (sm * 16u) * CST + sn * 32u, CST, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(acc1, Cs, (sm * 16u) * CST + sn * 32u + 16u, CST, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (uint idx = tid; idx < TN * TM; idx += 256u) {
        uint m = idx / TM;
        uint k = idx - m * TM;
        uint mm = m0 + m;
        uint kk = k0 + k;
        if (mm < p.M && kk < p.K) {
            uint oi = kk + mm * p.K;
            if (p.acc == 1u) {
                o[oi] += Cs[m * CST + k];
            } else {
                o[oi] = Cs[m * CST + k];
            }
        }
    }
}
