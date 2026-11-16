// Tensor-core (VK_KHR_cooperative_matrix) version of matmul_wq_body.glsl:
//   C[n,m] = sum_k dequant(W[n,k]) * B[m,k]      (stored c[m*N + n])
//
// Same bindings, push constants, 64x64 output tile and 256-invocation workgroup
// as the fp32 body, so dispatch is unchanged. Requires subgroup size 32 (8
// subgroups per workgroup) and 16x16x16 f16*f16+f32 cooperative matrices.
//
// Precision: the dequantized weight tile is split into hi + lo fp16 parts. The activation tile is scaled by a power of two so its
// max |x| maps to 2^14, then split x*s = hi + lo into two fp16 parts (~21-bit
// mantissa, no fp16 under/overflow); both parts are multiplied and the fp32
// product is unscaled. Accumulation is fp32.

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) readonly  buffer W { uint data[]; };
layout(binding = 1) readonly  buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };

layout(push_constant) uniform P {
    uint K;
    uint N;
    uint M;
    uint pad;
} p;

#include "quant_decode.glsl"

#define TM 64            // output n per workgroup
#define TN 64            // output m per workgroup
#define TK 32            // reduction step
#define AST (TK + 8)     // f16 row stride of the activation tile [m][k]
#define BST (TM + 8)     // f16 row stride of the weight tile [k][n]
#define CST (TM + 4)     // f32 row stride of the output staging tile [m][n]

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
    uint n0 = gl_WorkGroupID.x * TM;
    uint m0 = gl_WorkGroupID.y * TN;
    uint sg = gl_SubgroupID;      // 0..7
    uint sm = sg & 3u;            // 16-row band of m
    uint sn = sg >> 2u;           // 32-column band of n

    ACC acc0 = ACC(0.0);
    ACC acc1 = ACC(0.0);

    for (uint k0 = 0u; k0 < p.K; k0 += TK) {
        // Activation tile A[m][k] = B[m0+m, k0+k]; consecutive lanes walk k (contiguous).
        float av[8];
        float lmax = 0.0;
        for (uint i = 0u; i < 8u; i++) {
            uint idx = tid + i * 256u;
            uint m = idx / TK;
            uint k = idx - m * TK;
            uint mm = m0 + m;
            uint kk = k0 + k;
            float v = (mm < p.M && kk < p.K) ? b[mm * p.K + kk] : 0.0;
            av[i] = v;
            lmax = max(lmax, abs(v));
        }
        lmax = subgroupMax(lmax);
        if (subgroupElect()) {
            smax[sg] = lmax;
        }
        // Weight tile Bw[k][n] = dequant(W[n0+n, k0+k]); decode walks k like the fp32 body.
        for (uint i = 0u; i < 8u; i++) {
            uint idx = tid + i * 256u;
            uint n = idx / TK;
            uint k = idx - n * TK;
            uint nn = n0 + n;
            uint kk = k0 + k;
            float wv = (nn < p.N && kk < p.K) ? QDECODE(nn, kk) : 0.0;
            float wh = uintBitsToFloat(floatBitsToUint(wv) & 0xFFFFE000u);
            Bw[k * BST + n] = float16_t(wh);
            Bwl[k * BST + n] = float16_t(wv - wh);
        }
        barrier();

        float tmax = max(max(max(smax[0], smax[1]), max(smax[2], smax[3])),
                         max(max(smax[4], smax[5]), max(smax[6], smax[7])));
        float e = tmax > 0.0 ? clamp(14.0 - ceil(log2(tmax)), -100.0, 100.0) : 0.0;
        float s = exp2(e);
        for (uint i = 0u; i < 8u; i++) {
            uint idx = tid + i * 256u;
            uint m = idx / TK;
            uint k = idx - m * TK;
            // Split by truncating the fp32 mantissa to fp16's 10 bits with integer ops.
            // Float round-trips (float16_t(x), packHalf2x16) get folded by the NVIDIA
            // compiler, which silently zeroes the lo part. The scaled tile max is 2^14,
            // so hf is a normal fp16 value (or a tiny value whose loss is negligible).
            float x = av[i] * s;
            float hf = uintBitsToFloat(floatBitsToUint(x) & 0xFFFFE000u);
            Ahi[m * AST + k] = float16_t(hf);
            Alo[m * AST + k] = float16_t(x - hf);
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
        uint n = idx - m * TM;
        uint mm = m0 + m;
        uint nn = n0 + n;
        if (mm < p.M && nn < p.N) {
            c[mm * p.N + nn] = Cs[m * CST + n];
        }
    }
}
