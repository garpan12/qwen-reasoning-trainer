// Pre-split tensor-core GEMM: C[m, j] = tinv[m / 128] * sum_r A[m, r] * B[r, j]
// (stored c[m*J + j]), with A and B given as packed fp16 hi/lo parts
// (act_split_h2, dequant_e_h2_<t>). Products use hi*hi + lo*hi + hi*lo (the
// lo*lo term is negligible), accumulated in fp32.
//   BT defined: B is stored [J][R] (forward, the weight W[n][k] used as W^T).
//   otherwise:  B is stored [R][J] (backward dX = dY * W).
// 128x128 output tile, 256 invocations = 8 subgroups of 32 (2 along m x 4
// along j), each owning a 64x32 block of 4x2 16x16 accumulators. The
// accumulators are named variables, not an array: an array of coopmats made the
// NVIDIA driver spill them (3.4x slower). Requires
// M % 128 == 0, J % 128 == 0, R % 32 == 0 (checked by the caller).

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly  buffer A { uvec4 a[]; };
layout(binding = 1) readonly  buffer Bm { uvec4 w[]; };
layout(binding = 2) readonly  buffer TI { float tinv[]; };
layout(binding = 3) writeonly buffer C { float c[]; };

layout(push_constant) uniform P {
    uint M;
    uint J;
    uint R;
    uint pad;
} p;

#ifndef TERMS
#define TERMS 3  // 3: hi*hi + lo*hi + hi*lo; 2: no weight lo (opt-in)
#endif
#if TERMS >= 2
#define LOAD_AL(al, off) coopMatLoad(al, sAl, off, AH, gl_CooperativeMatrixLayoutRowMajor);
#define MMA_AL(C, al, b) C = coopMatMulAdd(al, b, C);
#else
#define LOAD_AL(al, off)
#define MMA_AL(C, al, b)
#endif
#if TERMS >= 3
#define MMA_BL(C, ah, bl) C = coopMatMulAdd(ah, bl, C);
#else
#define MMA_BL(C, ah, bl)
#endif
#ifndef WTM
#define WTM 4    // 16-row fragments per subgroup along m
#endif
#define TMM (32u * WTM)
#define TJ 128u
#define TK 32u
#define AS 5u    // uvec4 per shared row of 32 halves (+8 pad)
#define AH 40u   // the same row stride in halves
// Store the 8 halves of a uvec4 into a float16_t shared array at half offset o.
#define ST8(arr, o, v) { vec2 q0 = unpackHalf2x16(v.x); vec2 q1 = unpackHalf2x16(v.y); vec2 q2 = unpackHalf2x16(v.z); vec2 q3 = unpackHalf2x16(v.w); \
    arr[(o) + 0u] = float16_t(q0.x); arr[(o) + 1u] = float16_t(q0.y); arr[(o) + 2u] = float16_t(q1.x); arr[(o) + 3u] = float16_t(q1.y); \
    arr[(o) + 4u] = float16_t(q2.x); arr[(o) + 5u] = float16_t(q2.y); arr[(o) + 6u] = float16_t(q3.x); arr[(o) + 7u] = float16_t(q3.y); }

shared float16_t sAh[TMM * AH];
shared float16_t sAl[TMM * AH];
#ifdef BT
shared float16_t sBh[TJ * AH];
shared float16_t sBl[TJ * AH];
#else
#define BH 136u  // halves per shared row of 128 halves (+8 pad)
shared float16_t sBh[TK * BH];
shared float16_t sBl[TK * BH];
#endif

#define ACC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define MATA coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MATB coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint m0 = gl_WorkGroupID.x * TMM;   // m fastest: neighbours share a weight strip in L2
    uint wmr = 16u * WTM;               // rows per subgroup
    uint j0 = gl_WorkGroupID.y * TJ;
    uint sg = gl_SubgroupID;
    uint wm = sg & 1u;
    uint wj = sg >> 1u;
    uint R8 = p.R >> 3u;
    uint loA = p.M * R8;
    uint loW = p.J * R8;

    ACC c00 = ACC(0.0);
    ACC c01 = ACC(0.0);
    ACC c10 = ACC(0.0);
    ACC c11 = ACC(0.0);
#if WTM == 4
    ACC c20 = ACC(0.0);
    ACC c21 = ACC(0.0);
    ACC c30 = ACC(0.0);
    ACC c31 = ACC(0.0);
#endif

    for (uint k0 = 0u; k0 < p.R; k0 += TK) {
        uint k8 = k0 >> 3u;
        for (uint idx = tid; idx < TMM * 4u; idx += 256u) {
            uint row = idx >> 2u;
            uint cc = idx & 3u;
            uint g = (m0 + row) * R8 + k8 + cc;
            uvec4 vh = a[g];
#if TERMS >= 2
            uvec4 vl = a[loA + g];
            ST8(sAh, row * AH + cc * 8u, vh);
            ST8(sAl, row * AH + cc * 8u, vl);
#endif
        }
        for (uint i = 0u; i < 2u; i++) {
            uint idx = tid + i * 256u;
#ifdef BT
            uint row = idx >> 2u;
            uint cc = idx & 3u;
            uint gw = (j0 + row) * R8 + k8 + cc;
            uvec4 uh = w[gw];
#if TERMS >= 3
            uvec4 ul = w[loW + gw];
#endif
            ST8(sBh, row * AH + cc * 8u, uh);
#if TERMS >= 3
            ST8(sBl, row * AH + cc * 8u, ul);
#endif
#else
            uint rr = idx >> 4u;
            uint c2 = idx & 15u;
            uint gw = (k0 + rr) * (p.J >> 3u) + (j0 >> 3u) + c2;
            uvec4 uh = w[gw];
#if TERMS >= 3
            uvec4 ul = w[loW + gw];
#endif
            ST8(sBh, rr * BH + c2 * 8u, uh);
#if TERMS >= 3
            ST8(sBl, rr * BH + c2 * 8u, ul);
#endif
#endif
        }
        barrier();
        for (uint kk = 0u; kk < TK; kk += 16u) {
            MATB bh0;
            MATB bh1;
            MATB bl0;
            MATB bl1;
#ifdef BT
            coopMatLoad(bh0, sBh, (wj * 32u) * AH + kk, AH, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(bh1, sBh, (wj * 32u + 16u) * AH + kk, AH, gl_CooperativeMatrixLayoutColumnMajor);
#if TERMS >= 3
            coopMatLoad(bl0, sBl, (wj * 32u) * AH + kk, AH, gl_CooperativeMatrixLayoutColumnMajor);
            coopMatLoad(bl1, sBl, (wj * 32u + 16u) * AH + kk, AH, gl_CooperativeMatrixLayoutColumnMajor);
#endif
#else
            coopMatLoad(bh0, sBh, kk * BH + wj * 32u, BH, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(bh1, sBh, kk * BH + wj * 32u + 16u, BH, gl_CooperativeMatrixLayoutRowMajor);
#if TERMS >= 3
            coopMatLoad(bl0, sBl, kk * BH + wj * 32u, BH, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(bl1, sBl, kk * BH + wj * 32u + 16u, BH, gl_CooperativeMatrixLayoutRowMajor);
#endif
#endif
#define HROW(I, C0, C1) { \
                MATA ah; MATA al; \
                uint mr_ = wm * wmr + (I) * 16u; \
                coopMatLoad(ah, sAh, mr_ * AH + kk, AH, gl_CooperativeMatrixLayoutRowMajor); \
                LOAD_AL(al, mr_ * AH + kk) \
                C0 = coopMatMulAdd(ah, bh0, C0); MMA_AL(C0, al, bh0) MMA_BL(C0, ah, bl0) \
                C1 = coopMatMulAdd(ah, bh1, C1); MMA_AL(C1, al, bh1) MMA_BL(C1, ah, bl1) }
            HROW(0u, c00, c01);
            HROW(1u, c10, c11);
#if WTM == 4
            HROW(2u, c20, c21);
            HROW(3u, c30, c31);
#endif
        }
        barrier();
    }

    float inv = tinv[m0 >> 7u];
#define HSTORE(I, C0, C1) { \
        uint off_ = (m0 + wm * wmr + (I) * 16u) * p.J + j0 + wj * 32u; \
        ACC r0_ = C0 * inv; \
        ACC r1_ = C1 * inv; \
        coopMatStore(r0_, c, off_, p.J, gl_CooperativeMatrixLayoutRowMajor); \
        coopMatStore(r1_, c, off_ + 16u, p.J, gl_CooperativeMatrixLayoutRowMajor); }
    HSTORE(0u, c00, c01);
    HSTORE(1u, c10, c11);
#if WTM == 4
    HSTORE(2u, c20, c21);
    HSTORE(3u, c30, c31);
#endif
}
