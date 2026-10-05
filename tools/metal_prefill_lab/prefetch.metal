#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// Same tiling as tiled.metal (BK=32) but the global loads of step k+1 are issued before the mma of step k
// (register prefetch), and the dequantisation/conversion happens when they are stored to the tile.
#ifndef ACT
#define ACT float
#endif
#ifndef WGT
#define WGT float
#endif
#define NT (32 * SG_M * SG_N)
#define FM (BM / SG_M / 8)
#define FN (BN / SG_N / 8)
#define KF 4
#define TA ((BM * KF + NT - 1) / NT)
#define TB ((BN * 4 + NT - 1) / NT)

inline void scale_min(device const uchar* sc, uint s, thread float& scale, thread float& mn) {
    const int a = s < 4 ? int(sc[s]) & 63 : (int(sc[s + 4]) & 15) | ((int(sc[s - 4]) >> 6) << 4);
    const int b = s < 4 ? int(sc[s + 4]) & 63 : (int(sc[s + 4]) >> 4) | ((int(sc[s]) >> 6) << 4);
    scale = float(a); mn = float(b);
}

struct BReg { uint2 raw; float ds; float dmn; };

kernel void k_p(device const float* a [[buffer(0)]], device const uchar* b [[buffer(1)]], device float* c [[buffer(2)]],
                constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                uint3 bid3 [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]],
                uint sgid [[simdgroup_index_in_threadgroup]]) {
    const uint tid = tid3.x;
    const uint2 bid = bid3.xy;
    threadgroup ACT sa[BM * 32];
    threadgroup WGT sb[BN * 32];
    const uint sg_m = sgid / SG_N, sg_n = sgid % SG_N;
    simdgroup_float8x8 acc[FM * FN];
    for (uint i = 0; i < FM * FN; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row0 = bid.y * BM, col0 = bid.x * BN;
    const uint kblocks = k / 256;
    float4 ra[TA][2];
    BReg rb[TB];
#define LOAD_GLOBAL(KB) \
    { \
        _Pragma("unroll") for (uint t = 0; t < TA; ++t) { \
            const uint g = tid + t * NT; \
            const uint r = g / KF, kq = (g % KF) * 8; \
            device const float* src = a + (ulong)(row0 + r) * k + (KB) + kq; \
            ra[t][0] = *(device const float4*)src; ra[t][1] = *(device const float4*)(src + 4); \
        } \
        _Pragma("unroll") for (uint t = 0; t < TB; ++t) { \
            const uint g = tid + t * NT; \
            const uint ni = g / 4, lane8 = g % 4; \
            const ulong blk = ((ulong)(col0 + ni) * kblocks + (KB) / 256) * 144; \
            const uint s = ((KB) % 256) / 32, chunk = s / 2; \
            float sc, mn; scale_min(b + blk + 4, s, sc, mn); \
            const float d = float(as_type<half>(*(device const ushort*)(b + blk))); \
            const float dm = float(as_type<half>(*(device const ushort*)(b + blk + 2))); \
            rb[t].raw = *(device const uint2*)(b + blk + 16 + chunk * 32 + lane8 * 8); \
            rb[t].ds = sc * d; rb[t].dmn = mn * dm; \
        } \
    }
    LOAD_GLOBAL(0)
    for (uint kb = 0; kb < k; kb += 32) {
        {
            _Pragma("unroll") for (uint t = 0; t < TA; ++t) {
                const uint g = tid + t * NT;
                const uint r = g / KF, kq = (g % KF) * 8;
                threadgroup ACT* dst = sa + ((r / 8) * KF + kq / 8) * 64 + (r % 8) * 8;
                *(threadgroup vec<ACT, 4>*)dst = vec<ACT, 4>(ra[t][0]);
                *(threadgroup vec<ACT, 4>*)(dst + 4) = vec<ACT, 4>(ra[t][1]);
            }
            const uint hf = ((kb % 256) / 32) & 1;
            const uint sh = hf * 4;
            _Pragma("unroll") for (uint t = 0; t < TB; ++t) {
                const uint g = tid + t * NT;
                const uint ni = g / 4, lane8 = g % 4;
                threadgroup WGT* dst = sb + ((ni / 8) * KF + lane8) * 64 + (ni % 8) * 8;
                _Pragma("unroll") for (uint h = 0; h < 2; ++h) {
                    const uint w = h == 0 ? rb[t].raw.x : rb[t].raw.y;
                    vec<WGT, 4> v;
                    _Pragma("unroll") for (uint i = 0; i < 4; ++i) v[i] = WGT(float((w >> (i * 8 + sh)) & 15) * rb[t].ds - rb[t].dmn);
                    *(threadgroup vec<WGT, 4>*)(dst + h * 4) = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + 32 < k) LOAD_GLOBAL(kb + 32)
        for (uint ks = 0; ks < KF; ++ks) {
            simdgroup_matrix<ACT, 8, 8> fa[FM];
            simdgroup_matrix<WGT, 8, 8> fb[FN];
            for (uint i = 0; i < FM; ++i) simdgroup_load(fa[i], sa + ((sg_m * FM + i) * KF + ks) * 64, 8);
            for (uint j = 0; j < FN; ++j) simdgroup_load(fb[j], sb + ((sg_n * FN + j) * KF + ks) * 64, 8, ulong2(0, 0), true);
            for (uint i = 0; i < FM; ++i)
                for (uint j = 0; j < FN; ++j)
                    simdgroup_multiply_accumulate(acc[i * FN + j], fa[i], fb[j], acc[i * FN + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j)
            simdgroup_store(acc[i * FN + j], c, n, ulong2(col0 + (sg_n * FN + j) * 8, row0 + (sg_m * FM + i) * 8), false);
}
