#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// Parametrised Q4_K prefill kernel: BM x BN output tile, SG_M x SG_N simdgroups, k step BK (32 or 64),
// fragment-blocked threadgroup tiles (each 8x8 contiguous), weights dequantised 8 (BK=32) or 16 (BK=64)
// values per task with one scale decode per sub-block.  Shapes must be multiples of the tile (no edge code).
// STUB: 0 full; 2 weight tile not filled; 3 no tile fills; 4 mma only
#ifndef ACT
#define ACT float
#endif
#ifndef WGT
#define WGT float
#endif
#ifndef STUB
#define STUB 0
#endif
#define NT (32 * SG_M * SG_N)
#define FM (BM / SG_M / 8)
#define FN (BN / SG_N / 8)
#define KF (BK / 8)

inline void scale_min(device const uchar* sc, uint s, thread float& scale, thread float& mn) {
    const int a = s < 4 ? int(sc[s]) & 63 : (int(sc[s + 4]) & 15) | ((int(sc[s - 4]) >> 6) << 4);
    const int b = s < 4 ? int(sc[s + 4]) & 63 : (int(sc[s + 4]) >> 4) | ((int(sc[s]) >> 6) << 4);
    scale = float(a); mn = float(b);
}

kernel void k_t(device const float* a [[buffer(0)]], device const uchar* b [[buffer(1)]], device float* c [[buffer(2)]],
                constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                uint3 bid3 [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]],
                uint sgid [[simdgroup_index_in_threadgroup]]) {
    const uint tid = tid3.x;
#ifdef SWIZZLE
    const uint gx = n / BN, gy = m / BM, lin = bid3.y * gx + bid3.x;
    const uint2 bid = uint2(lin / gy, lin % gy);
#else
    const uint2 bid = bid3.xy;
#endif
    threadgroup ACT sa[BM * BK];
    threadgroup WGT sb[BN * BK];
    const uint sg_m = sgid / SG_N, sg_n = sgid % SG_N;
    simdgroup_float8x8 acc[FM * FN];
    for (uint i = 0; i < FM * FN; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint row0 = bid.y * BM, col0 = bid.x * BN;
    const uint kblocks = k / 256;
    for (uint i = tid; i < BM * BK; i += NT) sa[i] = ACT(1.0f);
    for (uint i = tid; i < BN * BK; i += NT) sb[i] = WGT(1.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<ACT, 8, 8> fa[FM];
    simdgroup_matrix<WGT, 8, 8> fb[FN];
    for (uint i = 0; i < FM; ++i) simdgroup_load(fa[i], sa + i * 64, 8);
    for (uint j = 0; j < FN; ++j) simdgroup_load(fb[j], sb + j * 64, 8, ulong2(0, 0), true);
    for (uint kb = 0; kb < k; kb += BK) {
#if STUB < 3
        for (uint g = tid; g < BM * KF; g += NT) {
            const uint r = g / KF, kq = (g % KF) * 8;
            device const float* src = a + (ulong)(row0 + r) * k + kb + kq;
            threadgroup ACT* dst = sa + ((r / 8) * KF + kq / 8) * 64 + (r % 8) * 8;
            *(threadgroup vec<ACT, 4>*)dst = vec<ACT, 4>(*(device const float4*)src);
            *(threadgroup vec<ACT, 4>*)(dst + 4) = vec<ACT, 4>(*(device const float4*)(src + 4));
        }
#endif
#if STUB < 2
#if BK == 32
        for (uint g = tid; g < BN * 4; g += NT) {
            const uint ni = g / 4, lane8 = g % 4;
            const ulong blk = ((ulong)(col0 + ni) * kblocks + kb / 256) * 144;
            const uint s = (kb % 256) / 32, chunk = s / 2, hf = s & 1;
            float sc, mn;
            scale_min(b + blk + 4, s, sc, mn);
            const float d = float(as_type<half>(*(device const ushort*)(b + blk)));
            const float dm = float(as_type<half>(*(device const ushort*)(b + blk + 2)));
            const uint2 raw = *(device const uint2*)(b + blk + 16 + chunk * 32 + lane8 * 8);
            const float ds = sc * d, dmn = mn * dm;
            threadgroup WGT* dst = sb + ((ni / 8) * KF + lane8) * 64 + (ni % 8) * 8;
            const uint sh = hf * 4;
            #pragma unroll
            for (uint h = 0; h < 2; ++h) {
                const uint w = h == 0 ? raw.x : raw.y;
                vec<WGT, 4> v;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) v[i] = WGT(float((w >> (i * 8 + sh)) & 15) * ds - dmn);
                *(threadgroup vec<WGT, 4>*)(dst + h * 4) = v;
            }
        }
#else
        for (uint g = tid; g < BN * 4; g += NT) {
            const uint ni = g / 4, lane8 = g % 4;
            const ulong blk = ((ulong)(col0 + ni) * kblocks + kb / 256) * 144;
            const uint chunk = (kb % 256) / 64;
            float sc0, mn0, sc1, mn1;
            scale_min(b + blk + 4, chunk * 2, sc0, mn0);
            scale_min(b + blk + 4, chunk * 2 + 1, sc1, mn1);
            const float d = float(as_type<half>(*(device const ushort*)(b + blk)));
            const float dm = float(as_type<half>(*(device const ushort*)(b + blk + 2)));
            const uint2 raw = *(device const uint2*)(b + blk + 16 + chunk * 32 + lane8 * 8);
            const float ds0 = sc0 * d, dmn0 = mn0 * dm, ds1 = sc1 * d, dmn1 = mn1 * dm;
            threadgroup WGT* dst0 = sb + ((ni / 8) * KF + lane8) * 64 + (ni % 8) * 8;
            threadgroup WGT* dst1 = sb + ((ni / 8) * KF + 4 + lane8) * 64 + (ni % 8) * 8;
            #pragma unroll
            for (uint h = 0; h < 2; ++h) {
                const uint w = h == 0 ? raw.x : raw.y;
                vec<WGT, 4> lo, hi;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    lo[i] = WGT(float((w >> (i * 8)) & 15) * ds0 - dmn0);
                    hi[i] = WGT(float((w >> (i * 8 + 4)) & 15) * ds1 - dmn1);
                }
                *(threadgroup vec<WGT, 4>*)(dst0 + h * 4) = lo;
                *(threadgroup vec<WGT, 4>*)(dst1 + h * 4) = hi;
            }
        }
#endif
#endif
#if STUB < 4
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        for (uint ks = 0; ks < KF; ++ks) {
#if STUB < 4
            for (uint i = 0; i < FM; ++i) simdgroup_load(fa[i], sa + ((sg_m * FM + i) * KF + ks) * 64, 8);
            for (uint j = 0; j < FN; ++j) simdgroup_load(fb[j], sb + ((sg_n * FN + j) * KF + ks) * 64, 8, ulong2(0, 0), true);
#endif
            for (uint i = 0; i < FM; ++i)
                for (uint j = 0; j < FN; ++j)
                    simdgroup_multiply_accumulate(acc[i * FN + j], fa[i], fb[j], acc[i * FN + j]);
        }
#if STUB < 4
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }
    for (uint i = 0; i < FM; ++i)
        for (uint j = 0; j < FN; ++j)
            simdgroup_store(acc[i * FN + j], c, n, ulong2(col0 + (sg_n * FN + j) * 8, row0 + (sg_m * FM + i) * 8), false);
}
