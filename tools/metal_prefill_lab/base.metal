#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// Transcription of boring's native Q4_K prefill schedule (64x32 tile, 256 threads, k step 32).
// STUB: 0 full; 1 dequant arithmetic replaced by a bare nibble; 2 weight tile not filled;
//       3 neither tile filled (barriers + threadgroup loads + mma); 4 mma only (no loads, no barriers)
#ifndef ACT
#define ACT float
#endif
#ifndef WGT
#define WGT float
#endif
#ifndef STUB
#define STUB 0
#endif
kernel void k_base(device const float* a [[buffer(0)]], device const uchar* b [[buffer(1)]], device float* c [[buffer(2)]],
                   constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                   uint3 bid3 [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]]) {
    const uint2 bid = bid3.xy; const uint tid = tid3.x;
    threadgroup ACT a_tile[2048];
    threadgroup WGT b_tile[1024];
    threadgroup float out_tile[2048];
    const uint warp = tid / 32;
    const uint row = bid.y * 64, col = bid.x * 32;
    simdgroup_float8x8 r0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 r1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 r2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 r3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<ACT, 8, 8> fa;
    simdgroup_matrix<WGT, 8, 8> fb0, fb1, fb2, fb3;
    for (uint i = tid; i < 2048; i += 256) a_tile[i] = ACT(1.0f);
    for (uint i = tid; i < 1024; i += 256) b_tile[i] = WGT(1.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0; base < k; base += 32) {
#if STUB < 3
        {
            const uint in_row = row + tid / 4;
            const uint in_inner = base + (tid % 4) * 8;
            threadgroup ACT* ap = a_tile + (tid / 4) * 32 + (tid % 4) * 8;
            if (in_row < m) {
                device const float* as = a + (ulong)in_row * k + in_inner;
                *(threadgroup vec<ACT, 4>*)(ap) = vec<ACT, 4>(*(device const float4*)(as));
                *(threadgroup vec<ACT, 4>*)(ap + 4) = vec<ACT, 4>(*(device const float4*)(as + 4));
            } else {
                *(threadgroup vec<ACT, 4>*)(ap) = vec<ACT, 4>(0.0f);
                *(threadgroup vec<ACT, 4>*)(ap + 4) = vec<ACT, 4>(0.0f);
            }
        }
#endif
#if STUB < 2
        {
            const uint dc = tid % 32, dr0 = (tid / 32) * 4;
            const uint ocol = col + dc;
            if (ocol < n) {
                const ulong flat0 = (ulong)ocol * k + base + dr0;
                const ulong block = (flat0 / 256) * 144;
                const uint pos = uint(flat0 % 256), chunk = pos / 64, within = pos % 64;
                const uint half_ = within / 32, lane = within % 32, subblock = chunk * 2 + half_;
                const ulong scales = block + 4;
                const int subscale = subblock < 4 ? int(b[scales + subblock]) & 63 : (int(b[scales + subblock + 4]) & 15) | ((int(b[scales + subblock - 4]) >> 6) << 4);
                const int submin = subblock < 4 ? int(b[scales + subblock + 4]) & 63 : (int(b[scales + subblock + 4]) >> 4) | ((int(b[scales + subblock]) >> 6) << 4);
                const uchar4 packed4 = as_type<uchar4>(*(device const uint*)(b + block + 16 + chunk * 32 + lane));
                const half2 dd = as_type<half2>(*(device const uint*)(b + block));
                for (uint j = 0; j < 4; ++j) {
                    const int qv = half_ == 0 ? int(packed4[j] & 15) : int((packed4[j] >> 4) & 15);
#if STUB == 1
                    b_tile[(dr0 + j) * 32 + dc] = WGT(float(qv));
#else
                    b_tile[(dr0 + j) * 32 + dc] = WGT(float(qv * subscale) * float(dd.x) - float(submin) * float(dd.y));
#endif
                }
            } else {
                for (uint j = 0; j < 4; ++j) b_tile[(dr0 + j) * 32 + dc] = WGT(0.0f);
            }
        }
#endif
#if STUB < 4
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        for (uint kk = 0; kk < 32; kk += 8) {
#if STUB < 4
            simdgroup_load(fa, a_tile, 32, ulong2(kk, warp * 8), false);
            simdgroup_load(fb0, b_tile, 32, ulong2(0, kk), false);
            simdgroup_load(fb1, b_tile, 32, ulong2(8, kk), false);
            simdgroup_load(fb2, b_tile, 32, ulong2(16, kk), false);
            simdgroup_load(fb3, b_tile, 32, ulong2(24, kk), false);
#endif
            simdgroup_multiply_accumulate(r0, fa, fb0, r0);
            simdgroup_multiply_accumulate(r1, fa, fb1, r1);
            simdgroup_multiply_accumulate(r2, fa, fb2, r2);
            simdgroup_multiply_accumulate(r3, fa, fb3, r3);
        }
#if STUB < 4
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    }
    simdgroup_store(r0, out_tile + (warp * 4 + 0) * 64, 8, ulong2(0, 0), false);
    simdgroup_store(r1, out_tile + (warp * 4 + 1) * 64, 8, ulong2(0, 0), false);
    simdgroup_store(r2, out_tile + (warp * 4 + 2) * 64, 8, ulong2(0, 0), false);
    simdgroup_store(r3, out_tile + (warp * 4 + 3) * 64, 8, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cell = tid; cell < 2048; cell += 256) {
        const uint frag = cell / 64, loc = cell % 64;
        const uint orow = row + (frag / 4) * 8 + loc / 8;
        const uint ocol = col + (frag % 4) * 8 + loc % 8;
        if (orow < m && ocol < n) c[(ulong)orow * n + ocol] = out_tile[cell];
    }
}
