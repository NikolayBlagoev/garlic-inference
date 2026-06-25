#include "moe.cuh"
#include <cfloat>

// Inspired by llama.cpp
namespace PackedStorage {
    __device__ __forceinline__ uint32_t pack(uint32_t token, uint32_t k) {
        return (token & 0x3FFFFFu) | (k << 22);
    }
    __device__ __forceinline__ uint32_t unpack_token(uint32_t data) {
        return data & 0x3FFFFFu;
    }
    __device__ __forceinline__ uint32_t unpack_k(uint32_t data) {
        return data >> 22;
    }
}

// Reduce-any (OR) within a power-of-2 group of threads inside the warp.
template<int group_size>
__device__ __forceinline__ int group_any(int x) {
#pragma unroll
    for (int off = group_size >> 1; off > 0; off >>= 1)
        x |= __shfl_xor_sync(0xffffffff, x, off, group_size);
    return x;
}

// ---------------------------------------------------------------------------
// Phase 1: index building — one block (one warp) per expert.
//
// K_TEMPLATE = 0 → generic:   one token per warp step, threads stripe over K
// K_TEMPLATE > 0 → optimized: WARP_SIZE/K_padded tokens per warp step,
//                              K_padded threads handle one token in parallel.
//
// smem: T * sizeof(uint32_t) bytes — compact (tok, k) list for this expert
// ---------------------------------------------------------------------------
template<int K_TEMPLATE>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void gather_ids_kernel(
    const uint32_t* __restrict__ expert_ids,
    uint32_t*       __restrict__ token_map,
    uint32_t*       __restrict__ slot_map,
    uint32_t*       __restrict__ expert_offsets,
    int T, int K_VAR) {

    const int expert = blockIdx.x;
    const int tid    = threadIdx.x;
    const int K      = K_TEMPLATE == 0 ? K_VAR : K_TEMPLATE;

    extern __shared__ char smem_raw[];
    uint32_t* store = reinterpret_cast<uint32_t*>(smem_raw);

    int nex_prev   = 0;
    int it_compact = 0;

    if constexpr (K_TEMPLATE == 0) {
        // Generic: one token per step; warp threads stripe over K slots.
        for (int it = 0; it < T; it++) {
            int iex_used = -1;
            for (int iex = tid; iex < K; iex += WARP_SIZE) {
                const int e = (int)__ldg(expert_ids + it * K + iex);
                nex_prev += (e < expert) ? 1 : 0;
                if (e == expert) iex_used = iex;
            }
            if (iex_used != -1)
                store[it_compact] = PackedStorage::pack(it, iex_used);

            int any = (iex_used != -1) ? 1 : 0;
            for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
                any |= __shfl_xor_sync(0xffffffff, any, off);
            if (any) it_compact++;
        }
    } else {
        constexpr int tokens_per_iter = WARP_SIZE / K_TEMPLATE;

        for (int it0 = 0; it0 < T; it0 += tokens_per_iter) {
            const int it  = it0 + tid / K_TEMPLATE;
            const int iex = tid % K_TEMPLATE;

            // __ldg: expert_ids is read-only across all expert warps
            const int e = (it < T) ? (int)__ldg(expert_ids + it * K + iex) : INT_MAX;
            nex_prev += (e < expert) ? 1 : 0;
            const int iex_used = (e == expert) ? iex : -1;

            const int it_compact_add_self = group_any<K_TEMPLATE>(iex_used != -1 ? 1 : 0);

            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = K_TEMPLATE; offset < WARP_SIZE; offset += K_TEMPLATE) {
                const int tmp = __shfl_up_sync(0xffffffff, it_compact_add_self, offset);
                if (tid >= offset) it_compact_add_lower += tmp;
            }

            if (iex_used != -1)
                store[it_compact + it_compact_add_lower] = PackedStorage::pack(it, iex_used);

            it_compact += __shfl_sync(0xffffffff,
                                      it_compact_add_lower + it_compact_add_self,
                                      WARP_SIZE - 1);
        }
    }

#pragma unroll
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        nex_prev += __shfl_xor_sync(0xffffffff, nex_prev, off);

    if (tid == 0) {
        expert_offsets[expert] = nex_prev;
        if (expert == (int)gridDim.x - 1)
            expert_offsets[gridDim.x] = nex_prev + it_compact;
    }
    __syncwarp();
    for (int itc = tid; itc < it_compact; itc += WARP_SIZE) {
        const uint32_t s   = store[itc];
        const uint32_t tok = PackedStorage::unpack_token(s);
        const uint32_t k   = PackedStorage::unpack_k(s);
        const int slot     = nex_prev + itc;
        slot_map[tok * K + k] = slot;
        token_map[slot] = tok;
    }
}

// ---------------------------------------------------------------------------
// Phase 2: copy tokens into expert-sorted layout.
// Uses 128-bit (uint4) vectorized loads when dmodel*sizeof(T) % 16 == 0.
// ---------------------------------------------------------------------------
template<typename T>
static __global__ void gather_copy_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ slot_map,
    T* __restrict__ gathered,
    int K, int dmodel) {

    const int tok_idx = blockIdx.x / K;
    const int exp_idx = blockIdx.x % K;
    const int slot    = (int)__ldg(slot_map + tok_idx * K + exp_idx);

    const T* __restrict__ src = x       + (int64_t)tok_idx * dmodel;
    T*       __restrict__ dst = gathered + (int64_t)slot    * dmodel;

    const int byte_width = dmodel * (int)sizeof(T);
    if (byte_width % 16 == 0) {
        const int nwords = byte_width / 16;
        const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);
        uint4*       __restrict__ d4 = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < nwords; i += blockDim.x)
            d4[i] = __ldg(s4 + i);
    } else {
        for (int d = threadIdx.x; d < dmodel; d += blockDim.x)
            dst[d] = __ldg(src + d);
    }
}

// ---------------------------------------------------------------------------
// Vectorized weighted-sum accumulator (128-bit per vector, fp32 intermediate).
// Explicit constructors zero-initialize v[] — do NOT use NSDMIs (float v[] = {})
// for device-code local variables; NVCC may skip them.
// ---------------------------------------------------------------------------
template<typename T> struct VecAccum;

template<> struct VecAccum<float> {
    static constexpr int N = 4;
    float v[4];

    __device__ __forceinline__ VecAccum() {
#pragma unroll
        for (int i = 0; i < N; i++) v[i] = 0.f;
    }

    __device__ __forceinline__ void madd(const float* __restrict__ src, float w) {
        const float4 e = __ldg(reinterpret_cast<const float4*>(src));
        v[0] = fmaf(w, e.x, v[0]);
        v[1] = fmaf(w, e.y, v[1]);
        v[2] = fmaf(w, e.z, v[2]);
        v[3] = fmaf(w, e.w, v[3]);
    }

    __device__ __forceinline__ void store(float* dst) const {
        *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
    }
};

template<> struct VecAccum<__nv_bfloat16> {
    static constexpr int N = 8;
    float v[8];

    __device__ __forceinline__ VecAccum() {
#pragma unroll
        for (int i = 0; i < N; i++) v[i] = 0.f;
    }

    __device__ __forceinline__ void madd(const __nv_bfloat16* __restrict__ src, float w) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
        const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
        float2 f;
        f = __bfloat1622float2(b[0]); v[0]=fmaf(w,f.x,v[0]); v[1]=fmaf(w,f.y,v[1]);
        f = __bfloat1622float2(b[1]); v[2]=fmaf(w,f.x,v[2]); v[3]=fmaf(w,f.y,v[3]);
        f = __bfloat1622float2(b[2]); v[4]=fmaf(w,f.x,v[4]); v[5]=fmaf(w,f.y,v[5]);
        f = __bfloat1622float2(b[3]); v[6]=fmaf(w,f.x,v[6]); v[7]=fmaf(w,f.y,v[7]);
    }

    __device__ __forceinline__ void store(__nv_bfloat16* dst) const {
        uint4 raw;
        __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&raw);
        b[0] = __float22bfloat162_rn(make_float2(v[0], v[1]));
        b[1] = __float22bfloat162_rn(make_float2(v[2], v[3]));
        b[2] = __float22bfloat162_rn(make_float2(v[4], v[5]));
        b[3] = __float22bfloat162_rn(make_float2(v[6], v[7]));
        *reinterpret_cast<uint4*>(dst) = raw;
    }
};

template<> struct VecAccum<__half> {
    static constexpr int N = 8;
    float v[8];

    __device__ __forceinline__ VecAccum() {
#pragma unroll
        for (int i = 0; i < N; i++) v[i] = 0.f;
    }

    __device__ __forceinline__ void madd(const __half* __restrict__ src, float w) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
        const __half2* h = reinterpret_cast<const __half2*>(&raw);
        float2 f;
        f = __half22float2(h[0]); v[0]=fmaf(w,f.x,v[0]); v[1]=fmaf(w,f.y,v[1]);
        f = __half22float2(h[1]); v[2]=fmaf(w,f.x,v[2]); v[3]=fmaf(w,f.y,v[3]);
        f = __half22float2(h[2]); v[4]=fmaf(w,f.x,v[4]); v[5]=fmaf(w,f.y,v[5]);
        f = __half22float2(h[3]); v[6]=fmaf(w,f.x,v[6]); v[7]=fmaf(w,f.y,v[7]);
    }

    __device__ __forceinline__ void store(__half* dst) const {
        uint4 raw;
        __half2* h = reinterpret_cast<__half2*>(&raw);
        h[0] = __float22half2_rn(make_float2(v[0], v[1]));
        h[1] = __float22half2_rn(make_float2(v[2], v[3]));
        h[2] = __float22half2_rn(make_float2(v[4], v[5]));
        h[3] = __float22half2_rn(make_float2(v[6], v[7]));
        *reinterpret_cast<uint4*>(dst) = raw;
    }
};

// ---------------------------------------------------------------------------
// Phase 3: weighted scatter-add back to token space.
// K is templated so the inner loop is fully unrolled by the compiler.
// Weights and slot indices are preloaded into registers.
// Uses vectorized (128-bit) loads + fp32 accumulation; scalar fallback for
// non-16-byte-aligned dmodel.
// ---------------------------------------------------------------------------
template<typename T, int K>
static __global__ void scatter_kernel(
    const T*        __restrict__ expert_out,
    const float*    __restrict__ expert_weights,
    const uint32_t* __restrict__ slot_map,
    T*              __restrict__ out,
    int dmodel) {

    const int tok_idx = blockIdx.x;

    float w[K];
    int slots[K];
#pragma unroll
    for (int k = 0; k < K; k++) {
        w[k]     = __ldg(expert_weights + tok_idx * K + k);
        slots[k] = (int)__ldg(slot_map   + tok_idx * K + k);
    }

    using VA = VecAccum<T>;
    constexpr int N = VA::N;

    if (dmodel * (int)sizeof(T) % 16 == 0) {
        const int nvec = dmodel / N;
        for (int iv = threadIdx.x; iv < nvec; iv += blockDim.x) {
            VA va;
#pragma unroll
            for (int k = 0; k < K; k++)
                va.madd(expert_out + (int64_t)slots[k] * dmodel + (int64_t)iv * N, w[k]);
            va.store(out + (int64_t)tok_idx * dmodel + (int64_t)iv * N);
        }
    } else {
        for (int d = threadIdx.x; d < dmodel; d += blockDim.x) {
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < K; k++)
                acc = fmaf(w[k], (float)__ldg(expert_out + (int64_t)slots[k]*dmodel + d), acc);
            out[(int64_t)tok_idx * dmodel + d] = (T)acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

void moe_gather(const Tensor& x,
                const Tensor& expert_ids,
                Tensor& gathered,
                Tensor& token_map,
                Tensor& slot_map,
                Tensor& expert_offsets,
                int K, int num_experts) {

    const int seql   = x.shape[0];
    const int dmodel = x.shape[1];
    const int total  = seql * K;
    const size_t smem = (size_t)seql * sizeof(uint32_t);

#define LAUNCH_IDS(K_VAL) \
    gather_ids_kernel<K_VAL><<<num_experts, WARP_SIZE, smem>>>( \
        (const uint32_t*)expert_ids.data(), (uint32_t*)token_map.data(), \
        (uint32_t*)slot_map.data(), (uint32_t*)expert_offsets.data(), seql, K)

    switch (K) {
        case  2: LAUNCH_IDS( 2); break;
        case  4: LAUNCH_IDS( 4); break;
        case  8: LAUNCH_IDS( 8); break;
        case 16: LAUNCH_IDS(16); break;
        case 32: LAUNCH_IDS(32); break;
        default: LAUNCH_IDS( 0); break;
    }
#undef LAUNCH_IDS

    // Round up to nearest warp, cap at 256 threads.
    const int copy_threads = (int)std::min<int64_t>(
        ((int64_t)dmodel + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE, 256);

    if (x.dtype() == CUDA_R_32F)
        gather_copy_kernel<float><<<total, copy_threads>>>(
            (const float*)x.data(), (const uint32_t*)slot_map.data(),
            (float*)gathered.data(), K, dmodel);
    else if (x.dtype() == CUDA_R_16BF)
        gather_copy_kernel<__nv_bfloat16><<<total, copy_threads>>>(
            (const __nv_bfloat16*)x.data(), (const uint32_t*)slot_map.data(),
            (__nv_bfloat16*)gathered.data(), K, dmodel);
    else if (x.dtype() == CUDA_R_16F)
        gather_copy_kernel<__half><<<total, copy_threads>>>(
            (const __half*)x.data(), (const uint32_t*)slot_map.data(),
            (__half*)gathered.data(), K, dmodel);
}

void moe_scatter(Tensor& out,
                 const Tensor& expert_out,
                 const Tensor& expert_weights,
                 const Tensor& slot_map,
                 int seql, int K) {

    const int dmodel  = expert_out.shape[1];
    const int threads = (int)std::min<int64_t>(
        ((int64_t)dmodel + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE, 256);

#define LAUNCH_SCATTER(TYPE, K_VAL) \
    scatter_kernel<TYPE, K_VAL><<<seql, threads>>>( \
        (const TYPE*)expert_out.data(), (const float*)expert_weights.data(), \
        (const uint32_t*)slot_map.data(), (TYPE*)out.data(), dmodel)

#define DISPATCH_K(TYPE) \
    switch (K) { \
        case  2: LAUNCH_SCATTER(TYPE,  2); break; \
        case  4: LAUNCH_SCATTER(TYPE,  4); break; \
        case  8: LAUNCH_SCATTER(TYPE,  8); break; \
        case 16: LAUNCH_SCATTER(TYPE, 16); break; \
        case 32: LAUNCH_SCATTER(TYPE, 32); break; \
        case 64: LAUNCH_SCATTER(TYPE, 64); break; \
    }

    if (expert_out.dtype() == CUDA_R_32F)
        DISPATCH_K(float)
    else if (expert_out.dtype() == CUDA_R_16BF)
        DISPATCH_K(__nv_bfloat16)
    else if (expert_out.dtype() == CUDA_R_16F)
        DISPATCH_K(__half)

#undef LAUNCH_SCATTER
#undef DISPATCH_K
}
