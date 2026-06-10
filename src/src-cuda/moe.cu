#include "moe.cuh"
#include <cfloat>

// Inspired by llama.cpp
namespace PackedStorage{
    __device__ __forceinline__ uint32_t pack(uint32_t token, uint32_t k){
        return (tok & 0x3FFFFFu) | (k << 22);
    }

    __device__ __forceinline__ uint32_t unpack_token(uint32_t data){
        return data & 0x3FFFFFu;
    }

    __device__ __forceinline__ uint32_t unpack_k(uint32_t data){
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
// smem: T * sizeof(GatherStore) bytes — compact (tok, k) list for this expert
// ---------------------------------------------------------------------------
template<int K_TEMPLATE>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void gather_ids_kernel(
    const uint32_t* __restrict__ expert_ids,     // [T, K]
    uint32_t*       __restrict__ token_map,      // [T*K]
    uint32_t*       __restrict__ slot_map,       // [T*K]
    uint32_t*       __restrict__ expert_offsets, // [E+1]
    int T, int K_VAR) {
    const int expert = blockIdx.x;
    const int tid    = threadIdx.x;
    const int K      = K_TEMPLATE == 0 ? K_VAR : K_TEMPLATE;

    extern __shared__ char smem_raw[];
    uint32_t* store = reinterpret_cast<uint32_t*>(smem_raw);

    int nex_prev   = 0;  // (tok,k) pairs routed to experts < this one
    int it_compact = 0;  // (tok,k) pairs routed to this expert so far

    if constexpr (K_TEMPLATE == 0) {
        // Generic: one token per step; warp threads stripe over K slots.
        for (int it = 0; it < T; it++) {
            int iex_used = -1;
            for (int iex = tid; iex < K; iex += WARP_SIZE) {
                const int e = expert_ids[it * K + iex];
                nex_prev += (e < expert) ? 1 : 0;
                if (e == expert) iex_used = iex;
            }
            if (iex_used != -1)
                store[it_compact] = PackedStorage::pack(it, iex_used);

            // warp-wide OR: does any thread have a match for this token?
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

            const int e = (iex < K_TEMPLATE && it < T) ?
                          expert_ids[it * K + iex] : INT_MAX;
            nex_prev += (e < expert) ? 1 : 0;
            const int iex_used = (e == expert) ? iex : -1;

            // Does this token group route to this expert? (same value for all
            // threads in the K_padded group after the group-level OR)
            const int it_compact_add_self = group_any<K_TEMPLATE>(iex_used != -1 ? 1 : 0);

            // Exclusive prefix sum over token groups within the warp.
            // Thread in group g accumulates sum of groups 0..g-1.
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = K_TEMPLATE; offset < WARP_SIZE; offset += K_TEMPLATE) {
                const int tmp = __shfl_up_sync(0xffffffff, it_compact_add_self, offset);
                if (tid >= offset) it_compact_add_lower += tmp;
            }

            if (iex_used != -1)
                store[it_compact + it_compact_add_lower] = PackedStorage::pack(it, iex_used);

            // Last warp lane holds the total for this step; broadcast to all.
            it_compact += __shfl_sync(0xffffffff,
                                    it_compact_add_lower + it_compact_add_self,
                                    WARP_SIZE - 1);
        }
    }

    // Reduce nex_prev (int) across the full warp
#pragma unroll
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        nex_prev += __shfl_xor_sync(0xffffffff, nex_prev, off);

    // Write expert_offsets; last expert also writes the final total
    if (tid == 0) {
        expert_offsets[expert] = nex_prev;
        if (expert == (int)gridDim.x - 1)
            expert_offsets[gridDim.x] = nex_prev + it_compact;
    }

    // Distribute slot_map / token_map writes across the warp
    for (int itc = tid; itc < it_compact; itc += WARP_SIZE) {
        const uint32_t s = store[itc];
        const uint32_t tok = PackedStorage::unpack_token(s);
        const uint32_t k = PackedStorage::unpack_k(s);
        const int slot = nex_prev + itc;
        slot_map[tok * K + k] = slot;
        token_map[slot] = tok;
    }
}

// Reorganize x into expert-sorted tensor...
// it seems this is how other inference engines do it but i am not too happy
// TODO: Does it make sense to avoid this and do matrix matmuls on x in a paged manner + slot offset manner?
// While slower than the GEMM implementation maybe the additional overhead will justify the switch
template<typename T>
static __global__ void gather_copy_kernel(
    const T* __restrict__ x,
    const int32_t* __restrict__ slot_map,
    T* __restrict__ gathered,
    int K, int dmodel){
    const int64_t tok_idx  = blockIdx.x / K;
    const int64_t exp_idx = blockIdx.x % K;

    // TODO: CAN THIS BE IN TEXTURE CACHE?
    const int slot = slot_map[tok*K + k];
    const T* src = x + tok*dmodel;

    T* dst = gathered + slot*dmodel;
    for (int d = threadIdx.x; d < dmodel; d += blockDim.x){
        // Does it make sense here to load a bigger byte group, e.g. 16 bytes?
        // since it is coalesced i think not but i might be wrong
        dst[d] = src[d];
    }
}

void moe_gather(const Tensor& x,
                const Tensor& expert_ids,
                Tensor& gathered,
                Tensor& token_map,
                Tensor& slot_map,
                Tensor& expert_offsets,
                int K, int num_experts){
    const int seql = x.shape[0];
    const int dmodel = x.shape[1];
    const int total = seql * K;

    
    const size_t smem = (size_t) seql * sizeof(uint32_t);
#define LAUNCH_IDS(K_VAL) \
    gather_ids_kernel<K_VAL><<<num_experts, WARP_SIZE, smem>>>( \
        (const uint32_t*)expert_ids.data(), (uint32_t*)token_map.data(), (uint32_t*) slot_map.data(), (uint32_t*)expert_offsets.data(), seql, K)
    switch (K) {
        case  2: LAUNCH_IDS(2); break;
        case  4: LAUNCH_IDS(4); break;
        case  8: LAUNCH_IDS(8); break;
        case 16: LAUNCH_IDS(16); break;
    }
#undef LAUNCH_IDS

    if (x.dtype() == CUDA_R_32F)
        gather_copy_kernel<float><<<total, 256>>>(
            (const float*)x.data(), (uint32_t*) slot_map.data(), (float*)gathered.data(), K, dmodel);
    else if (x.dtype() == CUDA_R_16BF)
        gather_copy_kernel<__nv_bfloat16><<<total, 256>>>(
            (const __nv_bfloat16*)x.data(), (uint32_t*) slot_map.data(), (__nv_bfloat16*) gathered.data(), K, dmodel);
}

template<typename T>
static __global__ void scatter_kernel(
    const T* __restrict__ expert_out,
    const float* __restrict__ expert_weights, 
    const uint32_t* __restrict__ slot_map,
    T* __restrict__ out,
    int K, int dmodel) {
    const int tok_idx = blockIdx.x;
    const float* pos_weight = expert_weights + tok_idx * K;
    const uint32_t* pos_slot = slot_map + tok_idx * K;

    for(int d = threadIdx.x; d < dmodel; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++){
            acc += pos_weight[k] * (float) expert_out[pos_slot[k] * dmodel + d];
        } 
        out[tok * D + d] = (T) acc;
    }
}

void moe_scatter(Tensor& out,
                const Tensor& expert_out,
                const Tensor& expert_weights,
                const Tensor& slot_map,
                int seql, int K){
    const int dmodel = expert_out.shape[1];
    const int64_t threads = std::min<int64_t>(((dmodel + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE,256);


    if(expert_out.dtype() == CUDA_R_32F){
        scatter_kernel<float><<<seql, threads>>>(
            (const float*)expert_out.data(), (float*) expert_weights.data(), (uint32_t*)slot_map.data(),
            (float*)out.data(), K, dmodel);
    }else if(expert_out.dtype() == CUDA_R_16BF){

        scatter_kernel<__nv_bfloat16><<<seql, threads>>>(
            (const __nv_bfloat16*)expert_out.data(), (float*)expert_weights.data(), (uint32_t*)slot_map.data(),
            (__nv_bfloat16*)out.data(), K, dmodel);
            
    }
}
