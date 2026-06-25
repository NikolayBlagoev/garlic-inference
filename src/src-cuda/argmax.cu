#include "argmax.cuh"
#include <cfloat>

// Each block reduces one outer row along the last (innermost) dimension.
// Uses shared memory to hold per-thread (best_val, best_idx) pairs.
template<typename T>
static __global__ void argmax_last_dim_kernel(const T* __restrict__ x, 
                                            uint32_t* __restrict__ dst,
                                            const int64_t ncols) {
    const int64_t row = blockIdx.x;
    
    const int n_warps = blockDim.x >> 5; //2^5 = 32
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    float maxval = -FLT_MAX;
    uint32_t argmax = 0;
    const T* rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = (float) rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }


    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
        __syncthreads();
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

std::vector<uint32_t> argmax(const Tensor& x, Tensor& out, int dim) {
    dim = -1;
    const int ndim = x.ndim();
    if (dim < 0) dim += ndim;

    int nee = 1;
    for (int i = 0; i < dim; i++) nee *= x.shape[i];
    int row_len = x.shape[dim];


    const int64_t num_threads = std::min<int64_t>(1024, (row_len + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);

    if (x.dtype() == CUDA_R_16BF) {
        argmax_last_dim_kernel<__nv_bfloat16><<<nee, num_threads, 0, get_compute_stream()>>>(
            (const __nv_bfloat16*)x.data(), (uint32_t*)out.data(), row_len);
    } else if (x.dtype() == CUDA_R_32F) {
        argmax_last_dim_kernel<float><<<nee, num_threads, 0, get_compute_stream()>>>(
            (const float*)x.data(), (uint32_t*)out.data(), row_len);
    } else if (x.dtype() == CUDA_R_16F) {
        argmax_last_dim_kernel<__half><<<nee, num_threads, 0, get_compute_stream()>>>(
            (const __half*)x.data(), (uint32_t*)out.data(), row_len);
    }

    std::vector<uint32_t> result(nee);
    cudaMemcpyAsync(result.data(), out.data(), (size_t)nee * sizeof(uint32_t), cudaMemcpyDeviceToHost, get_compute_stream());
    cudaStreamSynchronize(get_compute_stream());
    

    return result;
}

// something interesting - > since most moe routers enforce weights to sum to 1, we don't need to perform
// softmax first:
// class Qwen3MoeTopKRouter(nn.Module):
//     def __init__(self, config):
//         super().__init__()
//         self.top_k = config.num_experts_per_tok
//         self.num_experts = config.num_experts
//         self.norm_topk_prob = config.norm_topk_prob
//         self.hidden_dim = config.hidden_size
//         self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

//     def forward(self, hidden_states):
//         hidden_states = hidden_states.reshape(-1, self.hidden_dim)
//         router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
//         router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
//         router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
//         if self.norm_topk_prob: -> when this is true removes the need for the previous softmax
//             router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
//         router_top_value = router_top_value.to(router_logits.dtype)
//         router_scores = router_top_value
//         return router_logits, router_scores, router_indices
// TODO: IMPLEMENT A PATH IF NOT self.norm_topk_prob
template<typename T>
__global__ void topk_softmax_last_dim_kernel(
    const T*    __restrict__ x,
    uint32_t*    __restrict__ dst_ids,
    float*      __restrict__ dst_weights,
    const int64_t ncols, const int K) {

    const int64_t row = blockIdx.x;
    const int n_warps = blockDim.x >> 5; //2^5 = 32
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const T* rowx = x + row * ncols;

    constexpr int    max_warps = 32;
    extern __shared__ char smem_raw[];
    uint32_t* winners = reinterpret_cast<uint32_t*>(smem_raw);
    uint32_t* shared_argmax = (winners + K);
    float* shared_maxval = reinterpret_cast<float*>(shared_argmax + max_warps);

    for(int k = 0; k < K; k++){
        float maxval = -FLT_MAX;
        uint32_t argmax = 0;
        for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            bool cont = false;
            for(int m = 0; m < k; m++){
                if(winners[m] == col){
                    cont = true;
                    break;
                }
            }
            if(cont) continue;

            const float val = (float) rowx[col];
            if (val > maxval) {
                maxval = val;
                argmax = col;
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
            const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
            if (val > maxval) {
                maxval = val;
                argmax = col;
            }
        }

        if (n_warps > 1) {
            if (lane_id == 0) {
                shared_maxval[warp_id] = maxval;
                shared_argmax[warp_id] = argmax;
            }

            __syncthreads();

            if (warp_id == 0) {
                if (lane_id < n_warps) {
                    maxval = shared_maxval[lane_id];
                    argmax = shared_argmax[lane_id];
                }
#pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                    const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                    const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                    if (val > maxval) {
                        maxval = val;
                        argmax = col;
                    }
                }
            }
        }

        if (warp_id == 0 && lane_id == 0) {
            winners[k] = argmax;
        }

        __syncthreads();
    }


    if (warp_id == 0 && lane_id == 0) {
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; k++) {
            float v = (float) rowx[winners[k]];
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += expf((float)rowx[winners[k]] - max_val);

        const int64_t base = (int64_t) row * K;
        for (int k = 0; k < K; k++) {
            dst_ids[base + k] = winners[k];
            dst_weights[base + k] = expf((float)rowx[winners[k]] - max_val) / sum;
        }
    }
}


void topk(const Tensor& x, Tensor& ids, Tensor& weights, int K, cudaStream_t compute_stream) {
    int nee = x.num_elements();
    int row_len = x.shape[x.ndim() - 1];
    nee = nee / row_len;
    if(compute_stream == nullptr) compute_stream = get_compute_stream();

    const int64_t num_threads = std::min<int64_t>(1024, (row_len + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    // Kernel layout: winners[K] | shared_argmax[32] | shared_maxval[32]
    // Must use 32 here to match the kernel's `constexpr int max_warps = 32` layout.
    size_t smem = (size_t) K * sizeof(uint32_t)
            + 32 * sizeof(uint32_t)
            + 32 * sizeof(float);
    
    if (x.dtype() == CUDA_R_16BF) {
        topk_softmax_last_dim_kernel<__nv_bfloat16><<<nee, num_threads, smem, compute_stream>>>(
            (const __nv_bfloat16*)x.data(), (uint32_t*) ids.data(), (float*) weights.data(),
            row_len, K);
    } else if (x.dtype() == CUDA_R_32F) {
        topk_softmax_last_dim_kernel<float><<<nee, num_threads, smem, compute_stream>>>(
            (const float*)x.data(), (uint32_t*) ids.data(), (float*) weights.data(),
            row_len, K);
    } else if (x.dtype() == CUDA_R_16F) {
        topk_softmax_last_dim_kernel<__half><<<nee, num_threads, smem, compute_stream>>>(
            (const __half*)x.data(), (uint32_t*) ids.data(), (float*) weights.data(),
            row_len, K);
    }
}