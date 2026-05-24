#include "argmax.cuh"
#include <cfloat>

// Each block reduces one outer row along the last (innermost) dimension.
// Uses shared memory to hold per-thread (best_val, best_idx) pairs.
template<typename T>
static __global__ void argmax_last_dim_kernel(const T* __restrict__ x, uint32_t* __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;
    
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

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
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
        argmax_last_dim_kernel<__nv_bfloat16><<<nee, num_threads>>>(
            (const __nv_bfloat16*)x.data(), (uint32_t*)out.data(), row_len);
    } else if (x.dtype() == CUDA_R_32F) {
        argmax_last_dim_kernel<float><<<nee, num_threads>>>(
            (const float*)x.data(), (uint32_t*)out.data(), row_len);
    }


    std::vector<uint32_t> result(nee);
    cudaMemcpy(result.data(), out.data(), (size_t)nee * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    

    return result;
}
