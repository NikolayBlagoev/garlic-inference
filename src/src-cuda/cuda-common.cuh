#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
// Bfloat 16 type:
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// float 8 if supported (lucky!)
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080
#include "tensor.h"

#define WARP_SIZE 32

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))



#define CUBLAS_CHECK(call) do { \
    cublasStatus_t  err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " — " << cublasGetStatusString(err) << "\n"; \
        std::exit(1); \
    } \
} while(0)

struct __align__(8) bf16x4 {
    __nv_bfloat16 x, y, z, w;
};

static cublasHandle_t _cublas_handle = nullptr;

// singleton:
static cublasHandle_t cublas_handle() {
    if (!_cublas_handle) cublasCreate(&_cublas_handle);
    return _cublas_handle;
}

static cudnnHandle_t _cudnn_handle = nullptr;
// singleton:
static cudnnHandle_t cudnn_handle() {
    if (!_cudnn_handle) cudnnCreate(&_cudnn_handle);
    return _cudnn_handle;
}

#ifdef __CUDACC__
template<int width = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = MAX(__shfl_xor_sync(0xffffffff, x, offset, width), x);
    }
    return x;
}
#endif


