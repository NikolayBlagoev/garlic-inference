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

template<typename T>
__device__ inline void write4(T* p, T a, T b, T c, T d) {
    p[0] = a; p[1] = b; p[2] = c; p[3] = d;
}
template<>
__device__ inline void write4<float>(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}
template<>
__device__ inline void write4<__nv_bfloat16>(__nv_bfloat16* p, __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d) {
    reinterpret_cast<__nv_bfloat162*>(p)[0] = __nv_bfloat162(a,b);
    reinterpret_cast<__nv_bfloat162*>(p)[1] = __nv_bfloat162(c,d);
}
template<>
__device__ inline void write4<__half>(__half* p, __half a, __half b, __half c, __half d) {
    reinterpret_cast<__half2*>(p)[0] = __half2(a, b);
    reinterpret_cast<__half2*>(p)[1] = __half2(c,d);
}
template<typename T>
__device__ inline void write4(T* p, T* a) {
    p[0] = a[0]; 
    p[1] = a[1]; 
    p[2] = a[2]; 
    p[3] = a[3];
}
template<>
__device__ inline void write4<__nv_bfloat16>(__nv_bfloat16* p, __nv_bfloat16* a) {
    reinterpret_cast<__nv_bfloat162*>(p)[0] = *reinterpret_cast<__nv_bfloat162*>(a);
    reinterpret_cast<__nv_bfloat162*>(p)[1] = *reinterpret_cast<__nv_bfloat162*>(a + 2);
}
template<>
__device__ inline void write4<__half>(__half* p, __half* a) {
    reinterpret_cast<__half2*>(p)[0] = *reinterpret_cast<__half2*>(a);
    reinterpret_cast<__half2*>(p)[1] = *reinterpret_cast<__half2*>(a + 2);
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
        x = fmaxf(__shfl_xor_sync(0xffffffff, x, offset, width), x);
    }
    return x;
}
#endif


