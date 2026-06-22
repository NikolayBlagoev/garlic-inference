#include "activations.cuh"

__device__ __forceinline__ float silu(float x) {
    return x / (1 + expf(-x));
}

template<typename T>
__device__ inline void silu4(T* p) {
    p[0] = T(silu(float(p[0])));
    p[1] = T(silu(float(p[1])));
    p[2] = T(silu(float(p[2])));
    p[3] = T(silu(float(p[3])));
}

template<>
__device__ inline void silu4<float>(float* p) {
    float4 v = *reinterpret_cast<float4*>(p);
    *reinterpret_cast<float4*>(p) = make_float4(silu(v.x), silu(v.y), silu(v.z), silu(v.w));
}

template<>
__device__ inline void silu4<__nv_bfloat16>(__nv_bfloat16* p) {
    
    __nv_bfloat162 v = reinterpret_cast<__nv_bfloat162*>(p)[0];
    reinterpret_cast<__nv_bfloat162*>(p)[0] = __nv_bfloat162(__nv_bfloat16(silu(float(v.x))),__nv_bfloat16(silu(float(v.y))));
    v = reinterpret_cast<__nv_bfloat162*>(p)[1];
    reinterpret_cast<__nv_bfloat162*>(p)[1] = __nv_bfloat162(__nv_bfloat16(silu(float(v.x))),__nv_bfloat16(silu(float(v.y))));
}

template<>
__device__ inline void silu4<__half>(__half* p) {
    __half2 v = reinterpret_cast<__half2*>(p)[0];
    reinterpret_cast<__half2*>(p)[0] = __half2(__half(silu(float(v.x))),__half(silu(float(v.y))));
    v = reinterpret_cast<__half2*>(p)[1];
    reinterpret_cast<__half2*>(p)[1] = __half2(__half(silu(float(v.x))),__half(silu(float(v.y))));

}

template<typename T>
__global__ void silu_kernel(
    T* __restrict__ x,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        silu4<T>(x+i);
    } else {
        for (; i < N; i++) x[i] = T(silu(float(x[i])));
    }
}

void silu(Tensor& x) {
    
    if (x.dtype() == CUDA_R_32F) {
        const long long N = x.num_elements();
        const int threads = 256;
        const int blocks = (N / 4 + threads - 1) / threads;
        silu_kernel<float><<<blocks, threads>>>((float*)x.data(),  N);
    } else if (x.dtype() == CUDA_R_16BF) {
        const long long N = x.num_elements();
        const int threads = 256;
        const int blocks = (N / 4 + threads - 1) / threads;
        silu_kernel<__nv_bfloat16><<<blocks, threads>>>((__nv_bfloat16*)x.data(),  N);
    } else if (x.dtype() == CUDA_R_16F) {
        const long long N = x.num_elements();
        const int threads = 256;
        const int blocks = (N / 4 + threads - 1) / threads;
        silu_kernel<__half><<<blocks, threads>>>((__half*)x.data(),  N);
    }

}