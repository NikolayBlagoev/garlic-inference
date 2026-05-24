#include "activations.cuh"

__device__ __forceinline__ float silu(float x) {
    return x / (1 + expf(-x));
}

__global__ void silu_kernel_f32(
    float* __restrict__ x,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(x + i);

        *reinterpret_cast<float4*>(x + i) = make_float4(silu(v.x), silu(v.y), silu(v.z), silu(v.w));
    } else {
        for (; i < N; i++)
            x[i] = silu(x[i]);
    }
}

__global__ void silu_kernel_bf16(
    __nv_bfloat16* __restrict__ x,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(x + i);
        *reinterpret_cast<__nv_bfloat162*>(x + i) = __nv_bfloat162(__nv_bfloat16(silu(v.x)), __nv_bfloat16(silu(v.y)));
        v = *reinterpret_cast<const __nv_bfloat162*>(x + i + 2);
        *reinterpret_cast<__nv_bfloat162*>(x + i + 2) = __nv_bfloat162(__nv_bfloat16(silu(v.x)), __nv_bfloat16(silu(v.y)));
    } else {
        for (; i < N; i++)
            x[i] = __nv_bfloat16(silu(x[i]));
    }
}


void silu(Tensor& x) {
    const long long N = x.num_elements();
    const int threads = 256;
    const int blocks = (N / 4 + threads - 1) / threads;
    if (x.dtype() == CUDA_R_32F) {
        
        silu_kernel_f32<<<blocks, threads>>>((float*)x.data(),  N);
    } else if (x.dtype() == CUDA_R_16BF) {
        silu_kernel_bf16<<<blocks, threads>>>((__nv_bfloat16*)x.data(),  N);
    }

}