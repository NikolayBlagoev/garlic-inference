#include "mat-ops.cuh"

// __global__ void elm_wise_kernel_f32_f8(
//     float* __restrict__ x,
//     const __nv_fp8_e4m3* __restrict__ w,
//     long long stride,
//     long long N,
//     float* scale_w) {

//     int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
//     if (i + 3 < N) {
//         uint32_t raw = *reinterpret_cast<const uint32_t*>(w + (i % stride));
//         float4 v = *reinterpret_cast<const float4*>(x + i);
//         __nv_fp8_e4m3 x1,x2,x3,x4;
//         x1.__x = raw & 0xFF; // load bottom most 2 bytes
//         raw = raw >> 8;
//         x2.__x = raw & 0xFF; // load bottom most 2 bytes
//         raw = raw >> 8;
//         x3.__x = raw & 0xFF; // load bottom most 2 bytes
//         raw = raw >> 8;
//         x4.__x = raw & 0xFF; // load bottom most 2 bytes
//         *reinterpret_cast<float4*>(x + i) = make_float4(((float) x1) * v.x * scale_w[0], ((float) x2) * v.y * scale_w[0], ((float) x3)  * v.z * scale_w[0], ((float) x4)  * v.w * scale_w[0]);

//     } else {
//         for (; i < N; i++)
//             x[i] = (float)w[i % stride] * scale_w[0] * x[i];
//     }
// }

__global__ void elm_wise_kernel_f32_f32(
    float* __restrict__ x,
    const float* __restrict__ w,
    long long stride,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(x + i);
        float4 v2 = *reinterpret_cast<const float4*>(w + (i%stride));
        *reinterpret_cast<float4*>(x + i) = make_float4(v.x*v2.x, v.y*v2.y, v.z*v2.z, v.w*v2.w);

    } else {
        for (; i < N; i++)
            x[i] = w[i % stride] * x[i];
    }
}

__global__ void elm_wise_kernel_bf16_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    long long stride,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        __nv_bfloat162  v_0 = *reinterpret_cast<const __nv_bfloat162 *>(x + i);
        __nv_bfloat162  v2_0 = *reinterpret_cast<const __nv_bfloat162 *>(w + (i%stride));
        *reinterpret_cast<__nv_bfloat162*>(x + i) = __nv_bfloat162(v_0.x*v2_0.x, v_0.y*v2_0.y);
        v_0 = *reinterpret_cast<const __nv_bfloat162 *>(x + i + 2);
        v2_0 = *reinterpret_cast<const __nv_bfloat162 *>(w + ((i+2)%stride));
        *reinterpret_cast<__nv_bfloat162*>(x + i + 2) = __nv_bfloat162(v_0.x*v2_0.x, v_0.y*v2_0.y);

    } else {
        for (; i < N; i++)
            x[i] = w[i % stride] * x[i];
    }
}


__global__ void elm_wise_kernel_f16_f16(
    __half* __restrict__ x,
    const __half* __restrict__ w,
    long long stride,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        __half2  v_0 = *reinterpret_cast<const __half2 *>(x + i);
        __half2  v2_0 = *reinterpret_cast<const __half2 *>(w + (i%stride));
        *reinterpret_cast<__half2*>(x + i) =__half2(v_0.x*v2_0.x, v_0.y*v2_0.y);
        v_0 = *reinterpret_cast<const __half2 *>(x + i + 2);
        v2_0 = *reinterpret_cast<const __half2 *>(w + ((i+2)%stride));
        *reinterpret_cast<__half2*>(x + i + 2) = __half2(v_0.x*v2_0.x, v_0.y*v2_0.y);

    } else {
        for (; i < N; i++)
            x[i] = w[i % stride] * x[i];
    }
}
// TODO: make it x = a*x*W, where a is a scalar
// compute x = x * w
void elm_wise(Tensor& x, const Tensor& w, cudaStream_t compute_stream) {
    if(compute_stream == nullptr){
        compute_stream = get_compute_stream();
    }
    const long long N = x.num_elements();
    const long long stride = w.num_elements();
    const int threads = 256;
    const int blocks = (N / 4 + threads - 1) / threads;
    if (x.dtype() == CUDA_R_32F && w.dtype() == CUDA_R_32F) {
        elm_wise_kernel_f32_f32<<<blocks, threads, 0, compute_stream>>>(
            (float*)x.data(), (const float*)w.data(),
            stride, N);
    } else if (x.dtype() == CUDA_R_16BF && w.dtype() == CUDA_R_16BF) {
        elm_wise_kernel_bf16_bf16<<<blocks, threads, 0, compute_stream>>>(
            (__nv_bfloat16*)x.data(), (const __nv_bfloat16*)w.data(),
            stride, N);

    } else if (x.dtype() == CUDA_R_16F && w.dtype() == CUDA_R_16F) {
        elm_wise_kernel_f16_f16<<<blocks, threads, 0, compute_stream>>>(
            (__half*)x.data(), (const __half*)w.data(),
            stride, N);
        
    }
}