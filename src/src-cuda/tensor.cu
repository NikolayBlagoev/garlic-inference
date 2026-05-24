#include "cuda-common.cuh"
#include "tensor.h"
#include "simple-ops.cuh"
// Processes 4 floats per thread: one float4 load, one uint64 store.
__global__ void f32_to_bf16(const float* __restrict__ A, __nv_bfloat16* __restrict__ B, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(A + i);
        uint64_t packed = (uint64_t)__bfloat16_as_ushort(__float2bfloat16_rn(v.x))
                        | ((uint64_t)__bfloat16_as_ushort(__float2bfloat16_rn(v.y)) << 16)
                        | ((uint64_t)__bfloat16_as_ushort(__float2bfloat16_rn(v.z)) << 32)
                        | ((uint64_t)__bfloat16_as_ushort(__float2bfloat16_rn(v.w)) << 48);
        *reinterpret_cast<uint64_t*>(B + i) = packed;
    } else {
        for (; i < N; i++)
            B[i] = __float2bfloat16_rn(A[i]);
    }
}

// Processes 4 bf16s per thread: one __nv_bfloat162 load, one float4 store.
__global__ void bf16_to_f32(const __nv_bfloat16* __restrict__ A, float* __restrict__ B, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float2 tmp1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(A + i));
        float2 tmp2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(A + i + 2));
        *reinterpret_cast<float4*>(B + i) = make_float4(tmp1.x, tmp1.y, tmp2.x, tmp2.y);
    } else {
        for (; i < N; i++)
            B[i] = __bfloat162float(A[i]);
    }
} 

#ifdef FP8_AVAILABLE
// Processes 4 floats per thread: one float4 load, one uint32 store.
__global__ void f32_to_f8(const float* __restrict__ A, __nv_fp8_e4m3* __restrict__ B, int N, float scale) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(A + i);
        // Need raw .__x to avoid conversion issues:
        // also the fact that is liddle endian messed me up so hard... a bit counterintuitive but some trial and error got it
        uint32_t packed = (uint32_t)__nv_fp8_e4m3(v.x / scale).__x
                        | ((uint32_t)__nv_fp8_e4m3(v.y / scale).__x << 8)
                        | ((uint32_t)__nv_fp8_e4m3(v.z / scale).__x << 16)
                        | ((uint32_t)__nv_fp8_e4m3(v.w / scale).__x << 24);
        *reinterpret_cast<uint32_t*>(B + i) = packed;
    } else {
        for (; i < N; i++){
            B[i] = __nv_fp8_e4m3(A[i]);
        }
    }
}
// Processes 4 fp8s per thread: one uint32 load, one float4 store.
__global__ void f8_to_f32(const __nv_fp8_e4m3* __restrict__ A, float* __restrict__ B, int N, float* scale) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float s = 1.0f;
    if (scale != nullptr) s = __ldg(scale);
    if (i + 3 < N) {
        uint32_t raw = *reinterpret_cast<const uint32_t*>(A + i);
        __nv_fp8_e4m3 x1,x2,x3,x4;
        x1.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x2.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x3.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x4.__x = raw & 0xFF; // load bottom most 2 bytes
        *reinterpret_cast<float4*>(B + i) = make_float4(((float) x1) * s, ((float) x2) * s, ((float) x3) * s, ((float) x4) * s);
        
        
    } else {
        for (; i < N; i++)
            B[i] = ((float)A[i]) * s;
    }
}
#endif


Tensor Tensor::cast_to(cudaDataType_t new_dtype) const {
    if(!_data) return *this;
    if(dtype() == new_dtype) return *this;
    cudaSetDevice(device());
    Tensor out(shape, new_dtype, device());
    if (dtype() == CUDA_R_32F && new_dtype == CUDA_R_16BF) {
        int N = out.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        f32_to_bf16<<<blocks, threads>>>(static_cast<const float*>(data()), static_cast<__nv_bfloat16*>(out.data()), N);
    } else if (dtype() == CUDA_R_16BF && new_dtype == CUDA_R_32F) {
        int N = out.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        bf16_to_f32<<<blocks, threads>>>(static_cast<const __nv_bfloat16*>(data()), static_cast<float*>(out.data()), N);
    } 
#ifdef FP8_AVAILABLE
    else if (dtype() == CUDA_R_8F_E4M3 && new_dtype == CUDA_R_32F) {
        int N = out.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        f8_to_f32<<<blocks, threads>>>(static_cast<const __nv_fp8_e4m3*>(data()), static_cast<float*>(out.data()), N, scale());

    } else if(dtype() == CUDA_R_32F && new_dtype == CUDA_R_8F_E4M3){
        max_t(*this, out._data->scale);
        out.update_scale();
        int N = out.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        f32_to_f8<<<blocks, threads>>>(static_cast<const float*>(data()), static_cast<__nv_fp8_e4m3*>(out.data()), N, out.scale_val());
    } 
    
#endif

    return out;
}

