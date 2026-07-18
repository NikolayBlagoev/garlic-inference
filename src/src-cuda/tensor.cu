#include "cuda-common.cuh"
#include "tensor.h"
#include "simple-ops.cuh"

cudaStream_t compute_stream = nullptr;
cudaStream_t load_offload_stream = nullptr;
cudaStream_t secondary_compute_stream = nullptr;


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


__global__ void f16_to_bf16(__half* __restrict__ A, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if(i >= N) return;
    *reinterpret_cast<__nv_bfloat16*>(A + i) =(__nv_bfloat16)((float)(A[i]));
    
} 


__global__ void swap_kernel(uint8_t* __restrict__ a,
                            uint8_t* __restrict__ b,
                            size_t num_elms){
    const int n16 = (num_elms / 8) * 8;
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    size_t stride = (size_t)gridDim.x * blockDim.x * 8;
    for (; i < n16; i += stride) {
        uint64_t t1 = *reinterpret_cast<const uint64_t*>(a+i);
        uint64_t t2 = *reinterpret_cast<const uint64_t*>(b+i);
        *reinterpret_cast<uint64_t*>(a + i) = t2;
        *reinterpret_cast<uint64_t*>(b + i) = t1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = n16; i < num_elms; i++) {
            const uint8_t tmp = a[i];
            a[i]  = b[i];
            b[i] = tmp;
        }
    }
}

void Tensor::swap(Tensor& a, Tensor& b, cudaStream_t stream){
    if(a.on_cpu() == b.on_cpu()) return;
    int dev = 0;
    int sms = 0; 
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    sms = sms * 4;
    swap_kernel<<<sms, 256, 0, stream>>>(reinterpret_cast<uint8_t*>(a.data()), reinterpret_cast<uint8_t*>(b.data()), a.num_elements()*Tensor::element_size(a.dtype()));
    bool tmp = b.on_cpu();
    b._data->on_cpu = a.on_cpu();
    a._data->on_cpu = tmp;
    std::swap(a._data->data, b._data->data);

    return;
}
Tensor Tensor::cast_to(cudaDataType_t new_dtype) const {
    if(!_data) return *this;
    if(dtype() == new_dtype) return *this;
    cudaSetDevice(device());
    if(dtype() == CUDA_R_16F && new_dtype == CUDA_R_16BF){
        std::cout<<"ERROR"<<std::endl;
        std::exit(1);
        // Tensor out(shape, new_dtype, device());
        // int N = num_elements();
        // int threads = 1024;
        // int blocks = (N + threads - 1) / threads;
        // // copy first, then convert in place in out's own allocation
        // cudaMemcpy(out.data(), data(), (size_t)N * sizeof(__half), cudaMemcpyDeviceToDevice);
        // f16_to_bf16<<<blocks, threads>>>(static_cast<__half*>(out.data()), N);
        // return out;
    }
    Tensor out(shape, new_dtype, device());
    if (dtype() == CUDA_R_32F && new_dtype == CUDA_R_16BF) {
        int N = out.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        f32_to_bf16<<<blocks, threads, 0, get_compute_stream()>>>(static_cast<const float*>(data()), static_cast<__nv_bfloat16*>(out.data()), N);
    } else if (dtype() == CUDA_R_16BF && new_dtype == CUDA_R_32F) {
        int N = out.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        bf16_to_f32<<<blocks, threads, 0, get_compute_stream()>>>(static_cast<const __nv_bfloat16*>(data()), static_cast<float*>(out.data()), N);
    } 

    return out;
}

