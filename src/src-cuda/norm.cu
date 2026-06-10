#include "norm.cuh"

template <typename T, typename T2>
__global__ void rmsnorm_kernel(
    T* __restrict__ x,
    const T2* __restrict__ weight,
    int   dim_size,
    float eps,
    float* scale_x,
    float* scale_w){
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;

    float sum = 0.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float tmp = (float) x[(long long)blockIdx.x * dim_size + i];
        if(scale_x != nullptr) tmp *= scale_x[0];
        sum += tmp * tmp;
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0){
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if(lane == 0){
            smem[0] = val / (float)dim_size;
        } 
    }
    __syncthreads();

    const float rms_inv = rsqrtf(smem[0] + eps);

    for (int i = tid; i < dim_size; i += blockDim.x) {
        float tmp = (float) x[(long long)blockIdx.x * dim_size + i];
        tmp = tmp * rms_inv * (float)weight[i];
        if(scale_w != nullptr) tmp *= scale_w[0];
        x[(long long)blockIdx.x * dim_size + i] = (T) tmp;
    }
}



void rmsnorm(Tensor& x, const Tensor& weight, float eps, int dim_size) {
    if(dim_size == -1){
        dim_size = x.shape.back();
    }

    int stride = 1;
    for(int i = 0; i < x.shape.size(); i++){
        stride *= x.shape[i];
    }

    stride = (int) stride / dim_size;

    const int block   = 256;
    const int smem_sz = (block / 32) * sizeof(float);
    if (x.dtype() == CUDA_R_32F && weight.dtype() == CUDA_R_32F) {
        rmsnorm_kernel<float, float><<<stride, block, smem_sz>>>(
            (float*)x.data(), (const float*)weight.data(),
            dim_size, eps, nullptr, nullptr);
    } else if (x.dtype() == CUDA_R_32F && weight.dtype() == CUDA_R_16BF) {
        rmsnorm_kernel<float, __nv_bfloat16><<<stride, block, smem_sz>>>(
            (float*)x.data(), (const __nv_bfloat16*)weight.data(),
            dim_size, eps, nullptr, nullptr);
        
    } else if (x.dtype() == CUDA_R_16BF && weight.dtype() == CUDA_R_16BF) {
        rmsnorm_kernel<__nv_bfloat16, __nv_bfloat16><<<stride, block, smem_sz>>>(
            (__nv_bfloat16*)x.data(), (const __nv_bfloat16*)weight.data(),
            dim_size, eps, nullptr, nullptr);
    } else if (x.dtype() == CUDA_R_16F && weight.dtype() == CUDA_R_16F) {
        rmsnorm_kernel<__half, __half><<<stride, block, smem_sz>>>(
            (__half*)x.data(), (const __half*)weight.data(),
            dim_size, eps, nullptr, nullptr);
    }
// #ifdef FP8_AVAILABLE
//     else if (x.dtype() == CUDA_R_8F_E4M3 && weight.dtype() == CUDA_R_8F_E4M3) {
//         rmsnorm_kernel<__nv_fp8_e4m3, __nv_fp8_e4m3><<<stride, block, smem_sz>>>(
//             (__nv_fp8_e4m3*)x.data(), (const __nv_fp8_e4m3*)weight.data(),
//             dim_size, eps, x._data->scale, weight._data->scale);
//     } else if (x.dtype() == CUDA_R_32F && weight.dtype() == CUDA_R_8F_E4M3) {
        
//         rmsnorm_kernel<float, __nv_fp8_e4m3><<<stride, block, smem_sz>>>(
//             (float*)x.data(), (const __nv_fp8_e4m3*)weight.data(),
//             dim_size, eps, nullptr, weight._data->scale);
//     } else if (x.dtype() == CUDA_R_16BF && weight.dtype() == CUDA_R_8F_E4M3) {

//         rmsnorm_kernel<__nv_bfloat16, __nv_fp8_e4m3><<<stride, block, smem_sz>>>(
//             (__nv_bfloat16*)x.data(), (const __nv_fp8_e4m3*)weight.data(),
//             dim_size, eps, nullptr, weight._data->scale);
//     }
// #endif
    
}
