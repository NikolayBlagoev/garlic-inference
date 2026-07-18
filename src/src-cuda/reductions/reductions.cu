#include "reductions.cuh"

template <typename T>
__global__ void max_kernel(
                const T* __restrict__ inp, 
                float* __restrict__ out, 
                const int N) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5; // /32
    const int lane = tid & 31; // %32
    const int num_warps = blockDim.x >> 5; // /32

    float val = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
        val = MAX(val, fabsf((float)inp[i]));

    val = warp_reduce_max(val);
    if (lane == 0){
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_max(val);
        if (lane == 0) atomicMax((int*)out, __float_as_int(val));
    }
}


template <typename T>
__global__ void reduce_kernel(
    const T* __restrict__ inp,
    T*   __restrict__ out,
    int dim_size,
    bool norm){

    extern __shared__ float smem[];
    float sum = 0.0f;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5; // /32
    const int lane = tid & 31; // %32
    const int num_warps = blockDim.x >> 5; // /32
    for (int k = tid; k < dim_size; k += blockDim.x)
        sum += (float)inp[(long long)blockIdx.x * dim_size + k];
        

    sum = warp_reduce_sum(sum);

    if (lane == 0){
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if(lane == 0){
            if(norm){
                out[blockIdx.x] = (T)(val / (float)dim_size);
            }else{
                out[blockIdx.x] = (T)(val);
            }
        } 
    }
}

void reduce(Tensor& inp, Tensor& outp, int dim, bool norm) {
    int ndim = (int)inp.shape.size();
    if (dim < 0) dim += ndim;
    int dim_prod = 1;
    int dim_size = inp.shape[dim];

    for(int i = 0; i < ndim; i++){
        if(i == dim) continue;
        dim_prod *= inp.shape[i];
    }

    const int block   = 256;
    const int smem_sz = (block / 32) * sizeof(float);
    

    if (inp.dtype() == CUDA_R_32F) {
        reduce_kernel<float><<<dim_prod, block, smem_sz, get_compute_stream()>>>(
            (const float*) (inp.data()), (float*) (outp.data()), dim_size, norm);
    } else if (inp.dtype() == CUDA_R_16BF) {
        reduce_kernel<__nv_bfloat16><<<dim_prod, block, smem_sz, get_compute_stream()>>>(
            (const __nv_bfloat16*) (inp.data()), (__nv_bfloat16*) (outp.data()), dim_size, norm);
    }
#ifdef FP8_AVAILABLE
    // TODO: might need to do it differently here... float8 maxes out at 448.0f so it can give errors...
    else if (inp.dtype() == CUDA_R_8F_E4M3) {
        reduce_kernel<__nv_fp8_e4m3><<<dim_prod, block, smem_sz, get_compute_stream()>>>(
            (const __nv_fp8_e4m3*) (inp.data()), (__nv_fp8_e4m3*) (outp.data()), dim_size, norm);
    }
#endif
}



void max_t(const Tensor& inp, float* outp) {
    cudaMemsetAsync(outp, 0, sizeof(float), get_compute_stream());

    const int N       = inp.num_elements();
    const int threads = 256;
    const int blocks  = MIN(256, (N + threads - 1) / threads);
    const int smem_sz = (threads / 32) * sizeof(float);

    if (inp.dtype() == CUDA_R_32F)
        max_kernel<float><<<blocks, threads, smem_sz, get_compute_stream()>>>((const float*)(inp.data()), outp, N);
    else if (inp.dtype() == CUDA_R_16BF)
        max_kernel<__nv_bfloat16><<<blocks, threads, smem_sz, get_compute_stream()>>>((const __nv_bfloat16*)(inp.data()), outp, N);
#ifdef FP8_AVAILABLE
    else if (inp.dtype() == CUDA_R_8F_E4M3)
        max_kernel<__nv_fp8_e4m3><<<blocks, threads, smem_sz, get_compute_stream()>>>((const __nv_fp8_e4m3*)(inp.data()), outp, N);
#endif
}