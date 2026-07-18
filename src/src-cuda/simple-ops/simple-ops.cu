#include "simple-ops.cuh"

__global__ void transpose_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int elem_size,
    int N,
    int outer_size, int A, int mid_size, int B_dim, int inner_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int rem       = i;
    int inner_idx = rem % inner_size; rem /= inner_size;
    int a_idx     = rem % A;          rem /= A;
    int mid_idx   = rem % mid_size;   rem /= mid_size;
    int b_idx     = rem % B_dim;
    int outer_idx = rem / B_dim;

    int in_i = outer_idx * (A * mid_size * B_dim * inner_size)
             + a_idx     * (mid_size * B_dim * inner_size)
             + mid_idx   * (B_dim * inner_size)
             + b_idx     *  inner_size
             + inner_idx;

    memcpy(dst + (long long)i * elem_size,
           src + (long long)in_i * elem_size,
           elem_size);
}

Tensor transpose(const Tensor& inp, int d0, int d1) {
    if (d0 == d1) return inp;

    if (d0 > d1) { 
        //swap so d0 < d1
        int tmp = d0; 
        d0 = d1; 
        d1 = tmp; 
    }
    int ndim = inp.ndim();
    int outer_size = 1, mid_size = 1, inner_size = 1;
    for (int d = 0; d < d0; d++) outer_size *= inp.shape[d];
    int A = inp.shape[d0];
    for (int d = d0+1; d < d1; d++) mid_size *= inp.shape[d];
    int B_dim = inp.shape[d1];
    for (int d = d1+1; d < ndim; d++) inner_size *= inp.shape[d];
    if (A == 1 || B_dim == 1) {
        Tensor out = inp;
        out.shape[d0] = B_dim;
        out.shape[d1] = A;
        return out;
    }
    std::vector<int> out_shape = inp.shape;
    out_shape[d0] = B_dim;
    out_shape[d1] = A;

    cudaSetDevice(inp.device());
    Tensor out(out_shape, inp.dtype(), inp.device());
    int N       = inp.num_elements();
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    int esz     = Tensor::element_size(inp.dtype());

    transpose_kernel<<<blocks, threads, 0, get_compute_stream()>>>(
        static_cast<const char*>(inp.data()),
        static_cast<char*>(out.data()),
        esz, N,
        outer_size, A, mid_size, B_dim, inner_size
    );
    return out;
}

__global__ void fill_kernel_f32(float* data, float val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = val;
}

__global__ void fill_kernel_bf16(__nv_bfloat16* data, __nv_bfloat16 val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = val;
}

void fill_inplace(Tensor& t, float val){
    if(!t._data) return;
    if(!t._data->initialized) return;
    int N = t.num_elements();
    int threads = 256, blocks = (N + threads - 1) / threads;

    if(t.dtype() == CUDA_R_32F){
        fill_kernel_f32<<<blocks, threads, 0, get_compute_stream()>>>((float*)t.data(), val, N);
    }else if(t.dtype() == CUDA_R_16BF){
        fill_kernel_bf16<<<blocks, threads, 0, get_compute_stream()>>>((__nv_bfloat16*)t.data(), __nv_bfloat16(val), N);
    }
}
// Processes 4 floats per thread: one float4 load, one uint64 store.
__global__ void arange_kernel(
                uint32_t* __restrict__ data,
                int a,
                int step,
                int repeat,
                int N){
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i + 3 < N) {
        uint32_t i1 = ((i + 0) % repeat) * step + a;
        uint32_t i2 = ((i + 1) % repeat) * step + a;
        uint32_t i3 = ((i + 2) % repeat) * step + a;
        uint32_t i4 = ((i + 3) % repeat) * step + a;
        *reinterpret_cast<uint4*>(data + i) = make_uint4(i1, i2, i3, i4);
    } else {
        for (; i < N; i++) data[i] = (i  % repeat) * step + a;
    }
}

void arrange(Tensor& t, int a, int step){
    int repeat = t.shape.back();
    int N = t.num_elements();
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + 4 * threads - 1) / (4 * threads);
    arange_kernel<<<blocks, threads, 0, get_compute_stream()>>>(static_cast<uint32_t*>(t.data()), a, step, repeat, N);
}

__global__ void pow_kernel_f32(
                float* __restrict__ data,
                float val,
                int N){
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(data + i);
        
        *reinterpret_cast<float4*>(data + i) = make_float4(powf(v.x, val), powf(v.y, val), powf(v.z, val), powf(v.w, val));
    } else {
        for (; i < N; i++) data[i] = powf(data[i], val);
    }
}

void pow(Tensor& inp, float val){
    if(inp.dtype() == CUDA_R_32F){
        int N = inp.num_elements();
        int threads = 256;
        int blocks = (N / 4 + threads - 1) / threads;
        pow_kernel_f32<<<blocks, threads, 0, get_compute_stream()>>>(static_cast<float*>(inp.data()), val, N);
    }
}


