// #include "cutlass-runner.cuh"
#include "mat-ops.cuh"



__global__ void elm_wise_kernel_f32_f8(
    float* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
    long long stride,
    long long N,
    float* scale_w) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        uint32_t raw = *reinterpret_cast<const uint32_t*>(w + (i % stride));
        float4 v = *reinterpret_cast<const float4*>(x + i);
        __nv_fp8_e4m3 x1,x2,x3,x4;
        x1.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x2.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x3.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x4.__x = raw & 0xFF; // load bottom most 2 bytes
        *reinterpret_cast<float4*>(x + i) = make_float4(((float) x1) * v.x * scale_w[0], ((float) x2) * v.y * scale_w[0], ((float) x3)  * v.z * scale_w[0], ((float) x4)  * v.w * scale_w[0]);

    } else {
        for (; i < N; i++)
            x[i] = (float)w[i % stride] * scale_w[0] * x[i];
    }
}

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


void elm_wise(Tensor& x, const Tensor& w) {
    const long long N = x.num_elements();
    const long long stride = w.num_elements();
    const int threads = 256;
    const int blocks = (N / 4 + threads - 1) / threads;
    if (x.dtype() == CUDA_R_32F && w.dtype() == CUDA_R_32F) {
        elm_wise_kernel_f32_f32<<<blocks, threads, 0, get_compute_stream()>>>(
            (float*)x.data(), (const float*)w.data(),
            stride, N);
    } else if (x.dtype() == CUDA_R_16BF && w.dtype() == CUDA_R_16BF) {
        elm_wise_kernel_bf16_bf16<<<blocks, threads, 0, get_compute_stream()>>>(
            (__nv_bfloat16*)x.data(), (const __nv_bfloat16*)w.data(),
            stride, N);

    } else if (x.dtype() == CUDA_R_16F && w.dtype() == CUDA_R_16F) {
        elm_wise_kernel_f16_f16<<<blocks, threads, 0, get_compute_stream()>>>(
            (__half*)x.data(), (const __half*)w.data(),
            stride, N);
        
    }
}


__global__ void add_inplace_kernel_f32_f8(
    float* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ w,
    long long stride,
    long long N,
    float* scale_w) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        uint32_t raw = *reinterpret_cast<const uint32_t*>(w + (i % stride));
        float4 v = *reinterpret_cast<const float4*>(x + i);
        __nv_fp8_e4m3 x1,x2,x3,x4;
        x1.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x2.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x3.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x4.__x = raw & 0xFF; // load bottom most 2 bytes
        *reinterpret_cast<float4*>(x + i) = make_float4(((float) x1)* scale_w[0] + v.x, ((float) x2) * scale_w[0] + v.y, ((float) x3) * scale_w[0] + v.z, ((float) x4) * scale_w[0] + v.w );

    } else {
        for (; i < N; i++)
            x[i] = (float)w[i % stride] * scale_w[0] + x[i];
    }
}

__global__ void add_inplace_kernel_f32_f32(
    float* __restrict__ x,
    const float* __restrict__ w,
    long long stride,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        float4 v = *reinterpret_cast<const float4*>(x + i);
        float4 v2 = *reinterpret_cast<const float4*>(w + (i%stride));
        *reinterpret_cast<float4*>(x + i) = make_float4(v.x+v2.x, v.y+v2.y, v.z+v2.z, v.w+v2.w);

    } else {
        for (; i < N; i++)
            x[i] = w[i % stride] + x[i];
    }
}

__global__ void add_inplace_kernel_bf16_bf16(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    long long stride,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        __nv_bfloat162  v_0 = *reinterpret_cast<const __nv_bfloat162 *>(x + i);
        __nv_bfloat162  v2_0 = *reinterpret_cast<const __nv_bfloat162 *>(w + (i%stride));
        *reinterpret_cast<__nv_bfloat162*>(x + i) = __nv_bfloat162(v_0.x+v2_0.x, v_0.y+v2_0.y);
        v_0 = *reinterpret_cast<const __nv_bfloat162 *>(x + i + 2);
        v2_0 = *reinterpret_cast<const __nv_bfloat162 *>(w + ((i+2)%stride));
        *reinterpret_cast<__nv_bfloat162*>(x + i + 2) = __nv_bfloat162(v_0.x+v2_0.x, v_0.y+v2_0.y);

    } else {
        for (; i < N; i++)
            x[i] = w[i % stride] + x[i];
    }
}

__global__ void add_inplace_kernel_f16_f16(
    __half* __restrict__ x,
    const __half* __restrict__ w,
    long long stride,
    long long N) {

    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < N) {
        __half2  v_0 = *reinterpret_cast<const __half2 *>(x + i);
        __half2  v2_0 = *reinterpret_cast<const __half2 *>(w + (i%stride));
        *reinterpret_cast<__half2*>(x + i) = __half2(v_0.x+v2_0.x, v_0.y+v2_0.y);
        v_0 = *reinterpret_cast<const __half2 *>(x + i + 2);
        v2_0 = *reinterpret_cast<const __half2 *>(w + ((i+2)%stride));
        *reinterpret_cast<__half2*>(x + i + 2) = __half2(v_0.x+v2_0.x, v_0.y+v2_0.y);

    } else {
        for (; i < N; i++)
            x[i] = w[i % stride] + x[i];
    }
}


//computes x = x + w
void add_inplace(Tensor& x, Tensor& w) {
const long long N = x.num_elements();
    const long long stride = w.num_elements();
    const int threads = 256;
    const int blocks = (N / 4 + threads - 1) / threads;
    if (x.dtype() == CUDA_R_32F && w.dtype() == CUDA_R_32F) {
        add_inplace_kernel_f32_f32<<<blocks, threads, 0, get_compute_stream()>>>(
            (float*)x.data(), (const float*)w.data(),
            stride, N);
    } else if (x.dtype() == CUDA_R_16BF && w.dtype() == CUDA_R_16BF) {
        add_inplace_kernel_bf16_bf16<<<blocks, threads, 0, get_compute_stream()>>>(
            (__nv_bfloat16*)x.data(), (const __nv_bfloat16*)w.data(),
            stride, N);
    } else if (x.dtype() == CUDA_R_16F && w.dtype() == CUDA_R_16F) {
        add_inplace_kernel_f16_f16<<<blocks, threads, 0, get_compute_stream()>>>(
            (__half*)x.data(), (const __half*)w.data(),
            stride, N);
    }
}

// torch does column major it seems
void matmul(Tensor& y, Tensor& x, const Tensor& W, cudaStream_t compute_stream) {
    int x_dims = x.ndim();
    if(compute_stream == nullptr){
        compute_stream = get_compute_stream();
    }
    // static_assert(W.shape[1] == x.shape[x_dims - 1]);
#ifdef FP8_AVAILABLE
    // FP8 weight with 2-D block scales: dequant to BF16 then GEMM.
    if (W.dtype() == CUDA_R_8F_E4M3 && W.ndim_scale() == 2) {
        matmul_fp8_blockscale(y, x, W);
        return;
    }
    // if(x.dtype() == CUDA_R_8F_E4M3 && W.dtype() == CUDA_R_8F_E4M3){
    //     matmul_float8(y,x,W);
    //     return;
    // }
    // if(x.dtype() != CUDA_R_8F_E4M3 && W.dtype() == CUDA_R_8F_E4M3){
    //     x = x.cast_to(CUDA_R_8F_E4M3);
    //     matmul_float8(y,x,W);
    //     return;
    // }
#endif
    // if(x.dtype() != W.dtype()){
    //     x = x.cast_to(W.dtype());
    // }
    // std::cout<<x.dtype()<<" "<<W.dtype()<<std::endl; 
    int B = x.shape[0];
    int M = x.shape[1];
    int N = x.shape[2]; // in_features
    int K = W.shape[0]; // out_features

    // unfortunately cublas needs these for y = alpha*W*x + beta*y... might be useful to make them into function params
    const float alpha = 1.0f;
    const float beta = 0.0f;


    cublasSetStream(cublas_handle(), compute_stream);
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        cublas_handle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        K,           
        M, 
        N,
        &alpha,
        W.data(), W.dtype(),
        N, 
        0,  // 0 stride - one W for all batched elements
        x.data(), x.dtype(),
        N,  
        (long long) M * N,
        &beta,
        y.data(), y.dtype(),
        K,
        (long long) M * K,
        B,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
}

#ifdef FP8_AVAILABLE
static void matmul_fp8_blockscale_dequant(Tensor& y, Tensor& x, const Tensor& W) {
    int n = W.num_elements();
    if (n > dequant_buffer_nelems || !dequant_buffer._data || dequant_buffer.dtype() != x.dtype()) {
        dequant_buffer = Tensor(W.shape, x.dtype(), W.device());
        dequant_buffer_nelems = n;
    }
    dequant_buffer.shape = W.shape;

    dequant_fp8_blockscale(dequant_buffer, W);
    matmul(y, x, dequant_buffer);
}

static constexpr int kGemvBN   = 8;   // output rows per CTA (one warp each)
static constexpr int kGemvSK   = 128; // K scale granularity

// W: [N, K] FP8-E4M3, x: [M, K] BF16 -> y: [M, N] BF16
// W_scale: [ceil(N/128), ceil(K/128)] float, block-wise
//
// Block (32, kGemvBN): threadIdx.y selects one output row; all 32 lanes in
// that row form one warp and reduce across K via __shfl_xor_sync.
__global__ void fp8_gemv_groupwise_bf16_kernel(
        __nv_bfloat16* __restrict__ y,
        const __nv_fp8_e4m3* __restrict__ W,
        const float* __restrict__ W_scale,
        const __nv_bfloat16* __restrict__ x,
        int N, int K, int M, int scale_N, int scale_K)
{
    const int tx = threadIdx.x & 31;           // lane: 0..31
    const int ty = threadIdx.y;           // row slot: 0..kGemvBN-1
    const int n  = blockIdx.x * kGemvBN + ty;
    const int m  = blockIdx.y;

    if (n >= N || m >= M) return;

    const __nv_fp8_e4m3* w_row = W + (long long)n * K;
    const __nv_bfloat16* x_row = x + (long long)m * K;

    int s_col = n * scale_N / N;
    float factor = K / scale_K;

    float acc = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        float scale = __ldg(W_scale + s_col * scale_K + kg);
        const int k = kg * kGemvSK + tx * 4;
        float partial = 0.0f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w_row + k);
            
            __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);

            float4 wf = float4(w4);
            float2 xf01 = __bfloat1622float2(x01);
            float2 xf23 = __bfloat1622float2(x23);

            partial = fmaf(wf.x, xf01.x, partial);
            partial = fmaf(wf.y, xf01.y, partial);
            partial = fmaf(wf.z, xf23.x, partial);
            partial = fmaf(wf.w, xf23.y, partial);
        } else {
            for (int kk = k; kk < K && kk < k + 4; kk++)
                partial = fmaf(float(w_row[kk]), __bfloat162float(x_row[kk]), partial);
        }
        acc = fmaf(partial,scale,acc);
    }
    acc = warp_reduce_sum<32>(acc);
    
    if (tx == 0)
        y[(long long)m * N + n] = __float2bfloat16_rn(acc);
}

__global__ void fp8_gemv_groupwise_f16_kernel(
        __half* __restrict__ y,
        const __nv_fp8_e4m3* __restrict__ W,
        const float* __restrict__ W_scale,
        const __half* __restrict__ x,
        int N, int K, int M, int scale_N, int scale_K)
{
    const int tx = threadIdx.x & 31;           // lane: 0..31
    const int ty = threadIdx.y;           // row slot: 0..kGemvBN-1
    const int n  = blockIdx.x * kGemvBN + ty;
    const int m  = blockIdx.y;

    if (n >= N || m >= M) return;

    const __nv_fp8_e4m3* w_row = W + (long long)n * K;
    const __half* x_row = x + (long long)m * K;

    int s_col = n * scale_N / N;
    float factor = K / scale_K;

    float acc = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        float scale = __ldg(W_scale + s_col * scale_K + kg);
        const int k = kg * kGemvSK + tx * 4;
        float partial = 0.0f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w_row + k);
            
            __half2 x01 = *reinterpret_cast<const __half2*>(x_row + k);
            __half2 x23 = *reinterpret_cast<const __half2*>(x_row + k + 2);

            float4 wf = float4(w4);
            float2 xf01 = __half22float2(x01);
            float2 xf23 = __half22float2(x23);

            partial = fmaf(wf.x, xf01.x, partial);
            partial = fmaf(wf.y, xf01.y, partial);
            partial = fmaf(wf.z, xf23.x, partial);
            partial = fmaf(wf.w, xf23.y, partial);
        } else {
            for (int kk = k; kk < K && kk < k + 4; kk++)
                partial = fmaf(float(w_row[kk]), __half2float(x_row[kk]), partial);
        }
        acc = fmaf(partial,scale,acc);
    }
    acc = warp_reduce_sum<32>(acc);
    
    if (tx == 0)
        y[(long long)m * N + n] = __float2half_rn(acc);
}

void fp8_gemv_groupwise_bf16(Tensor& y, const Tensor& W, const Tensor& x) {
    int K = W.shape[1];
    int N = W.shape[0];
    // std::cout<<y.shape[0]*y.shape[1]<<"x"<<y.shape[2]<<" "<<x.shape[0]*x.shape[1]<<"x"<<x.shape[2]<<" "<<W.shape[0]<<"x"<<W.shape[1]<<" "<<W.shape_scale[0]<<"x"<<W.shape_scale[1]<<std::endl;
    int M = x.shape[0] * x.shape[1];
    // if (N != K) std::printf("[gemv] W=%dx%d scale=%dx%d\n",
    //     N, K, W.shape_scale[0], W.shape_scale[1]);
    dim3 block(32, kGemvBN);
    dim3 grid((N + kGemvBN - 1) / kGemvBN, M);
    if(y.dtype() == CUDA_R_16BF){
        fp8_gemv_groupwise_bf16_kernel<<<grid, block, 0, get_compute_stream()>>>(
            (__nv_bfloat16*)y.data(),
            (const __nv_fp8_e4m3*)W.data(), W.scale(),
            (const __nv_bfloat16*)x.data(),
            N, K, M, W.shape_scale[0], W.shape_scale[1]);
    }else if(y.dtype() == CUDA_R_16F){
        fp8_gemv_groupwise_f16_kernel<<<grid, block, 0, get_compute_stream()>>>(
            (__half*)y.data(),
            (const __nv_fp8_e4m3*)W.data(), W.scale(),
            (const __half*)x.data(),
            N, K, M, W.shape_scale[0], W.shape_scale[1]);
    } else {
        std::cerr << "[fp8_gemv] unsupported output dtype " << (int)y.dtype() << " — no kernel launched!\n";
        std::exit(1);
    }
}

void matmul_fp8_blockscale(Tensor& y, Tensor& x, const Tensor& W) {
    int B = x.shape[0];
    int S = x.shape[1];
    int K = x.shape[2];   // in_features
    int N = W.shape[0];   // out_features
    int M = B * S;
    if(M > 0){
        return fp8_gemv_groupwise_bf16(y,W,x);
    }

    matmul_fp8_blockscale_dequant(y,x,W);


}
__device__ __forceinline__ float silu(float x) {
    return x / (1 + expf(-x));
}
__global__ void fp8_grouped_groupwise_bf16_kernel(
        __nv_bfloat16* __restrict__ y,
        const __nv_fp8_e4m3* __restrict__ W1,
        const float* __restrict__ W1_scale,
        const __nv_fp8_e4m3* __restrict__ W2,
        const float* __restrict__ W2_scale,
        const __nv_bfloat16* __restrict__ x,
        int N, int K, int M, int scale_N, int scale_K)
{
    const int tx = threadIdx.x & 31;           // lane: 0..31
    const int ty = threadIdx.y;           // row slot: 0..kGemvBN-1
    const int n  = blockIdx.x * kGemvBN + ty;
    const int m  = blockIdx.y;

    if (n >= N || m >= M) return;

    const __nv_fp8_e4m3* w1_row = W1 + (long long)n * K;
    const __nv_fp8_e4m3* w2_row = W2 + (long long)n * K;
    const __nv_bfloat16* x_row = x + (long long)m * K;

    int s_col = n * scale_N / N;
    float factor = K / scale_K;

    float acc = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        float scale = __ldg(W1_scale + s_col * scale_K + kg);
        const int k = kg * kGemvSK + tx * 4;
        float partial = 0.0f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w1_row + k);
            
            __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);

            float4 wf = float4(w4);
            float2 xf01 = __bfloat1622float2(x01);
            float2 xf23 = __bfloat1622float2(x23);

            partial = fmaf(wf.x, xf01.x, partial);
            partial = fmaf(wf.y, xf01.y, partial);
            partial = fmaf(wf.z, xf23.x, partial);
            partial = fmaf(wf.w, xf23.y, partial);
        } else {
            for (int kk = k; kk < K && kk < k + 4; kk++)
                partial = fmaf(float(w1_row[kk]), __bfloat162float(x_row[kk]), partial);
        }
        acc = fmaf(partial,scale,acc);
    }


    acc = warp_reduce_sum<32>(acc);

    float acc2 = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        float scale = __ldg(W2_scale + s_col * scale_K + kg);
        const int k = kg * kGemvSK + tx * 4;
        float partial = 0.0f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w2_row + k);
            
            __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);

            float4 wf = float4(w4);
            float2 xf01 = __bfloat1622float2(x01);
            float2 xf23 = __bfloat1622float2(x23);

            partial = fmaf(wf.x, xf01.x, partial);
            partial = fmaf(wf.y, xf01.y, partial);
            partial = fmaf(wf.z, xf23.x, partial);
            partial = fmaf(wf.w, xf23.y, partial);
        } else {
            for (int kk = k; kk < K && kk < k + 4; kk++)
                partial = fmaf(float(w2_row[kk]), __bfloat162float(x_row[kk]), partial);
        }
        acc2 = fmaf(partial,scale,acc2);
    }


    acc2 = warp_reduce_sum<32>(acc2);
    
    if (tx == 0)
        y[(long long)m * N + n] = __float2bfloat16_rn(acc ) * __float2bfloat16_rn(silu(acc2));
}
void matmul(Tensor& y, Tensor& x, const Tensor& W1, const Tensor& W2) {
    int K = W1.shape[1];
    int N = W1.shape[0];
    // std::cout<<y.shape[0]*y.shape[1]<<"x"<<y.shape[2]<<" "<<x.shape[0]*x.shape[1]<<"x"<<x.shape[2]<<" "<<W.shape[0]<<"x"<<W.shape[1]<<" "<<W.shape_scale[0]<<"x"<<W.shape_scale[1]<<std::endl;
    int M = x.shape[0] * x.shape[1];
    // if (N != K) std::printf("[gemv] W=%dx%d scale=%dx%d\n",
    //     N, K, W.shape_scale[0], W.shape_scale[1]);
    dim3 block(32, kGemvBN);
    dim3 grid((N + kGemvBN - 1) / kGemvBN, M);
    if(y.dtype() == CUDA_R_16BF){
        fp8_grouped_groupwise_bf16_kernel<<<grid, block, 0, get_compute_stream()>>>(
            (__nv_bfloat16*)y.data(),
            (const __nv_fp8_e4m3*)W1.data(), W1.scale(),
            (const __nv_fp8_e4m3*)W2.data(), W2.scale(),
            (const __nv_bfloat16*)x.data(),
            N, K, M, W1.shape_scale[0], W1.shape_scale[1]);
    } else {
        std::cerr << "[fp8_gemv] unsupported output dtype " << (int)y.dtype() << " — no kernel launched!\n";
        std::exit(1);
    }
}

// ── Batched expert execution ──────────────────────────────────────────────
// All active experts dispatched in a single kernel call per projection.
// grid: (ceil(N/kGemvBN), max_n_e, num_active)   block: (32, kGemvBN)
//
// Fused gate+up SwiGLU — W1=up_proj, W2=gate_proj, y = up * silu(gate)
__global__ void fp8_batched_swiglu_gemv_kernel(
        __nv_bfloat16* __restrict__ y,
        void* const*  __restrict__ W1_ptrs,
        float* const* __restrict__ W1s_ptrs,
        void* const*  __restrict__ W2_ptrs,
        float* const* __restrict__ W2s_ptrs,
        const __nv_bfloat16* __restrict__ x,
        const uint32_t* __restrict__ offsets,
        const int* __restrict__ active,
        int N, int K, int scale_N, int scale_K)
{
    const int ae    = blockIdx.z;
    const int e     = active[ae];
    const uint32_t start = offsets[e];
    const uint32_t n_e   = offsets[e + 1] - start;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n  = blockIdx.x * kGemvBN + ty;
    const int m  = blockIdx.y;

    if (n >= N || m >= (int)n_e) return;

    const __nv_fp8_e4m3* w1_row = (const __nv_fp8_e4m3*)W1_ptrs[ae] + (long long)n * K;
    const __nv_fp8_e4m3* w2_row = (const __nv_fp8_e4m3*)W2_ptrs[ae] + (long long)n * K;
    const float*          W1s   = W1s_ptrs[ae];
    const float*          W2s   = W2s_ptrs[ae];
    const __nv_bfloat16* x_row  = x + (long long)(start + m) * K;

    const int s_col = n * scale_N / N;

    float acc1 = 0.f, acc2 = 0.f;
    for (int kg = 0; kg < scale_K; kg++) {
        const int k = kg * kGemvSK + tx * 4;
        float     s1 = __ldg(W1s + s_col * scale_K + kg);
        float     s2 = __ldg(W2s + s_col * scale_K + kg);
        float     p1 = 0.f, p2 = 0.f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 wf1 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w1_row + k);
            __nv_fp8x4_e4m3 wf2 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w2_row + k);
            __nv_bfloat162 x01  = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 x23  = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);
            float4 fw1 = float4(wf1), fw2 = float4(wf2);
            float2 fx01 = __bfloat1622float2(x01), fx23 = __bfloat1622float2(x23);
            p1 = fmaf(fw1.x,fx01.x, fmaf(fw1.y,fx01.y, fmaf(fw1.z,fx23.x, fw1.w*fx23.y)));
            p2 = fmaf(fw2.x,fx01.x, fmaf(fw2.y,fx01.y, fmaf(fw2.z,fx23.x, fw2.w*fx23.y)));
        } else {
            for (int kk = k; kk < K && kk < k + 4; kk++) {
                float xv = __bfloat162float(x_row[kk]);
                p1 = fmaf(float(w1_row[kk]), xv, p1);
                p2 = fmaf(float(w2_row[kk]), xv, p2);
            }
        }
        acc1 = fmaf(p1, s1, acc1);
        acc2 = fmaf(p2, s2, acc2);
    }
    acc1 = warp_reduce_sum<32>(acc1);
    acc2 = warp_reduce_sum<32>(acc2);

    if (tx == 0)
        y[(long long)(start + m) * N + n] = __float2bfloat16_rn(acc1 * silu(acc2));
}

// Down projection for all active experts
__global__ void fp8_batched_down_gemv_kernel(
        __nv_bfloat16* __restrict__ y,
        void* const*  __restrict__ Wd_ptrs,
        float* const* __restrict__ Wds_ptrs,
        const __nv_bfloat16* __restrict__ x,
        const uint32_t* __restrict__ offsets,
        const int* __restrict__ active,
        int N, int K, int scale_N, int scale_K)
{
    const int ae    = blockIdx.z;
    const int e     = active[ae];
    const uint32_t start = offsets[e];
    const uint32_t n_e   = offsets[e + 1] - start;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n  = blockIdx.x * kGemvBN + ty;
    const int m  = blockIdx.y;

    if (n >= N || m >= (int)n_e) return;

    const __nv_fp8_e4m3* wd_row = (const __nv_fp8_e4m3*)Wd_ptrs[ae] + (long long)n * K;
    const float*          Wds   = Wds_ptrs[ae];
    const __nv_bfloat16* x_row  = x + (long long)(start + m) * K;

    const int s_col = n * scale_N / N;

    float acc = 0.f;
    for (int kg = 0; kg < scale_K; kg++) {
        float scale = __ldg(Wds + s_col * scale_K + kg);
        const int k = kg * kGemvSK + tx * 4;
        float partial = 0.f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(wd_row + k);
            __nv_bfloat162 x01 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 x23 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);
            float4 wf = float4(w4);
            float2 xf01 = __bfloat1622float2(x01), xf23 = __bfloat1622float2(x23);
            partial = fmaf(wf.x,xf01.x, fmaf(wf.y,xf01.y, fmaf(wf.z,xf23.x, wf.w*xf23.y)));
        } else {
            for (int kk = k; kk < K && kk < k + 4; kk++)
                partial = fmaf(float(wd_row[kk]), __bfloat162float(x_row[kk]), partial);
        }
        acc = fmaf(partial, scale, acc);
    }
    acc = warp_reduce_sum<32>(acc);

    if (tx == 0)
        y[(long long)(start + m) * N + n] = __float2bfloat16_rn(acc);
}

void moe_batched_expert_forward(
        Tensor& gathered,    Tensor& intermediate,
        void**  d_W1,  float** d_W1s,
        void**  d_W2,  float** d_W2s,
        void**  d_Wd,  float** d_Wds,
        const Tensor& expert_offsets,
        int* d_active, int num_active, int max_n_e,
        int D, int moe_int,
        int swiglu_sN, int swiglu_sK,
        int down_sN,   int down_sK)
{
    if (num_active == 0) return;
    dim3 block(32, kGemvBN);
    cudaStream_t stream = get_compute_stream();

    dim3 swiglu_grid((moe_int + kGemvBN - 1) / kGemvBN, max_n_e, num_active);
    fp8_batched_swiglu_gemv_kernel<<<swiglu_grid, block, 0, stream>>>(
        (__nv_bfloat16*)intermediate.data(),
        d_W1, d_W1s, d_W2, d_W2s,
        (const __nv_bfloat16*)gathered.data(),
        (const uint32_t*)expert_offsets.data(),
        d_active, moe_int, D, swiglu_sN, swiglu_sK);

    dim3 down_grid((D + kGemvBN - 1) / kGemvBN, max_n_e, num_active);
    fp8_batched_down_gemv_kernel<<<down_grid, block, 0, stream>>>(
        (__nv_bfloat16*)gathered.data(),
        d_Wd, d_Wds,
        (const __nv_bfloat16*)intermediate.data(),
        (const uint32_t*)expert_offsets.data(),
        d_active, D, moe_int, down_sN, down_sK);
}

#endif
