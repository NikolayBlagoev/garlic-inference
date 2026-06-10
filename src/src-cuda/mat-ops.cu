#include "cutlass-runner.cuh"
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


void elm_wise(Tensor& x, const Tensor& w) {
    const long long N = x.num_elements();
    const long long stride = w.num_elements();
    const int threads = 256;
    const int blocks = (N / 4 + threads - 1) / threads;
    if (x.dtype() == CUDA_R_32F && w.dtype() == CUDA_R_32F) {
        elm_wise_kernel_f32_f32<<<blocks, threads>>>(
            (float*)x.data(), (const float*)w.data(),
            stride, N);
    } else if (x.dtype() == CUDA_R_16BF && w.dtype() == CUDA_R_16BF) {
        elm_wise_kernel_bf16_bf16<<<blocks, threads>>>(
            (__nv_bfloat16*)x.data(), (const __nv_bfloat16*)w.data(),
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

//computes x = x + w
void add_inplace(Tensor& x, Tensor& w) {
const long long N = x.num_elements();
    const long long stride = w.num_elements();
    const int threads = 256;
    const int blocks = (N / 4 + threads - 1) / threads;
    if (x.dtype() == CUDA_R_32F && w.dtype() == CUDA_R_32F) {
        add_inplace_kernel_f32_f32<<<blocks, threads>>>(
            (float*)x.data(), (const float*)w.data(),
            stride, N);
    } else if (x.dtype() == CUDA_R_16BF && w.dtype() == CUDA_R_16BF) {
        add_inplace_kernel_bf16_bf16<<<blocks, threads>>>(
            (__nv_bfloat16*)x.data(), (const __nv_bfloat16*)w.data(),
            stride, N);
    }
}

// torch does column major it seems
void matmul(Tensor& y, Tensor& x, const Tensor& W, bool reuse) {
#ifdef FP8_AVAILABLE
    // FP8 weight with 2-D block scales: dequant to BF16 then GEMM.
    if (W.dtype() == CUDA_R_8F_E4M3 && W.ndim_scale() == 2) {
        matmul_fp8_blockscale(y, x, W, reuse);
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
    if(x.dtype() != W.dtype()){
        x = x.cast_to(W.dtype());
    }
    int B = x.shape[0];
    int M = x.shape[1];
    int N = x.shape[2]; // in_features
    int K = W.shape[0]; // out_features

    // unfortunately cublas needs these for y = alpha*W*x + beta*y... might be useful to make them into function params
    const float alpha = 1.0f;
    const float beta = 0.0f;


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
        const int k0 = kg * factor;
        const int   k_end = MIN(k0 + factor, K);
        float partial = 0.0f;
        for (int k = k0 + tx; k < k_end; k += 32){
            partial = fmaf(static_cast<float>(w_row[k]), __bfloat162float(x_row[k]), partial);

            
        }
        acc = fmaf(partial,scale,acc);
            
    }

    
    acc = warp_reduce_sum<32>(acc);
    
    if (tx == 0)
        y[(long long)m * N + n] = __float2bfloat16_rn(acc);
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

    fp8_gemv_groupwise_bf16_kernel<<<grid, block>>>(
        (__nv_bfloat16*)y.data(),
        (const __nv_fp8_e4m3*)W.data(), W.scale(),
        (const __nv_bfloat16*)x.data(),
        N, K, M, W.shape_scale[0], W.shape_scale[1]);
}

void matmul_fp8_blockscale(Tensor& y, Tensor& x, const Tensor& W, bool reuse) {
    int B = x.shape[0];
    int S = x.shape[1];
    int K = x.shape[2];   // in_features
    int N = W.shape[0];   // out_features
    int M = B * S;
    if(M > 0){
        // return matmul_fp8_blockscale_dequant(y,x,W);
        return fp8_gemv_groupwise_bf16(y,W,x);
    }
#ifdef USE_CUTLASS_FP8
    {   
        if(!reuse){
            auto saved_shape = x.shape;
            x.shape = {M, K};
            // Allocate M_padded rows so CUTLASS TMA doesn't read past the buffer end.
            int M_padded = (M + 7) & ~7;
            int n = x.num_elements() * M_padded / M;
            if (n > dequant_buffer_nelems || !dequant_buffer._data || dequant_buffer.dtype() != W.dtype()) {
                dequant_buffer = Tensor({M_padded, K}, W.dtype(), W.device());
                dequant_buffer_nelems = n;
            }
            dequant_buffer.shape = {M_padded, K};
            
            per_token_fp8_quantize(dequant_buffer, x, M_padded);
            CUDA_CHECK(cudaGetLastError());
            x.shape = saved_shape;
        }
        garlic::gemm_fp8_groupwise_bf16_sm120(y.data(), W.data(), W.scale(), dequant_buffer.data(), dequant_buffer.scale(), N, K, M, 1);
    }
#else
    matmul_fp8_blockscale_dequant(y,x,W);
#endif

}

// both x and W need to be float8 (or we lose the float8 cores optimizations...)
// Also, scale needs to be a single scalar, for tensor scales we need cutlass
// void matmul_float8(Tensor& y, Tensor& x, const Tensor& W) {
//     int B = x.shape[0];
//     int M = x.shape[1];
//     int N = x.shape[2]; 
//     int K = W.shape[0]; 

//     const float alpha = 1.0f;
//     const float beta  = 0.0f;

//     cublasLtMatmulDesc_t op;
//     cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);

//     // transpose W
//     cublasOperation_t transpose_op = CUBLAS_OP_T;
//     cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_op, sizeof(transpose_op));

//     // Scale pointers for FP8.
//     if(W.scale()) cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &(W.scale()), sizeof(float*));
    
//     if(x.scale()) cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &(x.scale()), sizeof(float*));

//     cublasLtMatrixLayout_t lw, lx, ly;
//     cublasLtMatrixLayoutCreate(&lw, CUDA_R_8F_E4M3, N, K, N);
//     cublasLtMatrixLayoutCreate(&lx, x.dtype(), N, M, N);
//     cublasLtMatrixLayoutCreate(&ly, y.dtype(), K, M, K);

//     // Batch attributes.
//     long long stride_W = 0; // W shared across batch
//     long long stride_x = (long long) M * N;
//     long long stride_y = (long long) M * K;
//     cublasLtMatrixLayoutSetAttribute(lw, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &B, sizeof(int));
//     cublasLtMatrixLayoutSetAttribute(lw, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_W, sizeof(long long));
//     cublasLtMatrixLayoutSetAttribute(lx, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &B, sizeof(int));
//     cublasLtMatrixLayoutSetAttribute(lx, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_x, sizeof(long long));
//     cublasLtMatrixLayoutSetAttribute(ly, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &B, sizeof(int));
//     cublasLtMatrixLayoutSetAttribute(ly, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_y, sizeof(long long));

//     cublasLtMatmul((cublasLtHandle_t)cublas_handle(), op,
//                     &alpha,
//                     W.data(), lw,
//                     x.data(), lx,
//                     &beta,
//                     y.data(), ly,
//                     y.data(), ly,
//                     nullptr, nullptr, 0, 0);
    
//     // prevent memory leaks lol... 
//     cublasLtMatrixLayoutDestroy(lw);
//     cublasLtMatrixLayoutDestroy(lx);
//     cublasLtMatrixLayoutDestroy(ly);
// }



#endif
