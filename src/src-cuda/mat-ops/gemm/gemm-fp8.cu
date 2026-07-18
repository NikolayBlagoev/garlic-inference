#include "mat-ops.cuh"

#ifdef FP8_AVAILABLE
#include <cuda_pipeline.h>



__global__ void fp8_gemv_f16_kernel(
    __half* __restrict__ y,
    const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ W_scale,
    const __half* __restrict__ x,
    int N, int K, int M, int scale_N, int scale_K){
    
    // keep 2 buffers in memory -> one for weights
    //                          -> one for x
    // parallelize loading of buffer 0 while we compute on buffer 1
    __shared__ __align__(16) uint8_t sW[2][kGemvBN][kGemvSK]; // kGemvBN = number of warps
    __shared__ __align__(16) __half sX[2][kGemvSK];

    const int lane = threadIdx.x & 31; // equivalent to % 32
    
    const int warp_id = threadIdx.y; // since blocdimx is 32, threadix.y is the warp

    const int n = blockIdx.x * kGemvBN + warp_id;
    const int m = blockIdx.y;

    const bool row_ok = (n < N) && (m < M);

    const __nv_fp8_e4m3* w_row = W + n * K;
    const __half* x_row = x + m * K;
    

    // lambda function to stage the next buffer 
    auto stage = [&](int buf, int kg) {
        const int kbase = kg * kGemvSK;
        // each lane stages 16 bytes... since we are working with 128
        // sized blocks, we use only the first 8 lanes
        if (row_ok && lane < 8){
            __pipeline_memcpy_async(&sW[buf][warp_id][lane * 16],
                                    w_row + kbase + lane * 16, 16);
        }
        // for x stage 256 bytes so first 16 lanes
        if (warp_id == 0 && lane < 16){
            __pipeline_memcpy_async(&sX[buf][lane * 8],
                                    x_row + kbase + lane * 8, 16);
        }
            
    };

    
    stage(0, 0);
    __pipeline_commit();

    // we use two accumulators (seems to produce more stable results)
    float acc = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        const int cur = kg & 1; // which buffer we read

        // stage the next buffer
        if (kg + 1 < scale_K) stage(cur ^ 1, kg + 1);
        __pipeline_commit(); // commit nothing last time

        // wait for the previous stage
        __pipeline_wait_prior(1);
        __syncthreads(); // need to sync because of warp_id == 0

        if(row_ok){
            // read from texture cache
            float scale = __ldg(W_scale + (n * scale_N / N) * scale_K + kg);

            // second accumulator
            float partial = 0.0f;
            
            const int k4 = lane * 4;
            // read 4 at a time
            __nv_fp8x4_e4m3 w4 =
                *reinterpret_cast<const __nv_fp8x4_e4m3*>(&sW[cur][warp_id][k4]);
            __half2 x01 =
                *reinterpret_cast<const __half2*>(&sX[cur][k4]);
            __half2 x23 =
                *reinterpret_cast<const __half2*>(&sX[cur][k4 + 2]);
            float4 wf  = float4(w4);
            float2 xf01 = __half22float2(x01);
            float2 xf23 = __half22float2(x23);
            partial = fmaf(wf.x, xf01.x, partial);
            partial = fmaf(wf.y, xf01.y, partial);
            partial = fmaf(wf.z, xf23.x, partial);
            partial = fmaf(wf.w, xf23.y, partial);
            acc = fmaf(partial, scale, acc);
        }
        
        // sync to ensure we do NOT overwrite our sX buffer by a fast warp
        // within warm we are safe..
        // no need to do it last time
        if(kg + 1 < scale_K) __syncthreads();
    }

    acc = warp_reduce_sum<32>(acc);
    // only first lane writes:
    if (lane == 0 && row_ok)
        y[(long long)m * N + n] = __float2half_rn(acc);
}

__global__ void fp8_gemv_bf16_kernel(
    __nv_bfloat16* __restrict__ y,
    const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ W_scale,
    const __nv_bfloat16* __restrict__ x,
    int N, int K, int M, int scale_N, int scale_K){
    
    // keep 2 buffers in memory -> one for weights
    //                          -> one for x
    // parallelize loading of buffer 0 while we compute on buffer 1
    __shared__ __align__(16) uint8_t sW[2][kGemvBN][kGemvSK]; // kGemvBN = number of warps
    __shared__ __align__(16) __nv_bfloat16 sX[2][kGemvSK];

    const int lane = threadIdx.x & 31; // equivalent to % 32
    
    const int warp_id = threadIdx.y; // since blocdimx is 32, threadix.y is the warp

    const int n = blockIdx.x * kGemvBN + warp_id;
    const int m = blockIdx.y;

    const bool row_ok = (n < N) && (m < M);

    const __nv_fp8_e4m3* w_row = W + n * K;
    const __nv_bfloat16* x_row = x + m * K;
    

    // lambda function to stage the next buffer 
    auto stage = [&](int buf, int kg) {
        const int kbase = kg * kGemvSK;
        // each lane stages 16 bytes... since we are working with 128
        // sized blocks, we use only the first 8 lanes
        if (row_ok && lane < 8){
            __pipeline_memcpy_async(&sW[buf][warp_id][lane * 16],
                                    w_row + kbase + lane * 16, 16);
        }
        // for x stage 256 bytes so first 16 lanes
        if (warp_id == 0 && lane < 16){
            __pipeline_memcpy_async(&sX[buf][lane * 8],
                                    x_row + kbase + lane * 8, 16);
        }
            
    };

    
    stage(0, 0);
    __pipeline_commit();

    // we use two accumulators (seems to produce more stable results)
    float acc = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        const int cur = kg & 1; // which buffer we read

        // stage the next buffer
        if (kg + 1 < scale_K) stage(cur ^ 1, kg + 1);
        __pipeline_commit(); // commit nothing last time

        // wait for the previous stage
        __pipeline_wait_prior(1);
        __syncthreads(); // need to sync because of warp_id == 0

        if(row_ok){
            // read from texture cache
            float scale = __ldg(W_scale + (n * scale_N / N) * scale_K + kg);

            // second accumulator
            float partial = 0.0f;
            
            const int k4 = lane * 4;
            // read 4 at a time
            __nv_fp8x4_e4m3 w4 =
                *reinterpret_cast<const __nv_fp8x4_e4m3*>(&sW[cur][warp_id][k4]);
            __nv_bfloat162 x01 =
                *reinterpret_cast<const __nv_bfloat162*>(&sX[cur][k4]);
            __nv_bfloat162 x23 =
                *reinterpret_cast<const __nv_bfloat162*>(&sX[cur][k4 + 2]);
            float4 wf  = float4(w4);
            float2 xf01 = __bfloat1622float2(x01);
            float2 xf23 = __bfloat1622float2(x23);
            partial = fmaf(wf.x, xf01.x, partial);
            partial = fmaf(wf.y, xf01.y, partial);
            partial = fmaf(wf.z, xf23.x, partial);
            partial = fmaf(wf.w, xf23.y, partial);
            acc = fmaf(partial, scale, acc);
        }
        
        // sync to ensure we do NOT overwrite our sX buffer by a fast warp
        // within warm we are safe..
        // no need to do it last time
        if(kg + 1 < scale_K) __syncthreads();
    }

    acc = warp_reduce_sum<32>(acc);
    // only first lane writes:
    if (lane == 0 && row_ok)
        y[(long long)m * N + n] = __float2bfloat16_rn(acc);
}

void matmul_fp8_blockscale(Tensor& y, Tensor& x, const Tensor& W, cudaStream_t compute_stream) {
    int K = W.shape[1];
    int N = W.shape[0];
    
    int M = x.shape[0] * x.shape[1];


    dim3 block(32, kGemvBN);
    dim3 grid((N + kGemvBN - 1) / kGemvBN, M);
    if(y.dtype() == CUDA_R_16BF){
        fp8_gemv_bf16_kernel<<<grid, block, 0, compute_stream>>>(
            (__nv_bfloat16*)y.data(),
            (const __nv_fp8_e4m3*)W.data(), W.scale(),
            (const __nv_bfloat16*)x.data(),
            N, K, M, W.shape_scale[0], W.shape_scale[1]);
    }else if(y.dtype() == CUDA_R_16F){
        fp8_gemv_f16_kernel<<<grid, block, 0, compute_stream>>>(
            (__half*)y.data(),
            (const __nv_fp8_e4m3*)W.data(), W.scale(),
            (const __half*)x.data(),
            N, K, M, W.shape_scale[0], W.shape_scale[1]);
    }
}






// static void matmul_fp8_blockscale_dequant(Tensor& y, Tensor& x, const Tensor& W) {
//     int n = W.num_elements();
//     if (n > dequant_buffer_nelems || !dequant_buffer._data || dequant_buffer.dtype() != x.dtype()) {
//         dequant_buffer = Tensor(W.shape, x.dtype(), W.device());
//         dequant_buffer_nelems = n;
//     }
//     dequant_buffer.shape = W.shape;

//     dequant_fp8_blockscale(dequant_buffer, W);
//     matmul(y, x, dequant_buffer);
// }
#endif