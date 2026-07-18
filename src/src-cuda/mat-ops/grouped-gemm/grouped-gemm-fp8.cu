// TODO: Update with async preload

#include "mat-ops.cuh"
#include "activations.cuh"

#ifdef FP8_AVAILABLE
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


__global__ void fp8_grouped_groupwise_f16_kernel(
        __half* __restrict__ y,
        const __nv_fp8_e4m3* __restrict__ W1,
        const float* __restrict__ W1_scale,
        const __nv_fp8_e4m3* __restrict__ W2,
        const float* __restrict__ W2_scale,
        const __half* __restrict__ x,
        int N, int K, int M, int scale_N, int scale_K)
{
    const int tx = threadIdx.x & 31;           // lane: 0..31
    const int ty = threadIdx.y;           // row slot: 0..kGemvBN-1
    const int n  = blockIdx.x * kGemvBN + ty;
    const int m  = blockIdx.y;

    if (n >= N || m >= M) return;

    const __nv_fp8_e4m3* w1_row = W1 + (long long)n * K;
    const __nv_fp8_e4m3* w2_row = W2 + (long long)n * K;
    const __half* x_row = x + (long long)m * K;

    int s_col = n * scale_N / N;
    float factor = K / scale_K;

    float acc = 0.0f;

    for (int kg = 0; kg < scale_K; kg++) {
        float scale = __ldg(W1_scale + s_col * scale_K + kg);
        const int k = kg * kGemvSK + tx * 4;
        float partial = 0.0f;
        if (k + 3 < K) {
            __nv_fp8x4_e4m3 w4 = *reinterpret_cast<const __nv_fp8x4_e4m3*>(w1_row + k);
            
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
                partial = fmaf(float(w1_row[kk]), __half2float(x_row[kk]), partial);
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
                partial = fmaf(float(w2_row[kk]), __half2float(x_row[kk]), partial);
        }
        acc2 = fmaf(partial,scale,acc2);
    }


    acc2 = warp_reduce_sum<32>(acc2);
    
    if (tx == 0)
        y[(long long)m * N + n] = __float2half_rn(acc ) * __float2half_rn(silu(acc2));
}
void matmul(Tensor& y, Tensor& x, const Tensor& W1, const Tensor& W2, cudaStream_t compute_stream) {
    if(compute_stream == nullptr) compute_stream = get_compute_stream();
    int K = W1.shape[1];
    int N = W1.shape[0];
    // std::cout<<y.shape[0]*y.shape[1]<<"x"<<y.shape[2]<<" "<<x.shape[0]*x.shape[1]<<"x"<<x.shape[2]<<" "<<W.shape[0]<<"x"<<W.shape[1]<<" "<<W.shape_scale[0]<<"x"<<W.shape_scale[1]<<std::endl;
    int M = x.shape[0] * x.shape[1];
    // if (N != K) std::printf("[gemv] W=%dx%d scale=%dx%d\n",
    //     N, K, W.shape_scale[0], W.shape_scale[1]);
    dim3 block(32, kGemvBN);
    dim3 grid((N + kGemvBN - 1) / kGemvBN, M);
    if(y.dtype() == CUDA_R_16BF){
        fp8_grouped_groupwise_bf16_kernel<<<grid, block, 0, compute_stream>>>(
            (__nv_bfloat16*)y.data(),
            (const __nv_fp8_e4m3*)W1.data(), W1.scale(),
            (const __nv_fp8_e4m3*)W2.data(), W2.scale(),
            (const __nv_bfloat16*)x.data(),
            N, K, M, W1.shape_scale[0], W1.shape_scale[1]);
    } else if(y.dtype() == CUDA_R_16F){
        fp8_grouped_groupwise_f16_kernel<<<grid, block, 0, compute_stream>>>(
            (__half*)y.data(),
            (const __nv_fp8_e4m3*)W1.data(), W1.scale(),
            (const __nv_fp8_e4m3*)W2.data(), W2.scale(),
            (const __half*)x.data(),
            N, K, M, W1.shape_scale[0], W1.shape_scale[1]);
    } else {
        std::cerr << "[fp8_gemv] unsupported output dtype " << (int)y.dtype() << " — no kernel launched!\n";
        std::exit(1);
    }
}
#endif