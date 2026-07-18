#include "quant.h"


template<typename T>
__device__ inline void store4(T* p, float a, float b, float c, float d) {
    p[0] = T(a); p[1] = T(b); p[2] = T(c); p[3] = T(d);
}
template<>
__device__ inline void store4<float>(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}
template<>
__device__ inline void store4<__nv_bfloat16>(__nv_bfloat16* p, float a, float b, float c, float d) {
    reinterpret_cast<__nv_bfloat162*>(p)[0] = __nv_bfloat162(__nv_bfloat16(a), __nv_bfloat16(b));
    reinterpret_cast<__nv_bfloat162*>(p)[1] = __nv_bfloat162(__nv_bfloat16(c), __nv_bfloat16(d));
}
template<>
__device__ inline void store4<__half>(__half* p, float a, float b, float c, float d) {
    reinterpret_cast<__half2*>(p)[0] = __half2(__half(a), __half(b));
    reinterpret_cast<__half2*>(p)[1] = __half2(__half(c), __half(d));
}


template<typename T>
__global__ void dequant_fp8_blockscale_kernel(
    T* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ in,
    const float* __restrict__ scales,
    int rows, int cols, int scale_rows, int scale_cols)
{
    int row = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int col =  blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;

    int s_row = row * scale_rows / rows;
    int s_col = col * scale_cols / cols;
    float sc = __ldg(scales + s_col * scale_rows + s_row);

    long long base = (long long)col * rows + row;

    if (row + 3 < rows) {
        uint32_t raw = *reinterpret_cast<const uint32_t*>(in + base);
        __nv_fp8_e4m3 fp8[4];
        fp8[0].__x =  raw & 0xFF;
        fp8[1].__x = (raw >> 8) & 0xFF;
        fp8[2].__x = (raw >> 16) & 0xFF;
        fp8[3].__x = (raw >> 24) & 0xFF;
        store4<T>(out + base,
               float(fp8[0]) * sc,
               float(fp8[1]) * sc,
               float(fp8[2]) * sc,
               float(fp8[3]) * sc);
    } else {
        for (int i = 0; row + i < rows; i++)
            out[base + i] = T(float(in[base + i]) * sc);
    }
}


void dequant_fp8_blockscale(Tensor& out, const Tensor& in){
    int rows      = in.shape[1];
    int columns   = in.shape[0];
    int scale_rows = in.shape_scale[1];
    int scale_cols = in.shape_scale[0];

    dim3 threads(32, 8);
    
    dim3 blocks((rows + 127) / 128, (columns + 7) / 8);

    if(out.dtype() == CUDA_R_32F){
        dequant_fp8_blockscale_kernel<float><<<blocks, threads, 0, get_compute_stream()>>>(
            (float*)out.data(),
            (const __nv_fp8_e4m3*)in.data(),
            in.scale(),
            rows, columns, scale_rows, scale_cols);
    }else if(out.dtype() == CUDA_R_16BF){
        dequant_fp8_blockscale_kernel<__nv_bfloat16><<<blocks, threads, 0, get_compute_stream()>>>(
            (__nv_bfloat16*)out.data(),
            (const __nv_fp8_e4m3*)in.data(),
            in.scale(),
            rows, columns, scale_rows, scale_cols);
    }else if(out.dtype() == CUDA_R_16F){
        dequant_fp8_blockscale_kernel<__half><<<blocks, threads, 0, get_compute_stream()>>>(
            (__half*)out.data(),
            (const __nv_fp8_e4m3*)in.data(),
            in.scale(),
            rows, columns, scale_rows, scale_cols);
    }
}

