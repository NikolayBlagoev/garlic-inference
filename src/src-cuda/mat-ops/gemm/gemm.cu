#include "mat-ops.cuh"

// torch does column major it seems
void matmul_cublas(Tensor& y, Tensor& x, const Tensor& W, cudaStream_t compute_stream) {
    int x_dims = x.ndim();
    if(compute_stream == nullptr){
        compute_stream = get_compute_stream();
    }

    int B = x.shape[0];
    int M = x.shape[1];
    int N = x.shape[2]; // in_features
    int K = W.shape[0]; // out_features

    // TODO: spitting facts below
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