#include "mat-ops.cuh"

void matmul(Tensor& out, Tensor& A, const Tensor& B, cudaStream_t compute_stream){
    int x_dims = A.ndim();
    if(compute_stream == nullptr){
        compute_stream = get_compute_stream();
    }
    // static_assert(W.shape[1] == x.shape[x_dims - 1]);
#ifdef FP8_AVAILABLE
    
    if (B.dtype() == CUDA_R_8F_E4M3 && B.ndim_scale() == 2) {
        if(A.num_elements() / A.shape[x_dims - 1] >= 0){
            return matmul_fp8_blockscale(out, A, B, compute_stream);
        }
    }
#endif
    return matmul_cublas(out, A, B, compute_stream);

}