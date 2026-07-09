#pragma once
#include "cuda-common.cuh"
#include "quant.h"
// #include "cutlass_runner.cu"
static Tensor dequant_buffer;
static int dequant_buffer_nelems = 0;
// W is column major ;)
// [batch, M, N] x [K, N] -> [batch, M, K]
// B is shared across the batch (stride = 0).
void matmul(Tensor& out, Tensor& A, const Tensor& B, cudaStream_t compute_stream = nullptr);

void add_inplace(Tensor& x, Tensor& y, cudaStream_t compute_stream = nullptr);

void elm_wise(Tensor& x, const Tensor& w, cudaStream_t compute_stream = nullptr);

#ifdef FP8_AVAILABLE
// both x and W need to be float8
// W is column major ;)
// [batch, M, N] x [K, N] -> [batch, M, K]
// B is shared across the batch (stride = 0).
// void matmul_float8(Tensor& y, Tensor& x, const Tensor& W);

// x is float32, W is float8 with 2-D block scales stored in W.scale()
// [batch, M, N] x [K, N] -> [batch, M, K]; output y can be F32 or BF16
void matmul_fp8_blockscale(Tensor& y, Tensor& x, const Tensor& W, cudaStream_t compute_stream = nullptr);
void matmul(Tensor& y, Tensor& x, const Tensor& W1, const Tensor& W2, cudaStream_t compute_stream = nullptr);


#endif
