#pragma once
#include "cuda-common.cuh"

#include "quant.h"


// #include "cutlass_runner.cu"
static Tensor dequant_buffer;
static int dequant_buffer_nelems = 0;
// TODO: these hardcore the scale granularity... fix that
static constexpr int kGemvBN = 8;   // output rows per CTA (one warp each)
static constexpr int kGemvSK = 128; // K scale granularity


void add_inplace(Tensor& x, Tensor& y, cudaStream_t compute_stream = nullptr);

void elm_wise(Tensor& x, const Tensor& w, cudaStream_t compute_stream = nullptr);


void matmul_cublas(Tensor& out, Tensor& A, const Tensor& B, cudaStream_t compute_stream = nullptr);
#ifdef FP8_AVAILABLE
void matmul_fp8_blockscale(Tensor& y, Tensor& x, const Tensor& W, cudaStream_t compute_stream = nullptr);
void matmul(Tensor& y, Tensor& x, const Tensor& W1, const Tensor& W2, cudaStream_t compute_stream = nullptr);
#endif


// W is column major ;)
// [batch, M, N] x [K, N] -> [batch, M, K]
// B is shared across the batch (stride = 0).
void matmul(Tensor& out, Tensor& A, const Tensor& B, cudaStream_t compute_stream = nullptr);