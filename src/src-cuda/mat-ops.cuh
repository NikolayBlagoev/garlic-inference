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

void add_inplace(Tensor& x, Tensor& y);

void elm_wise(Tensor& x, const Tensor& w);

#ifdef FP8_AVAILABLE
// both x and W need to be float8
// W is column major ;)
// [batch, M, N] x [K, N] -> [batch, M, K]
// B is shared across the batch (stride = 0).
// void matmul_float8(Tensor& y, Tensor& x, const Tensor& W);

// x is float32, W is float8 with 2-D block scales stored in W.scale()
// [batch, M, N] x [K, N] -> [batch, M, K]; output y can be F32 or BF16
void matmul_fp8_blockscale(Tensor& y, Tensor& x, const Tensor& W);
void matmul(Tensor& y, Tensor& x, const Tensor& W1, const Tensor& W2);

// All active MoE experts: fused SwiGLU then down-proj, both batched over experts.
// gathered:     [T*K, D]      BF16 — input (read) and output (written by down proj)
// intermediate: [T*K, moe_int] BF16 — scratch for SwiGLU output
// d_W{1,2,d}_ptrs / d_W{1,2,d}s_ptrs: device arrays of [num_active] pointers
// d_active: device array of active expert indices [num_active]
void moe_batched_expert_forward(
        Tensor& gathered,    Tensor& intermediate,
        void**  d_W1,  float** d_W1s,
        void**  d_W2,  float** d_W2s,
        void**  d_Wd,  float** d_Wds,
        const Tensor& expert_offsets,
        int* d_active, int num_active, int max_n_e,
        int D, int moe_int,
        int swiglu_sN, int swiglu_sK,
        int down_sN,   int down_sK);
#endif
