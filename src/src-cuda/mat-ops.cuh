#pragma once
#include "cuda-common.cuh"

// W is column major ;)
// [batch, M, N] x [K, N] -> [batch, M, K]
// B is shared across the batch (stride = 0).
void matmul(Tensor& out, Tensor& A, const Tensor& B);

void add_inplace(Tensor& x, Tensor& y);

void elm_wise(Tensor& x, const Tensor& w);

#ifdef FP8_AVAILABLE
// both x and W need to be float8
// W is column major ;)
// [batch, M, N] x [K, N] -> [batch, M, K]
// B is shared across the batch (stride = 0).
void matmul_float8(Tensor& y, Tensor& x, const Tensor& W);
#endif
