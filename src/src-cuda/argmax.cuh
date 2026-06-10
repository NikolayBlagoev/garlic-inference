#pragma once
#include "cuda-common.cuh"


std::vector<uint32_t> argmax(const Tensor& x, Tensor& out, int dim = -1);

void topk(const Tensor& x, Tensor& ids, Tensor& weights, int K);