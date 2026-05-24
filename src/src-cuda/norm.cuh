#pragma once
#include "cuda-common.cuh"


void rmsnorm(Tensor& x, const Tensor& weight, float eps = 1e-6f, int dim_size = -1);
