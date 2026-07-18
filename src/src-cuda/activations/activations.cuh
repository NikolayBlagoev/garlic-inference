#pragma once
#include "cuda-common.cuh"

__device__ __forceinline__ float silu(float x) {
    return x / (1 + expf(-x));
}

void silu(Tensor& x);