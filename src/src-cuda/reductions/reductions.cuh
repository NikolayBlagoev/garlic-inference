#pragma once
#include "cuda-common.cuh"



void reduce(Tensor& inp, Tensor& outp, int dim, bool norm);

void mean_t(Tensor& inp, Tensor& outp, int dim){
    return reduce(inp, outp, dim, true);
}

void sum_t(Tensor& inp, Tensor& outp, int dim){
    return reduce(inp, outp, dim, false);
}

void max_t(const Tensor& inp, float* outp);