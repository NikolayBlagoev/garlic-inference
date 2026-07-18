#pragma once
#include "cuda-common.cuh"
// void zero_inplace(int M, int N, float* A);

void fill_inplace(Tensor& t, float val = 0.0);

void arrange(Tensor& t, int a, int step);

void pow(Tensor& inp, float val);



Tensor transpose(const Tensor& inp, int d0, int d1);

