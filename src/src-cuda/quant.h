#pragma once
#include "tensor.h"



void dequant_fp8_blockscale(Tensor& out, const Tensor& in);

// M_padded: allocate scale with this many rows (>= M); extra rows zeroed once on (re)allocation.
void per_token_fp8_quantize(Tensor& out, const Tensor& in, int M_padded = 0, int kBlockK = 128);