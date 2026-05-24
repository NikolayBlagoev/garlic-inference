#pragma once
#include "cuda-common.cuh"


void embedding_forward(Tensor& input_embeds, const Tensor& ids, const Tensor& weight);
