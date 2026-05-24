#pragma once
#include "cuda-common.cuh"
#include <utility>

void initialize_rope(float base, Tensor& t);

void rope_forward(
    Tensor& x,
    Tensor& position_ids,
    Tensor& inv_freq,
    Tensor& cos_emb,
    Tensor& sin_emb);


void apply_rotary_pos_emb(Tensor& x, const Tensor& cos, const Tensor& sin);