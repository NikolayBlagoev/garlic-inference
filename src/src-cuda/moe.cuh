#pragma once
#include "cuda-common.cuh"


// Gather tokens into expert-sorted contiguous layout.
// x:               [T, D]   input tokens (float32 or bfloat16)
// expert_ids:      [T, K]   int32  (from moe_topk)
// gathered:        [T*K, D] output tokens sorted by expert (same dtype as x)
// token_map:       [T*K]    int32 — gathered slot s came from token token_map[s]
// slot_map:        [T*K]    int32 — slot_map[t*K+k] = slot index for (token t, expert k)
// expert_offsets:  [E+1]    int32 — expert_offsets[e] is the first slot for expert e
void moe_gather(const Tensor& x,
                const int32_t* expert_ids,
                Tensor&        gathered,
                int32_t*       token_map,
                int32_t*       slot_map,
                int32_t*       expert_offsets,
                int K, int num_experts);

// Weighted combination of expert outputs back into token space.
// expert_out:     [T*K, D] expert computation results (float32 or bfloat16)
// expert_weights: [T, K]   float32 routing weights (from moe_topk)
// slot_map:       [T*K]    int32 (from moe_gather)
// out:            [T, D]   accumulated output (overwritten)
void moe_scatter(const Tensor& expert_out,
                 const float*   expert_weights,
                 const int32_t* slot_map,
                 Tensor&        out,
                 int T, int K);
