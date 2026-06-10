#pragma once
#include "tensor.h"
#include <string>
#include <vector>
#include "qwen3.h"

struct KVCache;
struct FlashAttnEngine;

//TODO inherit from Qwen3Config?
struct Qwen3MoeConfig {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    int head_dim;

    //MoeSpecific
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate_size;

    float rms_norm_eps;
    float rope_theta;

    //TODO: some qwen3moes have mlp only layers (implemented as Qwen3MLP)

    static Qwen3MoeConfig from_pretrained(const std::string& hf_dir);
};

struct Qwen3MoeSparseMoeBlock {
    int moe_intermediate_size;
    std::vector<Tensor> gate_proj, up_proj, down_proj;
    Tensor forward(const Tensor& hidden);
};

struct Qwen3MoeDecoderLayer {
    Tensor input_layernorm;
    Tensor post_attention_layernorm;
    float rms_norm_eps;
    int idx;
    Qwen3Attention self_attn;
    Qwen3MoeSparseMoeBlock mlp;
    

    Tensor forward(Tensor hidden,
                const Tensor& position_ids, const Tensor& inv_freq,
                std::vector<KVCache>& kvcache, FlashAttnEngine& engine);
};


struct Qwen3Moe {
    Qwen3MoeConfig config;
    Tensor embed_tokens; 
    std::vector<Qwen3MoeDecoderLayer> layers;
    Tensor norm;
    Tensor lm_head;
    Qwen3RotaryEmbedding rotary_embedding;
    Tensor residual;
    
    static Qwen3Moe from_pretrained(const std::string& hf_dir, int device = 0);


    Tensor forward(const Tensor& input_ids, Tensor& position_ids,
                    std::vector<KVCache>& kvcache,
                    FlashAttnEngine& engine);
};
