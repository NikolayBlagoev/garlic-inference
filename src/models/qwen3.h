#pragma once
#include "tensor.h"
#include <string>
#include <vector>


struct KVCache;
struct FlashAttnEngine;

struct Qwen3Config {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    int head_dim;

    float rms_norm_eps;
    float rope_theta;

    static Qwen3Config from_pretrained(const std::string& hf_dir);
};

struct Qwen3MLP {
    int intermediate_size;
    Tensor gate_proj, up_proj, down_proj;
    Tensor forward(const Tensor& hidden,
                Tensor& gate,
                Tensor& up,
                Tensor& down);
};
struct Qwen3Attention {
    int head_dim, num_heads, num_kv_heads;
    float rms_norm_eps;
    Tensor q_proj, k_proj, v_proj, o_proj;
    Tensor q_norm, k_norm;
    Tensor forward(Tensor& hidden,
                const Tensor& cos_emb, const Tensor& sin_emb,
                std::vector<KVCache>& kvcache, FlashAttnEngine& engine,
                int idx,
                Tensor& Q,
                Tensor& K,
                Tensor& V,
                Tensor& O,
                Tensor& output);
};
struct Qwen3DecoderLayer {
    Tensor input_layernorm;
    Tensor post_attention_layernorm;
    float rms_norm_eps;
    int idx;
    Qwen3Attention self_attn;
    Qwen3MLP mlp;
    

    Tensor forward(Tensor hidden,
                const Tensor& cos_emb, const Tensor& sin_emb,
                std::vector<KVCache>& kvcache, FlashAttnEngine& engine,
                Tensor& residual,
                Tensor& Q,
                Tensor& K,
                Tensor& V,
                Tensor& O,
                Tensor& output,
                Tensor& gate,
                Tensor& up,
                Tensor& down);
};

struct Qwen3RotaryEmbedding {
    float attention_scaling = 1.0f;
    Tensor inv_freq;
};

struct Qwen3 {
    Qwen3Config config;
    Tensor embed_tokens; 
    std::vector<Qwen3DecoderLayer> layers;
    Tensor norm;
    Tensor lm_head;
    Qwen3RotaryEmbedding rotary_embedding;
    Tensor residual;
    
    static Qwen3 from_pretrained(const std::string& hf_dir, int device = 0);


    Tensor forward(const Tensor& input_ids, Tensor& position_ids,
                    std::vector<KVCache>& kvcache,
                    FlashAttnEngine& engine);
};
