#pragma once
#include "tensor.h"
#include <string>
#include <vector>
#include "generic_model.h"
#include "../qwen3/qwen3.h"



struct KVCache;
struct FlashAttnEngine;


struct Qwen3MoeConfig : LanguageConfig {
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

struct Qwen3MoeDecoderLayer;

struct Qwen3MoeSparseMoeBlock {

    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate_size;
    int layer_idx;
    Tensor gate;
    std::vector<Tensor> gate_proj, up_proj, down_proj;

    // Batched expert execution — lazy-initialized on first forward pass
    void**  d_W1_ptrs    = nullptr;  // up_proj data pointers   [num_experts_per_tok]
    void**  d_W2_ptrs    = nullptr;  // gate_proj data pointers
    void**  d_Wd_ptrs    = nullptr;  // down_proj data pointers
    float** d_W1s_ptrs   = nullptr;  // up_proj scale pointers
    float** d_W2s_ptrs   = nullptr;  // gate_proj scale pointers
    float** d_Wds_ptrs   = nullptr;  // down_proj scale pointers
    int*    d_active     = nullptr;  // active expert indices [num_experts_per_tok]
    Tensor  intermediate;            // [total_slots, moe_intermediate_size] BF16
    int     intermediate_cap = 0;    // allocated capacity in token-slots

    Tensor forward(Tensor& hidden, Tensor& router_logits,
                Tensor& router_scores, Tensor& router_indices,
                Tensor& gathered, Tensor& token_map,
                Tensor& slot_map, Tensor& expert_offsets,
                std::vector<Qwen3MoeDecoderLayer>& layers);
};

struct Qwen3MoeDecoderLayer {
    Tensor input_layernorm;
    Tensor post_attention_layernorm;
    float rms_norm_eps;
    int idx;
    Qwen3Attention self_attn;
    Qwen3MoeSparseMoeBlock mlp;
    

    Tensor forward(Tensor hidden,
    const Tensor& cos_emb,
    const Tensor& sin_emb,
    std::vector<KVCache>& kvcache,
    FlashAttnEngine& engine,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& O,
    Tensor& output,
    Tensor& router_logits,
    Tensor& router_scores, Tensor& router_indices,
    Tensor& gathered, Tensor& token_map,
    Tensor& slot_map, Tensor& expert_offsets,
    std::vector<Qwen3MoeDecoderLayer>& layers);
};


struct Qwen3Moe : LanguageModel {
    Qwen3MoeConfig config;
    Tensor embed_tokens; 
    std::vector<Qwen3MoeDecoderLayer> layers;
    Tensor norm;
    Tensor lm_head;
    Qwen3RotaryEmbedding rotary_embedding;
    Tensor residual;
    
    static std::unique_ptr<Qwen3Moe> from_pretrained(const std::string& hf_dir, int device = 0);
    static std::unique_ptr<Qwen3Moe> from_pretrained(const Qwen3MoeConfig& config, int device = 0);


    Tensor forward(const Tensor& input_ids, Tensor& position_ids,
                    std::vector<KVCache>& kvcache,
                    FlashAttnEngine& engine) override;
};
