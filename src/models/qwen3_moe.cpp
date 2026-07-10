#include "qwen3_moe.h"


#include "../src-cuda/embedding.cuh"
#include "../src-cuda/norm.cuh"
#include "../src-cuda/rope.cuh"
#include "../src-cuda/mat-ops.cuh"
#include "../src-cuda/simple-ops.cuh"
#include "../src-cuda/activations.cuh"
#include "../src-cuda/fattn.cuh"
#include "../src-cuda/moe.cuh"
#include "../src-cuda/argmax.cuh"
#include "../kvcache.h"
#include "profiler.h"
#include "moe-cache.h"
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include "../external/json.hpp"
#include "loader.h"

namespace fs = std::filesystem;

Qwen3MoeConfig Qwen3MoeConfig::from_pretrained(const std::string& hf_dir) {
    std::ifstream f(hf_dir + "/config.json");
    json cfg = json::parse(f);
    Qwen3MoeConfig c;
    c.hidden_size = cfg["hidden_size"].get<int>();
    c.num_hidden_layers = cfg["num_hidden_layers"].get<int>();
    c.num_attention_heads = cfg["num_attention_heads"].get<int>();
    c.num_key_value_heads = cfg["num_key_value_heads"].get<int>();
    c.intermediate_size = cfg["intermediate_size"].get<int>();
    c.vocab_size = cfg["vocab_size"].get<int>();
    c.max_position_embeddings = cfg["max_position_embeddings"].get<int>();
    c.rms_norm_eps = cfg["rms_norm_eps"].get<float>();
    c.rope_theta = cfg["rope_theta"].get<float>();
    c.head_dim = cfg["head_dim"].get<int>();
    c.num_experts = cfg["num_experts"].get<int>();
    c.num_experts_per_tok = cfg["num_experts_per_tok"].get<int>();
    c.moe_intermediate_size = cfg["moe_intermediate_size"].get<int>();
    return c;
};

Qwen3Moe Qwen3Moe::from_pretrained(const std::string& hf_dir, int device) {
    Qwen3Moe model;
    model.config = Qwen3MoeConfig::from_pretrained(hf_dir);
    ModelLoader wl(hf_dir);

    model.rotary_embedding.attention_scaling = 1.0f;
    model.rotary_embedding.inv_freq = Tensor({model.config.head_dim / 2}, CUDA_R_32F, device);
    initialize_rope(model.config.rope_theta, model.rotary_embedding.inv_freq);

    model.embed_tokens = wl.load("model.embed_tokens.weight", device);
    model.norm = wl.load("model.norm.weight", device);

    model.lm_head = wl.load("lm_head.weight", device);

    model.layers.reserve(model.config.num_hidden_layers);
    
    for (int i = 0; i < model.config.num_hidden_layers; i++) {
        std::cout<<"LOADING LAYER "<<i<<std::endl;
        const std::string lp = "model.layers." + std::to_string(i) + ".";

        Qwen3MoeDecoderLayer layer;
        layer.idx = i;
        layer.rms_norm_eps = model.config.rms_norm_eps;

        layer.mlp.moe_intermediate_size = model.config.moe_intermediate_size;
        layer.mlp.num_experts_per_tok = model.config.num_experts_per_tok;
        layer.mlp.num_experts = model.config.num_experts;

        layer.input_layernorm = wl.load(lp + "input_layernorm.weight", device);
        layer.post_attention_layernorm = wl.load(lp + "post_attention_layernorm.weight", device);

        layer.self_attn.num_heads = model.config.num_attention_heads;
        layer.self_attn.head_dim = model.config.head_dim;
        layer.self_attn.num_kv_heads = model.config.num_key_value_heads;
        layer.self_attn.rms_norm_eps = model.config.rms_norm_eps;
        layer.self_attn.q_proj = wl.load(lp + "self_attn.q_proj.weight", device);
        layer.self_attn.k_proj = wl.load(lp + "self_attn.k_proj.weight", device);
        layer.self_attn.v_proj = wl.load(lp + "self_attn.v_proj.weight", device);
        layer.self_attn.o_proj = wl.load(lp + "self_attn.o_proj.weight", device);

        layer.self_attn.q_norm = wl.load(lp + "self_attn.q_norm.weight", device);
        layer.self_attn.k_norm = wl.load(lp + "self_attn.k_norm.weight", device);
        // TODO: CHECK IF tie word embeddings is true or false!
        layer.mlp.gate = wl.load(lp + "mlp.gate.weight", device);
        layer.mlp.layer_idx = i;
        layer.mlp.gate_proj.resize(model.config.num_experts);
        layer.mlp.up_proj.resize(model.config.num_experts);
        layer.mlp.down_proj.resize(model.config.num_experts);
        for (int j = 0; j < model.config.num_experts; j++) {
            const std::string ep = lp + "mlp.experts." + std::to_string(j) + ".";
            bool on_cpu = (i >= 2);

            // peek at metadata to get shapes and dtype without loading
            const WeightContainer& g_meta = wl.peek(ep + "gate_proj.weight");
            const WeightContainer& u_meta = wl.peek(ep + "up_proj.weight");
            const WeightContainer& d_meta = wl.peek(ep + "down_proj.weight");

            long long g_elems = 1; for (int d : g_meta.shape) g_elems *= d;
            long long u_elems = 1; for (int d : u_meta.shape) u_elems *= d;
            long long d_elems = 1; for (int d : d_meta.shape) d_elems *= d;
            long long total   = g_elems + u_elems + d_elems;
            size_t    elem_sz = Tensor::element_size(g_meta.dtype);

            // single contiguous allocation for gate + up + down
            auto dv = std::make_shared<DataView>(g_meta.dtype, total, device, on_cpu);
            wl.load_into(ep + "gate_proj.weight", *dv, 0);
            wl.load_into(ep + "up_proj.weight",   *dv, (int)(g_elems * elem_sz));
            wl.load_into(ep + "down_proj.weight",  *dv, (int)((g_elems + u_elems) * elem_sz));

            layer.mlp.gate_proj[j] = Tensor(g_meta.shape, dv, 0);
            layer.mlp.up_proj[j] = Tensor(u_meta.shape, dv, (uint64_t)(g_elems * elem_sz));
            layer.mlp.down_proj[j] = Tensor(d_meta.shape, dv, (uint64_t)((g_elems + u_elems) * elem_sz));

            wl.load_scale(ep + "gate_proj.weight", layer.mlp.gate_proj[j]);
            wl.load_scale(ep + "up_proj.weight",   layer.mlp.up_proj[j]);
            wl.load_scale(ep + "down_proj.weight",  layer.mlp.down_proj[j]);
            
            
            JoseMurinho->inform(std::to_string(i) + "-" + std::to_string(j), layer.mlp.gate_proj[j], i);
        }
        
        model.layers.push_back(std::move(layer));
    }

    return model;
}



Tensor Qwen3MoeSparseMoeBlock::forward(Tensor& hidden, Tensor& router_logits,
                                    Tensor& router_scores, Tensor& router_indices,
                                    Tensor& gathered, Tensor& token_map,
                                    Tensor& slot_map, Tensor& expert_offsets,
                                    std::vector<Qwen3MoeDecoderLayer>& layers){
    int B = hidden.shape[0];
    int S = hidden.shape[1];
    int D = hidden.shape[2];
    int T = B * S;

    
    
    
    matmul(router_logits, hidden, gate);
    topk(router_logits, router_indices, router_scores, num_experts_per_tok);
    
    
    
    hidden.shape = {T, D};
    
    moe_gather(hidden, router_indices, gathered, token_map, slot_map, expert_offsets, num_experts_per_tok, num_experts);
    
    std::vector<uint32_t> h_offsets(num_experts + 1);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets.data(),
               (num_experts + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost, get_compute_stream());
    cudaStreamSynchronize(get_compute_stream());
    // std::cout<<std::endl;
    const uint64_t elem_bytes = Tensor::element_size(hidden.dtype());
    int total_experts_activated = 0;
    uint32_t largest_n_e = 0;
    for (int e = 0; e < num_experts; e++) {
        uint32_t n_e = h_offsets[e + 1] - h_offsets[e];
        if (n_e == 0) continue;
        if(n_e > largest_n_e) largest_n_e = n_e;
        total_experts_activated+=1;
        // std::cout<<"Preparing "<<std::to_string(layer_idx) + "-" + std::to_string(e)<<std::endl;
        JoseMurinho->prepare(std::to_string(layer_idx) + "-" + std::to_string(e), 1.0, {layer_idx, layer_idx+1, layer_idx+2});
    }
    int offset = 2;
    int speculative_top_k = (int)(1.0f*num_experts_per_tok);
    if(layer_idx >= 2-offset && layer_idx < layers.size() - offset && T < 3){
        Tensor tmp_router_logits({B, S, num_experts}, hidden.dtype(), hidden.device());
        Tensor tmp_router_scores({B*S, speculative_top_k}, CUDA_R_32F, hidden.device());
        Tensor tmp_router_indices({B*S, speculative_top_k}, CUDA_R_32U, hidden.device());
        hidden.shape = {B,S, D};
        matmul(tmp_router_logits, hidden, layers[layer_idx + offset].mlp.gate, get_secondary_stream());
        topk(tmp_router_logits, tmp_router_indices, tmp_router_scores, speculative_top_k, get_secondary_stream());
        std::vector<uint32_t> h_router_indices(speculative_top_k);
        cudaMemcpyAsync(h_router_indices.data(), tmp_router_indices.data(),
                (speculative_top_k) * sizeof(uint32_t), cudaMemcpyDeviceToHost, get_secondary_stream());
        cudaStreamSynchronize(get_secondary_stream());
        for (int n_e = 0; n_e < speculative_top_k; n_e++) {
            uint32_t e = h_router_indices[n_e];
            
            // std::cout<<"Pre-Preparing "<<std::to_string(layer_idx+1) + "-" + std::to_string(e)<<std::endl;
            JoseMurinho->prepare(std::to_string(layer_idx+offset) + "-" + std::to_string(e), 0, {layer_idx, layer_idx+1, layer_idx+2});
        }
        hidden.shape = {T, D};
    }
    

    Tensor buffer_outputs({1, B*S, moe_intermediate_size}, hidden.dtype(), hidden.device());
    std::vector<bool> completed_experts(num_experts, false);
    int remaining = total_experts_activated;
    Tensor gate_out({1, largest_n_e, moe_intermediate_size}, buffer_outputs._data, 0);
    while(remaining > 0){
        for (int e = 0; e < num_experts; e++) {
            uint32_t n_e = h_offsets[e + 1] - h_offsets[e];
            if (completed_experts[e] || n_e == 0) continue;
            
            if(!JoseMurinho->is_done(std::to_string(layer_idx) + "-" + std::to_string(e))){
                continue;
            }
            completed_experts[e] = true;
            remaining--;
            uint64_t byte_off = (uint64_t)h_offsets[e] * D * elem_bytes;
            Tensor input_view({1, n_e, D}, gathered._data, byte_off);
            

            // Tensor gate_out({1, n_e, moe_intermediate_size}, buffer_outputs._data, 0);
            // Tensor up_out({1, n_e, moe_intermediate_size}, hidden.dtype(), hidden.device());
            gate_out.shape = {1, n_e, moe_intermediate_size};
            matmul(gate_out, input_view, up_proj[e], gate_proj[e]);


            Tensor out_view({1, n_e, D}, gathered._data, byte_off);
            matmul(out_view, gate_out, down_proj[e]);
            
        
        }
    }
    
    
    
    moe_scatter(hidden, gathered, router_scores, slot_map, T, num_experts_per_tok);
    hidden.shape = {B, S, D};
   
    return hidden;
}

Tensor Qwen3MoeDecoderLayer::forward(
    Tensor hidden,
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
    std::vector<Qwen3MoeDecoderLayer>& layers){
    
    Tensor residual;
    TIME_PROFILE(hidden.copy_into(residual), &tmrs.makecopy);
    CUDA_CHECK(cudaGetLastError()); 
    rmsnorm(hidden, input_layernorm, rms_norm_eps);
    
    TIME_PROFILE(hidden = self_attn.forward(hidden, cos_emb, sin_emb, kvcache, engine, idx, Q, K, V, O, output), &tmrs.selfattn);
    
    CUDA_CHECK(cudaGetLastError()); 
    TIME_PROFILE(rmsnorm_fused(hidden, residual, post_attention_layernorm, rms_norm_eps), &tmrs.makecopy);
    // std::cout<<"Entering MLP of Layer "<<idx<<std::endl;
    TIME_PROFILE(hidden = mlp.forward(hidden,router_logits,
                                    router_scores,router_indices,
                                    gathered, token_map,
                                    slot_map, expert_offsets, layers), &tmrs.mlp);
    
    CUDA_CHECK(cudaGetLastError()); 
    TIME_PROFILE(add_inplace(hidden, residual), &tmrs.addinplace);
    CUDA_CHECK(cudaGetLastError()); 
    return hidden;
}

Tensor Qwen3Moe::forward(
    const Tensor& input_ids,
    Tensor& position_ids,
    std::vector<KVCache>& kvcache,
    FlashAttnEngine& engine){

    const int B = position_ids.shape[0];
    const int S = position_ids.shape[1];
    CUDA_CHECK(cudaGetLastError()); 
    Tensor hidden({B, S, config.hidden_size}, lm_head.dtype(), input_ids.device());
    // kernel is Batch irrelevant :)
    TIME_PROFILE(embedding_forward(hidden, input_ids, embed_tokens), &tmrs.embedding);
    

    CUDA_CHECK(cudaGetLastError()); 
        // const std::vector<int> out_shape = {B, S, config.head_dim};
    Tensor cos_emb({B, S, config.head_dim}, CUDA_R_32F, hidden.device());
    Tensor sin_emb({B, S, config.head_dim}, CUDA_R_32F, hidden.device());
    TIME_PROFILE(rope_forward(hidden, position_ids, rotary_embedding.inv_freq, cos_emb, sin_emb), &tmrs.rope_forward);
    CUDA_CHECK(cudaGetLastError()); 
    auto tm1 = high_resolution_clock::now();
    Tensor Q = Tensor({B, S, config.num_attention_heads * config.head_dim}, hidden.dtype(), hidden.device());
    Tensor K = Tensor({B, S, config.num_key_value_heads * config.head_dim}, hidden.dtype(), hidden.device());
    Tensor V = Tensor({B, S, config.num_key_value_heads * config.head_dim}, hidden.dtype(), hidden.device());
    Tensor O = Tensor({B, config.num_attention_heads, S,  config.head_dim}, hidden.dtype(), hidden.device());
    Tensor output = Tensor({B, S, config.hidden_size}, hidden.dtype(), hidden.device());


    Tensor router_logits({B, S, config.num_experts}, hidden.dtype(), hidden.device());
    Tensor router_scores({B*S, config.num_experts_per_tok}, CUDA_R_32F, hidden.device());
    Tensor router_indices({B*S, config.num_experts_per_tok}, CUDA_R_32U, hidden.device());
    
    Tensor gathered({B * S * config.num_experts_per_tok, config.hidden_size}, hidden.dtype(), hidden.device());
    Tensor token_map({B*S, config.num_experts_per_tok}, CUDA_R_32U, hidden.device());
    Tensor slot_map({B*S, config.num_experts_per_tok}, CUDA_R_32U, hidden.device());
    Tensor expert_offsets({config.num_experts + 1}, CUDA_R_32U, hidden.device());
    
    
    tmrs.selfattn_inits += duration_cast<microseconds>(high_resolution_clock::now() - tm1).count();

    for (auto& layer : layers){
        hidden = layer.forward(hidden, cos_emb, sin_emb, kvcache, engine, Q, K, V, O, output, 
                            router_logits, router_scores, router_indices, gathered, token_map, slot_map, expert_offsets, layers);

    }
    // std::cout << "POST LOOP" << std::endl;
    CUDA_CHECK(cudaGetLastError()); 
    TIME_PROFILE(rmsnorm(hidden, norm, config.rms_norm_eps),&tmrs.norm);
    CUDA_CHECK(cudaGetLastError()); 
    // Tensor::list_values(hidden, 20);
    Tensor logits({B, S, config.vocab_size}, hidden.dtype(), hidden.device());
    TIME_PROFILE(matmul(logits, hidden, lm_head), &tmrs.deembedding);
    CUDA_CHECK(cudaGetLastError()); 
    // Tensor::list_values(logits, 20);
    return logits;
}
