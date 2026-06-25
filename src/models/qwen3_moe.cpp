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

    model.lm_head = model.embed_tokens;

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
        layer.mlp.gate = wl.load(lp + "mlp.gate.weight", device);
        layer.mlp.layer_idx = i;
        layer.mlp.gate_proj.resize(model.config.num_experts);
        layer.mlp.up_proj.resize(model.config.num_experts);
        layer.mlp.down_proj.resize(model.config.num_experts);
        for (int j = 0; j < model.config.num_experts; j++) {
            const std::string ep = lp + "mlp.experts." + std::to_string(j) + ".";
            bool on_cpu = (i >= 4);

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
            // Tensor::list_values(layer.mlp.gate_proj[j],20);
            // Tensor::list_values(layer.mlp.up_proj[j],20);
            // Tensor::list_values(layer.mlp.down_proj[j],20);
            // Tensor tmp = wl.load(ep + "gate_proj.weight", device);
            // Tensor::list_values(tmp,20);
            // tmp = wl.load(ep + "up_proj.weight", device);
            // Tensor::list_values(tmp,20);
            // tmp = wl.load(ep + "down_proj.weight", device);
            // Tensor::list_values(tmp,20);
            // load FP8 scales directly into each view
            wl.load_scale(ep + "gate_proj.weight", layer.mlp.gate_proj[j]);
            wl.load_scale(ep + "up_proj.weight",   layer.mlp.up_proj[j]);
            wl.load_scale(ep + "down_proj.weight",  layer.mlp.down_proj[j]);
            if(layer.mlp.down_proj[j]._scale == nullptr || layer.mlp.down_proj[j].scale() == nullptr) std::exit(2);
            if(layer.mlp.up_proj[j]._scale == nullptr || layer.mlp.up_proj[j].scale() == nullptr) std::exit(2);
            if(layer.mlp.gate_proj[j]._scale == nullptr || layer.mlp.gate_proj[j].scale() == nullptr) std::exit(2);
            
            // DataView::list_values(*layer.mlp.gate_proj[j]._scale);
            // DataView::list_values(*layer.mlp.up_proj[j]._scale);
            // DataView::list_values(*layer.mlp.down_proj[j]._scale);
            // std::exit(1);
            // one inform per expert — all three views share dv so offload/onload moves all three
            JoseMurinho->inform(std::to_string(i) + "-" + std::to_string(j), layer.mlp.gate_proj[j]);
        }
        
        model.layers.push_back(std::move(layer));
    }

    return model;
}



Tensor Qwen3MoeSparseMoeBlock::forward(Tensor& hidden){
    int B = hidden.shape[0];
    int S = hidden.shape[1];
    int D = hidden.shape[2];
    int T = B * S;

    // Router: compute logits and select top-k experts
    Tensor router_logits({B, S, num_experts}, hidden.dtype(), hidden.device());
    Tensor router_scores({T, num_experts_per_tok}, CUDA_R_32F, hidden.device());
    Tensor router_indices({T, num_experts_per_tok}, CUDA_R_32U, hidden.device());
    // std::cout<<layer_idx<<std::endl;
    // if(layer_idx == 30) std::cout << "[L30] gate dtype=" << (int)gate.dtype() << " hidden dtype=" << (int)hidden.dtype() << " hidden[0..3]=";
    // if(layer_idx == 30) { cudaDeviceSynchronize(); Tensor::list_values(hidden, 4); }
    matmul(router_logits, hidden, gate, false);
    // if(layer_idx == 30) { cudaDeviceSynchronize(); std::cout << "[L30] router_logits[0..7]="; Tensor::list_values(router_logits, 8); }
    topk(router_logits, router_indices, router_scores, num_experts_per_tok);
    
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    // Tensor::list_values(router_indices, num_experts_per_tok);
    // Tensor::list_values(router_indices, 4);
    // CUDA_CHECK(cudaGetLastError());
    // Tensor::list_values(router_scores, 8);
    // cudaDeviceSynchronize();
    // CUDA_CHECK(cudaGetLastError());
    // Gather: sort tokens into expert-contiguous layout
    Tensor gathered({T * num_experts_per_tok, D}, hidden.dtype(), hidden.device());
    Tensor token_map({T, num_experts_per_tok}, CUDA_R_32U, hidden.device());
    Tensor slot_map({T, num_experts_per_tok}, CUDA_R_32U, hidden.device());
    Tensor expert_offsets({num_experts + 1}, CUDA_R_32U, hidden.device());
    // moe_gather expects x as [T, D] (2D)
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    Tensor hidden_flat({T, D}, hidden._data, hidden.offset);
    moe_gather(hidden_flat, router_indices, gathered, token_map, slot_map, expert_offsets, num_experts_per_tok, num_experts);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    // Tensor::list_values(token_map, num_experts_per_tok);
    // Tensor::list_values(slot_map, num_experts_per_tok);
    // Tensor::list_values(expert_offsets, num_experts + 1);
    // get experts to CPU
    std::vector<uint32_t> h_offsets(num_experts + 1);
    cudaMemcpy(h_offsets.data(), expert_offsets.data(),
               (num_experts + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    const uint64_t elem_bytes = Tensor::element_size(hidden.dtype());
    for (int e = 0; e < num_experts; e++) {
        uint32_t n_e = h_offsets[e + 1] - h_offsets[e];
        if (n_e == 0) continue;
        // std::cout<<"Preparing "<<std::to_string(layer_idx) + "-" + std::to_string(e)<<std::endl;
        JoseMurinho->prepare(std::to_string(layer_idx) + "-" + std::to_string(e));
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    
    // if(layer_idx == 30){
    //     std::cout<<"\nLAYER 30\n";
    //     Tensor::list_values(router_indices, 8);
    //     Tensor::list_values(expert_offsets);
    // } 
    
    for (int e = 0; e < num_experts; e++) {
        uint32_t n_e = h_offsets[e + 1] - h_offsets[e];
        if (n_e == 0) continue;
        // JoseMurinho->wait(std::to_string(layer_idx) + "-" + std::to_string(e), 0);
        
        uint64_t byte_off = (uint64_t)h_offsets[e] * D * elem_bytes;
        Tensor input_view({1, n_e, D}, gathered._data, byte_off);


        Tensor gate_out({1, n_e, moe_intermediate_size}, hidden.dtype(), hidden.device());
        Tensor up_out({1, n_e, moe_intermediate_size}, hidden.dtype(), hidden.device());
        matmul(gate_out, input_view, gate_proj[e]);


        matmul(up_out, input_view, up_proj[e]);
        silu(gate_out);
        elm_wise(gate_out, up_out);

        Tensor out_view({1, n_e, D}, gathered._data, byte_off);
        matmul(out_view, gate_out, down_proj[e]);
        if(h_offsets[e] == 0){
            std::cout<<"\nLAYER"<<layer_idx<<std::endl;
            Tensor::list_values(out_view, 25);
        }
    
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    // Scatter: weighted sum of expert outputs back to token space
    Tensor output({T, D}, hidden.dtype(), hidden.device());
    moe_scatter(output, gathered, router_scores, slot_map, T, num_experts_per_tok);
    output.shape = {B, S, D};
    // if (layer_idx == 1) {
    //     cudaDeviceSynchronize();
    //     std::cout << "[L1 scores[0..3]]="; Tensor::list_values(router_scores, num_experts_per_tok);
    //     std::cout << "[L1 moe_output[0..3]]="; Tensor::list_values(output, num_experts_per_tok);
    // }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    return output;
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
    Tensor& output){
    
    Tensor residual;
    TIME_PROFILE(hidden.copy_into(residual), &tmrs.makecopy);
    CUDA_CHECK(cudaGetLastError()); 
    rmsnorm(hidden, input_layernorm, rms_norm_eps);
    
    TIME_PROFILE(hidden = self_attn.forward(hidden, cos_emb, sin_emb, kvcache, engine, idx, Q, K, V, O, output), &tmrs.selfattn);
    
    CUDA_CHECK(cudaGetLastError()); 
    TIME_PROFILE(rmsnorm_fused(hidden, residual, post_attention_layernorm, rms_norm_eps), &tmrs.makecopy);
    // std::cout<<"Entering MLP of Layer "<<idx<<std::endl;
    TIME_PROFILE(hidden = mlp.forward(hidden), &tmrs.mlp);
    
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
    // Tensor::list_values(sin_emb, 200);
    // Tensor::list_values(cos_emb, 200);
    auto tm1 = high_resolution_clock::now();
    Tensor Q = Tensor({B, S, config.num_attention_heads * config.head_dim}, hidden.dtype(), hidden.device());
    Tensor K = Tensor({B, S, config.num_key_value_heads * config.head_dim}, hidden.dtype(), hidden.device());
    Tensor V = Tensor({B, S, config.num_key_value_heads * config.head_dim}, hidden.dtype(), hidden.device());
    Tensor O = Tensor({B, config.num_attention_heads, S,  config.head_dim}, hidden.dtype(), hidden.device());
    Tensor output = Tensor({B, S, config.hidden_size}, hidden.dtype(), hidden.device());
    tmrs.selfattn_inits += duration_cast<microseconds>(high_resolution_clock::now() - tm1).count();

    for (auto& layer : layers){
        hidden = layer.forward(hidden, cos_emb, sin_emb, kvcache, engine, Q, K, V, O, output);

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
