#include "qwen3.h"


#include "../src-cuda/embedding.cuh"
#include "../src-cuda/norm.cuh"
#include "../src-cuda/rope.cuh"
#include "../src-cuda/mat-ops.cuh"
#include "../src-cuda/simple-ops.cuh"
#include "../src-cuda/activations.cuh"
#include "../src-cuda/fattn.cuh"
#include "../kvcache.h"
#include "profiler.h"
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include "../external/json.hpp"
#include "loader.h"

namespace fs = std::filesystem;

Qwen3Config Qwen3Config::from_pretrained(const std::string& hf_dir) {
    std::ifstream f(hf_dir + "/config.json");
    json cfg = json::parse(f);
    Qwen3Config c;
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
    return c;
};

Qwen3 Qwen3::from_pretrained(const std::string& hf_dir, int device) {
    Qwen3 model;
    model.config = Qwen3Config::from_pretrained(hf_dir);
    ModelLoader wl(hf_dir);

    model.rotary_embedding.attention_scaling = 1.0f;
    model.rotary_embedding.inv_freq = Tensor({model.config.head_dim / 2}, CUDA_R_32F, device);
    initialize_rope(model.config.rope_theta, model.rotary_embedding.inv_freq);

    model.embed_tokens = wl.load("model.embed_tokens.weight", device);
    model.norm = wl.load("model.norm.weight", device);

    model.lm_head = model.embed_tokens;

    model.layers.reserve(model.config.num_hidden_layers);
    for (int i = 0; i < model.config.num_hidden_layers; ++i) {
        const std::string lp = "model.layers." + std::to_string(i) + ".";

        Qwen3DecoderLayer layer;
        layer.idx = i;
        layer.rms_norm_eps = model.config.rms_norm_eps;

        layer.mlp.intermediate_size = model.config.intermediate_size;


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
        
        const WeightContainer& g_meta = wl.peek(lp + "mlp.gate_proj.weight");
        const WeightContainer& u_meta = wl.peek(lp + "mlp.up_proj.weight");
        const WeightContainer& d_meta = wl.peek(lp + "mlp.down_proj.weight");

        long long g_elems = 1; for (int d : g_meta.shape) g_elems *= d;
        long long u_elems = 1; for (int d : u_meta.shape) u_elems *= d;
        long long d_elems = 1; for (int d : d_meta.shape) d_elems *= d;
        long long total   = g_elems + u_elems + d_elems;
        size_t elem_sz = Tensor::element_size(g_meta.dtype);
        auto dv = std::make_shared<DataView>(g_meta.dtype, total, device, false);
        wl.load_into(lp + "mlp.gate_proj.weight", *dv, 0);
        wl.load_into(lp + "mlp.up_proj.weight", *dv, (int)(g_elems * elem_sz));
        wl.load_into(lp + "mlp.down_proj.weight", *dv, (int)((g_elems + u_elems) * elem_sz));

        layer.mlp.gate_proj = Tensor(g_meta.shape, dv, 0);
        layer.mlp.up_proj = Tensor(u_meta.shape, dv, (uint64_t)(g_elems * elem_sz));
        layer.mlp.down_proj = Tensor(d_meta.shape, dv, (uint64_t)((g_elems + u_elems) * elem_sz));
        
        wl.load_scale(lp + "mlp.gate_proj.weight", layer.mlp.gate_proj);
        wl.load_scale(lp + "mlp.up_proj.weight",   layer.mlp.up_proj);
        wl.load_scale(lp + "mlp.down_proj.weight",  layer.mlp.down_proj);

        model.layers.push_back(std::move(layer));
    }

    return model;
}


Tensor Qwen3Attention::forward(Tensor& hidden,
    const Tensor& cos_emb,
    const Tensor& sin_emb,
    std::vector<KVCache>& kvcache,
    FlashAttnEngine& engine,
    int idx,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& O,
    Tensor& output){
    
    KVCache& layer_kv_cache = kvcache[idx];
    int B = hidden.shape[0];
    int S = hidden.shape[1];
    int D = hidden.shape[2];
    Q.shape = {B, S, num_heads*head_dim};
    K.shape = {B, S, num_kv_heads*head_dim};
    V.shape = {B, S, num_kv_heads*head_dim};
    // std::cout << "Q "<< Q.dtype() << " " << q_proj.dtype()  << std::endl;
    // std::cout << "Q "<< Q.shape[0] << " " <<Q.shape[1] << " " <<Q.shape[2] << std::endl;
    // std::cout << "Q "<< q_proj.shape[0] << " " <<q_proj.shape[1] << " " <<q_proj.shape[2] << std::endl;
    // std::cout << "Q "<< hidden.shape[0] << " " <<hidden.shape[1] << " " <<hidden.shape[2] << std::endl;
    // Tensor::list_values(q_proj, 20);
    // std::cout<<"Q"<<std::endl;
    TIME_PROFILE(matmul(Q, hidden, q_proj, false), &tmrs.selfattn_projections);
    
    // std::cout << "Q value: ";
    // Tensor::list_values(Q, 20);
    // std::cout<<"K"<<std::endl;
    TIME_PROFILE(matmul(K, hidden, k_proj, true), &tmrs.selfattn_projections);
    // std::cout<<"V"<<std::endl;
    TIME_PROFILE(matmul(V, hidden, v_proj, true), &tmrs.selfattn_projections);
    
    TIME_PROFILE(rmsnorm(Q, q_norm, rms_norm_eps, head_dim), &tmrs.selfattn_rmsnorms);
    TIME_PROFILE(rmsnorm(K, k_norm, rms_norm_eps, head_dim), &tmrs.selfattn_rmsnorms);

    Q.shape = {B, S, num_heads, head_dim};
    K.shape = {B, S, num_kv_heads, head_dim};
    V.shape = {B, S, num_kv_heads, head_dim};

    // RoPE expects [B, S, num_heads, head_dim]; apply before transposing
    TIME_PROFILE(apply_rotary_pos_emb(Q, cos_emb, sin_emb),&tmrs.selfattn_posembs);
    TIME_PROFILE(apply_rotary_pos_emb(K, cos_emb, sin_emb),&tmrs.selfattn_posembs);

    TIME_PROFILE(Q = transpose(Q, 1, 2), &tmrs.selfattn_transpose);   // [B, num_heads, S, head_dim]
    TIME_PROFILE(K = transpose(K, 1, 2), &tmrs.selfattn_transpose);    // [B, num_kv_heads, S, head_dim]
    TIME_PROFILE(V = transpose(V, 1, 2), &tmrs.selfattn_transpose);   // [B, num_kv_heads, S, head_dim]
    // std::cout << "Q value: ";
    // Tensor::list_values(Q, 20);
    // std::cout << "K value before add: ";
    // Tensor::list_values(K, 20);

    TIME_PROFILE(layer_kv_cache.add_kv(K, V), &tmrs.selfattn_kvcacheadd);
    O.shape = {B, num_heads, S, head_dim};
    // Tensor O({B, num_heads, S, head_dim}, Q.dtype(), Q.device());
    TIME_PROFILE(engine.run(O, Q, layer_kv_cache), &tmrs.selfattn_attn);
    // std::cout << "O value: ";
    // Tensor::list_values(O, 20);
    
    O = transpose(O, 1, 2);   // [B, S, num_heads, head_dim]
    O.shape = {B, S, num_heads * head_dim};
    // std::cout << "O "<< O.dtype() << " " << o_proj.dtype()  << std::endl;
    // Tensor output = Tensor({B, S, D}, CUDA_R_16BF, hidden.device());
    TIME_PROFILE(matmul(output, O, o_proj), &tmrs.selfattn_projections);
    // std::cout << "output value: ";
    // Tensor::list_values(output, 20);
    // std::exit(1);
    return output;
}

Tensor Qwen3MLP::forward(Tensor& hidden, Tensor& gate, Tensor& up, Tensor& down){
    
    if(hidden.dtype() == CUDA_R_16BF) std::cout<<"ERORR\n";
    TIME_PROFILE(matmul(gate, hidden, gate_proj),&tmrs.mlp_matmul);
    TIME_PROFILE(matmul(up, hidden, up_proj),&tmrs.mlp_matmul);
    TIME_PROFILE(silu(gate),&tmrs.mlp_silu);
    TIME_PROFILE(elm_wise(gate, up),&tmrs.mlp_elmwise);
    
    TIME_PROFILE(matmul(down, gate, down_proj),&tmrs.mlp_matmul);
    return down;
}

Tensor Qwen3DecoderLayer::forward(
    Tensor& hidden,
    const Tensor& cos_emb,
    const Tensor& sin_emb,
    std::vector<KVCache>& kvcache,
    FlashAttnEngine& engine,
    Tensor& residual,
    Tensor& Q,
    Tensor& K,
    Tensor& V,
    Tensor& O,
    Tensor& output,
    Tensor& gate,
    Tensor& up,
    Tensor& down){
    
    // std::cout << idx << std::endl;
    // std::cout << "Entry "<< hidden.dtype() << std::endl;
    // Tensor::list_values(hidden, 20);
    
    TIME_PROFILE(hidden.copy_into(residual), &tmrs.makecopy);
    CUDA_CHECK(cudaGetLastError()); 
    rmsnorm(hidden, input_layernorm, rms_norm_eps);
    // std::cout << "POST LAYERNORM: "<< hidden.dtype() << std::endl;
    // Tensor::list_values(hidden, 20);
    TIME_PROFILE(hidden = self_attn.forward(hidden, cos_emb, sin_emb, kvcache, engine, idx, Q, K, V, O, output), &tmrs.selfattn);
    // std::cout << "POST ATTENTION: "<< hidden.dtype() << std::endl;
    // Tensor::list_values(hidden, 20);
    CUDA_CHECK(cudaGetLastError()); 
    // TIME_PROFILE(add_inplace(hidden, residual), &tmrs.addinplace);
    // CUDA_CHECK(cudaGetLastError()); 
    // TIME_PROFILE(hidden.copy_into(residual), &tmrs.makecopy);
    // CUDA_CHECK(cudaGetLastError()); 
    // rmsnorm(hidden, post_attention_layernorm, rms_norm_eps);
    TIME_PROFILE(rmsnorm_fused(hidden, residual, post_attention_layernorm, rms_norm_eps), &tmrs.makecopy);
    TIME_PROFILE(hidden = mlp.forward(hidden, gate, up, down), &tmrs.mlp);
    
    // std::cout << "POST MLP: "<< hidden.dtype() << std::endl;
    // Tensor::list_values(hidden, 20);
    // TIME_PROFILE(hidden = hidden.cast_to(CUDA_R_16BF), &tmrs.casting);
    CUDA_CHECK(cudaGetLastError()); 
    
    CUDA_CHECK(cudaGetLastError()); 
    TIME_PROFILE(add_inplace(hidden, residual), &tmrs.addinplace);
    // std::cout << "POST ADD: "<< hidden.dtype() << std::endl;
    // Tensor::list_values(hidden, 20);
    CUDA_CHECK(cudaGetLastError()); 
    return hidden;
}

Tensor Qwen3::forward(
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
    // Tensor::list_values(hidden, 20);

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
    
    
    tm1 = high_resolution_clock::now();
    Tensor gate({B, S, config.intermediate_size}, hidden.dtype(), hidden.device());
    Tensor up({B, S, config.intermediate_size}, hidden.dtype(), hidden.device());
    Tensor down({B, S, config.hidden_size}, hidden.dtype(), hidden.device());
    tmrs.mlp_inits += duration_cast<microseconds>(high_resolution_clock::now() - tm1).count();
    // std::cout << "Entering loop" << std::endl;
    // cudaStreamSynchronize(get_load_offload_stream());
    for (auto& layer : layers){
                hidden = layer.forward(hidden, cos_emb, sin_emb, kvcache, engine, residual, Q, K, V, O, output, gate, up, down);
        // Tensor::list_values(hidden, 20);

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
