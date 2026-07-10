#include <cuda_runtime.h>
#include "cuda-common.cuh"
#include "simple-ops.cuh"
#include "mat-ops.cuh"
#include "tensor.h"
#include "argmax.cuh"
#include "fattn.cuh"
#include "bpe.h"
#include <cuda_fp8.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;
#include "profiler.h"
_TIMING_RESULTS tmrs{};
#include "qwen3_moe.h"
#include "moe-cache.h"
MoEManager* JoseMurinho = nullptr;
// g_expert_pool is defined in tensor.cu; we initialize it after model load below
#include <nvml.h>
#include "loader.h"
#include <thread>
int main() {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    bool use_cpu = true;
    double joules, tm_ptr, watt_ptr;
    // {
    //     using namespace std::chrono_literals;
    //     auto elm = PowerProfiler(&joules, &tm_ptr, &watt_ptr, device, use_cpu);
    //     std::this_thread::sleep_for(10000ms);
    // }
    // std::cout<<"IDLE Joules: "<<joules<<"J Time: "<<tm_ptr<<"s "<<watt_ptr<<"W\n";
    BPETokenizer tokenizer = BPETokenizer::load("qwen3-30b-fp8");
    Qwen3MoeConfig config = Qwen3MoeConfig::from_pretrained("qwen3-30b-fp8");
    // JoseMurinho = new MRU_MoEManager(config.num_experts_per_tok*3*config.num_hidden_layers, config.num_hidden_layers);
    JoseMurinho = new LRU_MoEManager(config.num_experts_per_tok*5*config.num_hidden_layers);
    Qwen3Moe model;
    {
        // auto elm = PowerProfiler(&joules, &tm_ptr, &watt_ptr, device, use_cpu);
        model = Qwen3Moe::from_pretrained("qwen3-30b-fp8");
    }

    // Init pinned staging pool using actual expert DataView size (gate+up+down packed)
    // Use first offloaded expert from layer 2 onward
    // {
    //     const Tensor& rep = model.layers[2].mlp.gate_proj[0];
    //     size_t expert_bytes = rep._data->num_elements * Tensor::element_size(rep.dtype());
    //     g_expert_pool = new PinnedMemPool(config.num_experts_per_tok*8*config.num_hidden_layers, expert_bytes);
    // }

    std::cout<<"MODEL LOADING Joules: "<<joules<<"J Time: "<<tm_ptr<<"s "<<watt_ptr<<"W\n";
    // {
    //     using namespace std::chrono_literals;
    //     auto elm = PowerProfiler(&joules, &tm_ptr, &watt_ptr, device, use_cpu);
    //     std::this_thread::sleep_for(10000ms);
    // }
    // std::cout<<"IDLE WITH MODEL Joules: "<<joules<<"J Time: "<<tm_ptr<<"s "<<watt_ptr<<"W\n";
    
    std::string str; 

    std::cout << "Ask the model something: \n"; 
    std::getline(std::cin, str); 
    auto encode = tokenizer.encode(str);
    
    Tensor x({1, encode.size()}, CUDA_R_32U, 0);
    x.set_data<uint32_t>(encode, 4*encode.size());
    
    // fill_inplace(x, 901.0f);

    Tensor pos_ids = Tensor({1, encode.size()}, CUDA_R_32U);
    arrange(pos_ids, 0, 1);

    int batch_size = x.shape[0];
    const int max_pages = 256;
    const int max_pages_per_seq = 128;
    
    std::vector<KVCache> kvcaches;
    {
        auto elm = PowerProfiler(&joules, &tm_ptr, &watt_ptr, device, use_cpu);
        kvcaches.reserve(config.num_hidden_layers);
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            kvcaches.emplace_back(config.num_key_value_heads, config.head_dim,
                                    max_pages, batch_size, max_pages_per_seq, model.lm_head.dtype());
        }
    }
    // std::cout<<"Model ready"<<std::endl;
    // using namespace std::chrono_literals;
    // std::this_thread::sleep_for(60000ms);
    // std::exit(0);
    // std::cout<<"KV CACHE CREATION Joules: "<<joules<<"J Time: "<<tm_ptr<<"s "<<watt_ptr<<"W\n";
    // {
    //     using namespace std::chrono_literals;
    //     std::this_thread::sleep_for(10000ms);
    // }
    // {
    //     using namespace std::chrono_literals;
    //     auto elm = PowerProfiler(&joules, &tm_ptr, &watt_ptr, device, use_cpu);
    //     std::this_thread::sleep_for(10000ms);
    // }
    // std::cout<<"IDLE WITH KV CACHE Joules: "<<joules<<"J Time: "<<tm_ptr<<"s "<<watt_ptr<<"W\n";
    // std::cout << std::endl;
    FlashAttnEngine engine(batch_size, config.num_attention_heads,
                            config.num_key_value_heads,
                            config.head_dim, max_pages, max_pages_per_seq);
    auto tm1 = high_resolution_clock::now();
    int seq_len = x.shape[1];
    engine.get_graph(seq_len, model.lm_head.dtype());
    {
        // auto elm = PowerProfiler(&joules, &tm_ptr, &watt_ptr, device, use_cpu);
        for(int j = 0; j < 601; j++){
            // std::cout<<j<<std::endl;
            seq_len = x.shape[1];
            for(auto& kv_page : kvcaches) TIME_PROFILE(kv_page.prepare_table(seq_len), &tmrs.kv_cache_prepare);
            
            
            Tensor logits = model.forward(x, pos_ids, kvcaches, engine);
            auto tm2 = high_resolution_clock::now();
            auto ret = argmax(logits, x);
            auto delta2 = duration_cast<microseconds>(high_resolution_clock::now() - tm2);
            tmrs.argmax += delta2.count();
            tm2 = high_resolution_clock::now();
            std::cout << tokenizer.decode(ret) << std::flush;

            if(j == 0){
                x = Tensor({1,1}, CUDA_R_32U, 0);
                std::vector<uint32_t> tmp_data = {ret[ret.size()-1]};
                x.set_data<uint32_t>(tmp_data, 4);
            }

            // cudaStreamSynchronize(get_load_offload_stream());
            pos_ids = Tensor({1,1}, CUDA_R_32U, 0);
            std::vector<uint32_t> tmp_data = {(uint32_t)(encode.size() + j)};
            pos_ids.set_data<uint32_t>(tmp_data, 4);
            if(j == 0){
                engine.get_graph(1, model.lm_head.dtype());
                tm1 = high_resolution_clock::now();
            } 
            delta2 = duration_cast<microseconds>(high_resolution_clock::now() - tm2);
            tmrs.misc += delta2.count();

        }
    }
    std::cout<<"INFERENCE Joules: "<<joules<<"J Time: "<<tm_ptr<<"s "<<watt_ptr<<"W\n";
    
    std::cout<<std::endl;
    auto delta = duration_cast<milliseconds>(high_resolution_clock::now() - tm1);
    std::cout << "Time: " << delta.count() << std::endl;
    float s = (float) delta.count() / 1000.0f;
    std::cout << "Tok/s: " << 600/s << std::endl;

    // std::cout<< "KV cache prepare: " << tmrs.kv_cache_prepare / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "ARGMAX: " << tmrs.argmax / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Embedding: " << tmrs.embedding / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Deembedding: " << tmrs.deembedding / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Rope forward: " << tmrs.rope_forward / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Casting: " << tmrs.casting / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Add in place: " << tmrs.addinplace / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Make copy: " << tmrs.makecopy / (1000.0 * 1000.0) << std::endl;
    
    // std::cout<< "Attention total: " << tmrs.selfattn / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention Inits: " << tmrs.selfattn_inits / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention cast: " << tmrs.selfattn_cast / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention kvcache: " << tmrs.selfattn_kvcacheadd / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention projections: " << tmrs.selfattn_projections / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Atention rms norms: " << tmrs.selfattn_rmsnorms / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention posembs: " << tmrs.selfattn_posembs / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention transpose: " << tmrs.selfattn_transpose / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "Attention attention: " << tmrs.selfattn_attn / (1000.0 * 1000.0) << std::endl;
    

    // std::cout<< "MLP total: " << tmrs.mlp / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "MLP cast: " << tmrs.mlp_cast / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "MLP matmul: " << tmrs.mlp_matmul / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "MLP silu: " << tmrs.mlp_silu / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "MLP element wise: " << tmrs.mlp_elmwise / (1000.0 * 1000.0) << std::endl;
    // std::cout<< "MLP inits: " << tmrs.mlp_inits / (1000.0 * 1000.0) << std::endl;


    // std::cout<< "MISC: " << tmrs.misc / (1000.0 * 1000.0) << std::endl;
    return 0;
}