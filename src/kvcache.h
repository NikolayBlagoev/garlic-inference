#pragma once
#include "cuda-common.cuh"
#include <numeric>
#include <algorithm>


#define PAGE_SIZE 32


struct KVCache{
    

    cudaDataType_t      dtype;
    int                 device;
    int                 num_heads;
    int                 head_dim;
    int                 batch_size; // TODO: in the future I want batch size to be dynamic
    int                 max_pages; // maximum pages IN **GPU**!
    int                 max_pages_per_seq;
    std::vector<int>    empty_pages; // currently which page can be used

                        // allocate all pages...
                        // TODO: CAN THIS GROW/SHRINK EVERY 2^.. sizes? SHOULD BE AMORTIZED FINE
    Tensor              k_pages, v_pages;

                        // stores each logical page to which physical page it maps
    void*               page_table = nullptr;

    void*               qkv_lens = nullptr; // [batch_size, 1, 1, 1]
                                            // equivalent to sequelnce length q and kv for cudnn
    // to be easily editable between both..
    // TODO: HOW DOES... TORCH DEAL WITH THIS?
    std::vector<int>    host_qkv_lens;
    int*    host_page_table;


    KVCache() : num_heads(0), batch_size(0), head_dim(0), max_pages(0),
                max_pages_per_seq(0), device(0), dtype(CUDA_R_16BF) {}

    KVCache(int num_heads, int head_dim, int max_pages, int batch_size,
            int max_pages_per_seq, cudaDataType_t dtype = CUDA_R_16BF,
            int device = 0) :
            num_heads(num_heads), head_dim(head_dim),
            max_pages(max_pages), batch_size(batch_size),
            max_pages_per_seq(max_pages_per_seq),
            device(device), dtype(dtype),
            host_qkv_lens(batch_size, 0) {
                cudaSetDevice(device);
                k_pages = Tensor({max_pages, num_heads, PAGE_SIZE, head_dim}, dtype, device);
                v_pages = Tensor({max_pages, num_heads, PAGE_SIZE, head_dim}, dtype, device);
                cudaMallocHost(&host_page_table, (size_t) batch_size * max_pages_per_seq * sizeof(int));
                memset(host_page_table, 0xFF, (size_t) batch_size * max_pages_per_seq * sizeof(int));
                cudaMalloc(&page_table, (size_t) batch_size * max_pages_per_seq * sizeof(int));
                cudaMalloc(&qkv_lens, (size_t) batch_size * sizeof(int));
                cudaMemset(page_table, 0xFF, (size_t) batch_size * max_pages_per_seq * sizeof(int));
                cudaMemset(qkv_lens, 0, (size_t) batch_size * sizeof(int));

                empty_pages.resize(max_pages);
                std::iota(empty_pages.begin(), empty_pages.end(), 0); // works as arrange from 0 to ...
                std::reverse(empty_pages.begin(), empty_pages.end());
    }

    KVCache(const KVCache& o) = delete;
    void prepare_table(int seql);
    void add_kv(const Tensor& new_k, const Tensor& new_v);


    // ownership transfer
    KVCache(KVCache&& o) noexcept : num_heads(o.num_heads), head_dim(o.head_dim),
            max_pages(o.max_pages), batch_size(o.batch_size),
            max_pages_per_seq(o.max_pages_per_seq),
            device(o.device), dtype(o.dtype){

            k_pages = std::move(o.k_pages);
            v_pages = std::move(o.v_pages);

            page_table = o.page_table;
            o.page_table = nullptr;

            qkv_lens = o.qkv_lens;
            o.qkv_lens = nullptr;
            empty_pages = std::move(o.empty_pages);
            host_qkv_lens = std::move(o.host_qkv_lens);
            host_page_table = std::move(o.host_page_table);
            
    }
    
    // ownership transfer
    KVCache& operator=(KVCache&& o) noexcept {
        if (this != &o) {
            // TODO : CLEAR OLD STUFF on this
            num_heads         = o.num_heads;
            head_dim          = o.head_dim;
            max_pages         = o.max_pages;
            batch_size        = o.batch_size;
            max_pages_per_seq = o.max_pages_per_seq;
            device            = o.device;
            dtype             = o.dtype;
            empty_pages = std::move(o.empty_pages);
            host_qkv_lens = std::move(o.host_qkv_lens);
            host_page_table = std::move(o.host_page_table);

            k_pages = std::move(o.k_pages);
            v_pages = std::move(o.v_pages);

            page_table = o.page_table;
            o.page_table = nullptr;

            qkv_lens = std::move(o.qkv_lens);
            o.qkv_lens = nullptr;
        }
        return *this;
    }


private:
    int get_next_free_page(){
        // c++ cannot remove from front???
        int tmp = empty_pages.back();
        empty_pages.pop_back();
        return tmp;
    }
};
