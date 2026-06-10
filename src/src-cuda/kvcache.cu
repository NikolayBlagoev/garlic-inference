#include "kvcache.h"

__global__ void increment_qkv_lens_kernel(int* __restrict__ qkv_lens, int delta, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) qkv_lens[b] += delta;
}

__global__ void add_kv_kernel(__nv_bfloat16* __restrict__ new_k, 
                            __nv_bfloat16* __restrict__ new_v,
                            __nv_bfloat16* __restrict__ k_pages,
                            __nv_bfloat16* __restrict__ v_pages,
                            int*  __restrict__ page_table,
                            int*  __restrict__ qkv_lens,
                            const int B, 
                            const int num_heads,
                            const int max_pages_per_seq, 
                            const int seq_l, const int head_dim){
    int b = blockIdx.x;
    int head = blockIdx.y;
    int token = blockIdx.z;    
    if(b >= B || token >= seq_l || head >= num_heads) return;
    int seq_pos = qkv_lens[b] + token; // global sequence position
    int page_idx = seq_pos / PAGE_SIZE;
    int page_offset = seq_pos % PAGE_SIZE;
    int translated_page_id = page_table[b * max_pages_per_seq + page_idx];
    int aligned = (head_dim / 2) * 2;


    __nv_bfloat16* start_new_k = new_k + ((b * num_heads + head) * seq_l + token) * head_dim;
    __nv_bfloat16* start_new_v = new_v + ((b * num_heads + head) * seq_l + token) * head_dim;

    __nv_bfloat16* start_k_pages = k_pages + ((translated_page_id * num_heads + head) * PAGE_SIZE + page_offset) * head_dim;
    __nv_bfloat16* start_v_pages = v_pages + ((translated_page_id * num_heads + head) * PAGE_SIZE + page_offset) * head_dim;
    for(int i = threadIdx.x*2; i + 1 < aligned; i += blockDim.x*2){
        // writes two bfloat16 at once
        *reinterpret_cast<__nv_bfloat162*>(start_k_pages + i) = *reinterpret_cast<__nv_bfloat162*>(start_new_k+i);
        *reinterpret_cast<__nv_bfloat162*>(start_v_pages + i) = *reinterpret_cast<__nv_bfloat162*>(start_new_v+i);
    }
    if(threadIdx.x != 0) return;
    for(int i = aligned; i < head_dim; i++){
        start_k_pages[i] = start_new_k[i];
        start_v_pages[i] = start_new_v[i];
    }

}
void KVCache::prepare_table(int seql){
    for(int b = 0; b < batch_size; b++) {
        for(int t = 0; t < seql; t++) {
            int pos = host_qkv_lens[b] + t;
            int page_idx = pos / PAGE_SIZE;
            if (host_page_table[b * max_pages_per_seq + page_idx] == -1) {
                host_page_table[b * max_pages_per_seq + page_idx] = get_next_free_page();
            }
        }
    }

    cudaMemcpy(page_table, host_page_table,
                (size_t)batch_size * max_pages_per_seq * sizeof(int),
                cudaMemcpyHostToDevice);
}
void KVCache::add_kv(const Tensor& new_k, const Tensor& new_v) {
    
        int seql = new_k.shape[2];
        
        dim3 grid(batch_size, num_heads, seql);
        add_kv_kernel<<<grid, 256>>>(
            (__nv_bfloat16*) new_k.data(), (__nv_bfloat16*) new_v.data(),
            (__nv_bfloat16*) k_pages.data(),     (__nv_bfloat16*) v_pages.data(),
            (int*) page_table,  (int*) qkv_lens,
            batch_size, num_heads, max_pages_per_seq, seql, head_dim);

        for(int b = 0; b < batch_size; b++) host_qkv_lens[b] += seql;
        increment_qkv_lens_kernel<<<1, batch_size>>>((int*)qkv_lens, seql, batch_size);

    }
