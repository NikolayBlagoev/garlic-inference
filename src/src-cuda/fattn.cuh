#pragma once
#include "cuda-common.cuh"
#include "../kvcache.h"
#include <cudnn_frontend.h>
#include <optional>
#include <unordered_map>
#include <memory>

namespace CUDNN = cudnn_frontend;

#define CUDNN_FE_CHECK(expr) do { \
    auto _status = (expr); \
    if (_status.is_bad()) { \
        fprintf(stderr, "cuDNN FE error at %s:%d: %s\n", \
                __FILE__, __LINE__, _status.get_message().c_str()); \
        exit(1); \
    } \
} while(0)

struct PagedAttnGraph {
    int seq_q_param;
    std::shared_ptr<CUDNN::graph::Graph> graph;
    void* workspace = nullptr;
    int64_t workspace_bytes = 0;

    std::shared_ptr<CUDNN::graph::Tensor_attributes> q, k, v;
    std::shared_ptr<CUDNN::graph::Tensor_attributes> k_table, v_table;
    std::shared_ptr<CUDNN::graph::Tensor_attributes> seq_q, seq_kv;
    std::shared_ptr<CUDNN::graph::Tensor_attributes> o;
};


struct FlashAttnEngine {
    int batch_size, num_heads, num_kv_heads, head_dim, max_pages, max_pages_per_seq;

    void* seq_q_buf = nullptr;
    int cached_seq_q = -1;
    std::optional<PagedAttnGraph> glob_graph;

    FlashAttnEngine() = default;
    explicit FlashAttnEngine(int batch_size, int num_heads, int num_kv_heads, int head_dim, int max_pages, int max_pages_per_seq);
    FlashAttnEngine(const FlashAttnEngine&) = delete;

    void run(Tensor& o, const Tensor& q, const KVCache& cache);
    PagedAttnGraph& get_graph(int seq_q, cudaDataType_t dtype);
};
