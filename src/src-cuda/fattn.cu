#include "fattn.cuh"
#include <cmath>

__global__ void fill_int_kernel(int* buf, int val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

FlashAttnEngine::FlashAttnEngine(int batch_size, int num_heads, int num_kv_heads, int head_dim, int max_pages, int max_pages_per_seq) :
    batch_size(batch_size), num_heads(num_heads), num_kv_heads(num_kv_heads), head_dim(head_dim),
    max_pages(max_pages), max_pages_per_seq(max_pages_per_seq) {

    cudaMalloc(&seq_q_buf, (size_t)batch_size * sizeof(int));
}


PagedAttnGraph& FlashAttnEngine::get_graph(int seq_q, cudaDataType_t dtype) {
    if (glob_graph.has_value() && glob_graph->seq_q_param == seq_q) return *glob_graph;

    glob_graph.emplace();
    glob_graph->graph = std::make_shared<CUDNN::graph::Graph>();
    glob_graph->seq_q_param = seq_q;
    CUDNN::graph::Graph& g = *glob_graph->graph;
    if(dtype == CUDA_R_16F){
        g.set_io_data_type(CUDNN::DataType_t::HALF)
        .set_intermediate_data_type(CUDNN::DataType_t::FLOAT)
        .set_compute_data_type(CUDNN::DataType_t::FLOAT);
    } else {
        g.set_io_data_type(CUDNN::DataType_t::BFLOAT16)
        .set_intermediate_data_type(CUDNN::DataType_t::FLOAT)
        .set_compute_data_type(CUDNN::DataType_t::FLOAT);
    }
    

    glob_graph->q = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("q")
        .set_dim({batch_size, num_heads, seq_q, head_dim})
        .set_stride({num_heads*seq_q*head_dim, seq_q*head_dim, head_dim, 1}));

    glob_graph->k = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("k_cache")
        .set_dim({max_pages, num_kv_heads, PAGE_SIZE, head_dim})
        .set_stride({num_kv_heads*PAGE_SIZE*head_dim, PAGE_SIZE*head_dim, head_dim, 1}));

    glob_graph->v = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("v_cache")
        .set_dim({max_pages, num_kv_heads, PAGE_SIZE, head_dim})
        .set_stride({num_kv_heads*PAGE_SIZE*head_dim, PAGE_SIZE*head_dim, head_dim, 1}));

    glob_graph->k_table = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("k_table")
        .set_dim({batch_size, 1, max_pages_per_seq, 1})
        .set_stride({max_pages_per_seq, max_pages_per_seq, 1, 1})
        .set_data_type(CUDNN::DataType_t::INT32));

    glob_graph->v_table = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("v_table")
        .set_dim({batch_size, 1, max_pages_per_seq, 1})
        .set_stride({max_pages_per_seq, max_pages_per_seq, 1, 1})
        .set_data_type(CUDNN::DataType_t::INT32));

    glob_graph->seq_q = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("seq_q")
        .set_dim({batch_size, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(CUDNN::DataType_t::INT32));

    glob_graph->seq_kv = g.tensor(CUDNN::graph::Tensor_attributes()
        .set_name("seq_kv")
        .set_dim({batch_size, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(CUDNN::DataType_t::INT32));

    auto sdpa_attrs = CUDNN::graph::SDPA_attributes()
        .set_name("paged_sdpa")
        .set_is_inference(true)
        .set_causal_mask_bottom_right(true) // this was weird to find out it needs to be bottom right
        .set_padding_mask(true)
        .set_attn_scale(1.0f / sqrtf((float) head_dim))
        .set_paged_attention_k_table(glob_graph->k_table)
        .set_paged_attention_v_table(glob_graph->v_table)
        .set_paged_attention_max_seq_len_kv(max_pages_per_seq*PAGE_SIZE)
        .set_seq_len_q(glob_graph->seq_q)
        .set_seq_len_kv(glob_graph->seq_kv);

    auto [o_t, softmax_stats] = g.sdpa(glob_graph->q, glob_graph->k, glob_graph->v, sdpa_attrs);
    (void)softmax_stats;
    glob_graph->o = o_t;
    if(dtype == CUDA_R_16F){
        glob_graph->o->set_output(true)
        .set_data_type(CUDNN::DataType_t::HALF)
        .set_dim({batch_size, num_heads, seq_q, head_dim})
        .set_stride({num_heads*seq_q*head_dim, seq_q*head_dim, head_dim, 1});
    } else {
        glob_graph->o->set_output(true)
        .set_data_type(CUDNN::DataType_t::BFLOAT16)
        .set_dim({batch_size, num_heads, seq_q, head_dim})
        .set_stride({num_heads*seq_q*head_dim, seq_q*head_dim, head_dim, 1});
    }
    

    CUDNN_FE_CHECK(g.validate());
    CUDNN_FE_CHECK(g.build_operation_graph(cudnn_handle()));
    CUDNN_FE_CHECK(g.create_execution_plans({CUDNN::HeurMode_t::A}));
    CUDNN_FE_CHECK(g.check_support(cudnn_handle()));
    CUDNN_FE_CHECK(g.build_plans(cudnn_handle(), CUDNN::BuildPlanPolicy_t::HEURISTICS_CHOICE));
    CUDNN_FE_CHECK(g.get_workspace_size(glob_graph->workspace_bytes));

    if (glob_graph->workspace_bytes > 0)
        cudaMalloc(&glob_graph->workspace, (size_t)glob_graph->workspace_bytes);

    return *glob_graph;
}

void FlashAttnEngine::run(Tensor& o, const Tensor& q, const KVCache& cache) {
    int seq_q = q.shape[2];
    PagedAttnGraph& pg = get_graph(seq_q, q.dtype());

    if (cached_seq_q != seq_q) {
        fill_int_kernel<<<(batch_size + 255) / 256, 256>>>(
            (int*)seq_q_buf, seq_q, batch_size);
        cached_seq_q = seq_q;
    }
    // std::cout<<"SEQ" << seq_q << std::endl;
    // std::cout<<"K _ VALUES";
    // Tensor::list_values(cache.k_pages, 200);
    std::unordered_map<std::shared_ptr<CUDNN::graph::Tensor_attributes>, void*> vp = {
        {pg.q,       q.data()},
        {pg.k,       cache.k_pages.data()},
        {pg.v,       cache.v_pages.data()},
        {pg.k_table, cache.page_table},
        {pg.v_table, cache.page_table},
        {pg.seq_q,   seq_q_buf},
        {pg.seq_kv,  cache.qkv_lens},
        {pg.o,       o.data()},
    };

    CUDNN_FE_CHECK(pg.graph->execute(cudnn_handle(), vp, pg.workspace));
}
