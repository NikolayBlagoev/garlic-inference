namespace garlic{
    void gemm_fp8_groupwise_bf16_sm120(
        void* y,
        const void* W,
        float* sfa,
        const void* x,
        float* sfb,
        int N, int K, int M, int L);
}