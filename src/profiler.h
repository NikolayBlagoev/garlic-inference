#pragma once
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;

struct _TIMING_RESULTS{
    long long kv_cache_prepare;
    long long argmax;
    long long embedding;
    long long deembedding;
    long long rope_forward;
    long long casting;
    long long addinplace;
    long long norm;
    long long makecopy;
    long long mlp;
    long long mlp_cast;
    long long mlp_matmul;
    long long mlp_silu;
    long long mlp_elmwise;
    long long mlp_inits;
    long long selfattn;
    long long selfattn_cast;
    long long selfattn_inits;
    long long selfattn_kvcacheadd;
    long long selfattn_projections;
    long long selfattn_rmsnorms;
    long long selfattn_posembs;
    long long selfattn_transpose;
    long long selfattn_attn;
};

extern _TIMING_RESULTS tmrs;

#define TIME_PROFILE(call, ptr) do { \
    auto tm1 = high_resolution_clock::now(); \
    (call); \
    *ptr += duration_cast<microseconds>(high_resolution_clock::now() - tm1).count(); \
} while(0)
