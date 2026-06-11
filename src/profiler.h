#pragma once
#include <chrono>
#include "cuda-common.cuh"
#include <nvml.h>

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
    long long misc;
};

extern _TIMING_RESULTS tmrs;

#define TIME_PROFILE(call, ptr) do { \
    auto tm1 = high_resolution_clock::now(); \
    (call); \
    *ptr += duration_cast<microseconds>(high_resolution_clock::now() - tm1).count(); \
} while(0)

struct PowerProfiler{
    double *joules_ptr, *time_ptr, *watt_ptr;
    unsigned long long energy_before, energy_after;
    std::chrono::time_point<std::chrono::high_resolution_clock> tm_before;
    nvmlDevice_t device;
    PowerProfiler(double* joules_ptr, double* time_ptr, double* watt_ptr, nvmlDevice_t device) : 
    joules_ptr(joules_ptr), time_ptr(time_ptr), watt_ptr(watt_ptr), device(device){
        tm_before = high_resolution_clock::now();
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_before);
    }

    ~PowerProfiler(){
        cudaDeviceSynchronize();
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_after);
        auto v1 = (duration_cast<microseconds>(high_resolution_clock::now() - tm_before)).count()/(1000.0f * 1000.0f);
        *time_ptr = v1;
        *joules_ptr = (energy_after - energy_before) / 1000.0;
        *watt_ptr = ((energy_after - energy_before) / (v1*1000.0));
    }
};


