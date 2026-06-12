#pragma once
#include <fstream>
#include <string>
#include <cstdint>
#include <chrono>
#include "cuda-common.cuh"
#include <nvml.h>
#include <thread>
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
    bool use_cpu;
    uint64_t rapl_before, rapl_after, rapl_max;
    std::chrono::time_point<std::chrono::high_resolution_clock> tm_before;
    nvmlDevice_t device;

    uint64_t read_u64(const std::string& path) {
        std::ifstream f(path);
        uint64_t v = 0;
        f >> v;
        return v;
    }


    PowerProfiler(double* joules_ptr, double* time_ptr, double* watt_ptr, nvmlDevice_t device, bool use_cpu = false) : 
    joules_ptr(joules_ptr), time_ptr(time_ptr), watt_ptr(watt_ptr), device(device), use_cpu(use_cpu){
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(5000ms);
        if(use_cpu) rapl_max = read_u64("/sys/class/powercap/intel-rapl:0/max_energy_range_uj");
        tm_before = high_resolution_clock::now();
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_before);
        rapl_before = read_u64("/sys/class/powercap/intel-rapl:0/energy_uj");
    }

    ~PowerProfiler(){
        cudaDeviceSynchronize();
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_after);
        if(use_cpu) rapl_after = read_u64("/sys/class/powercap/intel-rapl:0/energy_uj");
        double cpu_j = 0;
        auto v1 = (duration_cast<microseconds>(high_resolution_clock::now() - tm_before)).count()/(1000.0f * 1000.0f);
        if(use_cpu){
            uint64_t delta = (rapl_after >= rapl_before)
                            ? (rapl_after - rapl_before)
                            : (rapl_max - rapl_before + rapl_after);
            cpu_j = delta / 1e6;
        }
        
        *time_ptr = v1;
        *joules_ptr = cpu_j + ((energy_after - energy_before) / 1000.0);
        *watt_ptr = ((*joules_ptr) / (v1));
    }
};


