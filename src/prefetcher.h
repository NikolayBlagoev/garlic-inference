#pragma once
#include "tensor.h"
#include <unordered_map>
#include <string.h>

struct TensorEventWrapper{
    cudaEvent_t event;
    Tensor t;
}
struct PrefetchWorker{
    unordered_map<std::string, TensorEventWrapper> loc_map;
    cudaStream_t _stream = nullptr;
    PrefetchWorker(){
        cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
    }
    void prefetch(std::string nm, Tensor& t){
        auto it = loc_map.find(name);
        if (it == loc_map.end()){
            TensorEventWrapper tmp;
            tmp
            loc_map 
        }
    }

    Tensor get(std::string nm)
};

extern static PrefetchWorker PREFETCHER;