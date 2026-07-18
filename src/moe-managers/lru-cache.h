#pragma once
#include "moe-cache.h"

struct LRU_MoEManager : MoEManager{
    unordered_map<std::string, TensorEventWrapper> loc_map;
    cudaStream_t _stream;
    int elements_per_expert;
    LIST_NODE head, tail;
    int curr_size;
    int capacity;
    void* gpu_slot = nullptr;

    LRU_MoEManager(int capacity):
                capacity(capacity),curr_size(0),
                head(""), tail("") {
        cudaStreamCreateWithPriority(&_stream, cudaStreamNonBlocking,-3);
        
        head.next = &tail;
        tail.prev = &head;
        
    }

    void inform(const std::string nm, Tensor& t, int layer) override {
        auto it = loc_map.find(nm);
        if (it != loc_map.end()) return;
        TensorEventWrapper tmp;
        tmp.t = t;
        tmp.layer = layer;
        loc_map[nm] = std::move(tmp);
    }

    void prepare(const std::string nm, float val = 0, const std::vector<int>& to_ignore = {}) override {
        
        auto it = loc_map.find(nm);
        if (it == loc_map.end()) return;
        TensorEventWrapper& tew = it->second;
        if(!tew.t.on_cpu()){
            if(!tew.node) return;
            // put at front:
            tew.node->prev->next = tew.node->next;
            tew.node->next->prev = tew.node->prev;

            tew.node->prev = &head;
            tew.node->next = head.next;
            head.next->prev = tew.node;
            head.next = tew.node;
            return;
        }
        // cpu resident, need to evict Least Recently Used
        if(curr_size >= capacity && tail.prev != &head){
            LIST_NODE* to_evict = tail.prev;
            auto evict_it = loc_map.find(to_evict->nm);
            TensorEventWrapper& tew_evict = evict_it->second;

           
            void* freed_gpu = tew_evict.t.offloadAsync(_stream, nullptr);
            

            gpu_slot = freed_gpu;
            to_evict->prev->next = &tail;
            tail.prev = to_evict->prev;
            to_evict->prev = nullptr;
            to_evict->next = nullptr;

            curr_size--;
        }

        if(tew.node == nullptr) tew.node = new LIST_NODE(nm);
        
        tew.node->prev = &head;
        tew.node->next = head.next;
        head.next->prev = tew.node;
        head.next = tew.node;

        if(tew.event == nullptr) cudaEventCreateWithFlags(&tew.event, cudaEventDisableTiming);

        void* freed_cpu = tew.t.onloadAsync(_stream, gpu_slot);
        
        cudaEventRecord(tew.event, _stream);
        curr_size++;
    }

    void wait(const std::string& nm, cudaStream_t compute_stream) override {
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){
            
            return;
        } 
        cudaStreamWaitEvent(compute_stream, it->second.event, 0);
    
    }

    bool is_done(const std::string& nm) override {
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){

            return true;
        } 
        return cudaEventQuery(it->second.event) == cudaSuccess;
       
    }
};