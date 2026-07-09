#pragma once
#include "tensor.h"
#include <unordered_map>
#include <string.h>
#include <stdexcept>
#include <limits>
#include <iostream>
using std::unordered_map;

// doubly linked list for LRU/LFU
struct LIST_NODE {
    LIST_NODE* prev;
    LIST_NODE* next;
    float val;
    int layer;
    std::string nm;
    LIST_NODE(std::string nm) : nm(nm), prev(nullptr), next(nullptr), val(0) {}
    LIST_NODE(std::string nm, float val) : nm(nm), prev(nullptr), next(nullptr), val(val) {}
};

struct TensorEventWrapper{
    cudaEvent_t event = nullptr;
    Tensor t;
    int layer;
    float val = 0;
    bool in_transit = false;
    LIST_NODE* node = nullptr;
};
struct MemBuffer {
    void* ptr = nullptr;
    cudaEvent_t ready = nullptr;
};
// abstract class:
struct MoEManager {
    MoEManager(int capacity){}
    MoEManager(){}
    virtual void inform(const std::string nm, Tensor& t, int layer){}
    virtual void prepare(const std::string nm, float val = 0, const std::vector<int>& to_ignore = {}){}
    virtual void wait(const std::string& nm, cudaStream_t compute_stream){}
    virtual bool is_done(const std::string& nm){}
};

struct LRU_MoEManager : MoEManager{
    unordered_map<std::string, TensorEventWrapper> loc_map;
    cudaStream_t _stream, _offload_stream;
    int elements_per_expert;
    LIST_NODE head, tail;
    int curr_size;
    int capacity;
    MemBuffer gpu_pool[2];
    MemBuffer cpu_pool[2];
    int gpu_slot = 0, cpu_slot = 0;

    LRU_MoEManager(int capacity):
                capacity(capacity),curr_size(0),
                head(""), tail("") {
        cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
        head.next = &tail;
        tail.prev = &head;
        for (int i = 0; i < 2; i++) {
            cudaEventCreateWithFlags(&gpu_pool[i].ready, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&cpu_pool[i].ready, cudaEventDisableTiming);
        }
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
        if(curr_size >= capacity - 2 && tail.prev != &head){
            LIST_NODE* to_evict = tail.prev;
            auto evict_it = loc_map.find(to_evict->nm);
            TensorEventWrapper& tew_evict = evict_it->second;

            MemBuffer& dst = cpu_pool[cpu_slot];
            cudaStreamWaitEvent(_offload_stream, dst.ready, 0);
            void* freed_gpu = tew_evict.t.offloadAsync(_offload_stream, dst.ptr);
            cudaEventRecord(dst.ready, _offload_stream);

            gpu_pool[gpu_slot].ptr = freed_gpu;
            cudaEventRecord(gpu_pool[gpu_slot].ready, _offload_stream);

            cpu_slot ^= 1;
            gpu_slot ^= 1;
            
            // fix both sides of the doubly-linked list
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
        MemBuffer& dst = gpu_pool[gpu_slot];
        cudaStreamWaitEvent(_stream, dst.ready, 0);
        void* freed_cpu = tew.t.onloadAsync(_stream, dst.ptr);
        cudaEventRecord(dst.ready, _stream);

        cpu_pool[cpu_slot].ptr = freed_cpu;
        cudaEventRecord(cpu_pool[cpu_slot].ready, _stream);

        
            
        
        // last_freed_gpu_buffer = nullptr;
        cudaEventRecord(tew.event, _stream);
        curr_size++;
    }

    void wait(const std::string& nm, cudaStream_t compute_stream) override {
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){
            // std::cout<<"No event for "<<nm<<std::endl;
            return;
        } 
        cudaStreamWaitEvent(compute_stream, it->second.event, 0);
        // std::cout<<"READY TO USE: "<<nm<<std::endl;
    }

    bool is_done(const std::string& nm) override {
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){

            return true;
        } 
        return cudaEventQuery(it->second.event) == cudaSuccess;
       
    }
};



struct MRU_MoEManager : MoEManager{
    unordered_map<std::string, TensorEventWrapper> loc_map;
    cudaStream_t _stream, _offload_stream;
    int elements_per_expert;
    LIST_NODE head, tail;
    int curr_size;
    int capacity;
    MemBuffer gpu_pool[2];
    MemBuffer cpu_pool[2];
    int gpu_slot = 0, cpu_slot = 0;
    int num_layers = 0;
    MRU_MoEManager(int capacity, int num_layers):
                capacity(capacity),curr_size(0),
                head(""), tail(""), num_layers(num_layers) {
        cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
        head.next = &tail;
        tail.prev = &head;
        head.val = std::numeric_limits<float>::max();
        for (int i = 0; i < 2; i++) {
            cudaEventCreateWithFlags(&gpu_pool[i].ready, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&cpu_pool[i].ready, cudaEventDisableTiming);
        }
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
        // cpu resident, need to evict Most Recently Used
        if(curr_size >= capacity - 2 && tail.prev != &head){
            LIST_NODE* start_search = tail.prev;
            LIST_NODE* to_evict = nullptr;
            int best_distance = 0;
            while(start_search != &head){
                if(std::find(to_ignore.begin(), to_ignore.end(), start_search->layer) == to_ignore.end()){
                    int dist = (start_search->layer - tew.layer + num_layers) % num_layers;
                    if(dist > best_distance){
                        to_evict = start_search;
                        best_distance = dist;
                    }
                }
                start_search = start_search->prev;
            }
            if(to_evict == nullptr) to_evict = tail.prev; // fallback: evict LRU
            auto evict_it = loc_map.find(to_evict->nm);
            TensorEventWrapper& tew_evict = evict_it->second;

            MemBuffer& dst = cpu_pool[cpu_slot];
            cudaStreamWaitEvent(_offload_stream, dst.ready, 0);
            void* freed_gpu = tew_evict.t.offloadAsync(_offload_stream, dst.ptr);
            cudaEventRecord(dst.ready, _offload_stream);

            gpu_pool[gpu_slot].ptr = freed_gpu;
            cudaEventRecord(gpu_pool[gpu_slot].ready, _offload_stream);

            cpu_slot ^= 1;
            
            // fix both sides of the doubly-linked list
            to_evict->prev->next = &tail;
            tail.prev = to_evict->prev;
            to_evict->prev = nullptr;
            to_evict->next = nullptr;

            curr_size--;
        }

        if(tew.node == nullptr) tew.node = new LIST_NODE(nm);
        tew.node->layer = tew.layer;

        tew.node->prev = &head;
        tew.node->next = head.next;
        head.next->prev = tew.node;
        head.next = tew.node;

        if(tew.event == nullptr) cudaEventCreateWithFlags(&tew.event, cudaEventDisableTiming);
        MemBuffer& dst = gpu_pool[gpu_slot];
        cudaStreamWaitEvent(_stream, dst.ready, 0);
        void* freed_cpu = tew.t.onloadAsync(_stream, dst.ptr);
        cudaEventRecord(dst.ready, _stream);

        cpu_pool[cpu_slot].ptr = freed_cpu;
        cudaEventRecord(cpu_pool[cpu_slot].ready, _stream);

        gpu_slot ^= 1;
            
        
        // last_freed_gpu_buffer = nullptr;
        cudaEventRecord(tew.event, _stream);
        curr_size++;
    }

    void wait(const std::string& nm, cudaStream_t compute_stream) override {
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){
            // std::cout<<"No event for "<<nm<<std::endl;
            return;
        } 
        cudaStreamWaitEvent(compute_stream, it->second.event, 0);
        // std::cout<<"READY TO USE: "<<nm<<std::endl;
    }

    bool is_done(const std::string& nm) override {
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){

            return true;
        } 
        return cudaEventQuery(it->second.event) == cudaSuccess;
       
    }
};
// struct LFU_MoEManager : MoEManager{
//     unordered_map<std::string, TensorEventWrapper> loc_map;
//     cudaStream_t _stream, _offload_stream;
//     int elements_per_expert;
//     LIST_NODE head, tail;
//     int curr_size;
//     int capacity;

//     LFU_MoEManager(int capacity):
//                 capacity(capacity),curr_size(0),
//                 head(""), tail("") {
//         cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
//         cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
//         head.next = &tail;
//         head.val = std::numeric_limits<float>::max();
//         tail.prev = &head;
//     }

//     void inform(const std::string nm, Tensor& t, int layer) override {
//         auto it = loc_map.find(nm);
//         if (it != loc_map.end()) return;
//         TensorEventWrapper tmp;
//         tmp.t = t;
//         tmp.layer = layer;
//         loc_map[nm] = std::move(tmp);
//     }

//     void prepare(const std::string nm, float val = 0, const std::vector<int>& to_ignore = {}) override {
        
//         auto it = loc_map.find(nm);
//         if (it == loc_map.end()) return;
//         TensorEventWrapper& tew = it->second;
//         tew.val += val;
//         if(!tew.t.on_cpu()){
//             if(!tew.node) return;
//             tew.node->val = tew.val;
//             tew.node->prev->next = tew.node->next;
//             tew.node->next->prev = tew.node->prev;
//             LIST_NODE* tmp = tew.node->prev;
//             while(tmp->val < tew.node->val){
//                 tmp = tmp->prev;
//             }
//             tew.node->prev = tmp;
//             tew.node->next = tmp->next;
//             tmp->next->prev = tew.node;
//             tmp->next = tew.node;
            
//             return;
//         }
//         // cpu resident, need to evict Least Frequently Used subject to ignore list
//         if(curr_size >= capacity - 1 && tail.prev != &head){
//             LIST_NODE* to_evict = tail.prev;
//             while(to_evict != &head && std::find(to_ignore.begin(), to_ignore.end(), to_evict->layer) != to_ignore.end()){
//                 to_evict = to_evict->prev;
//             }
//             auto evict_it = loc_map.find(to_evict->nm);
//             TensorEventWrapper& tew_evict = evict_it->second;
//             tew_evict.t.offloadAsync(_offload_stream);
//             // fix both sides of the doubly-linked list
//             to_evict->prev->next = to_evict->next;
//             to_evict->next->prev = to_evict->prev;
            
//             to_evict->prev = nullptr;
//             to_evict->next = nullptr;
//             curr_size--;
//         }

//         if(tew.node == nullptr) tew.node = new LIST_NODE(nm);
//         tew.node->val = tew.val;
//         tew.node->layer = tew.layer;
//         LIST_NODE* tmp = tail.prev;
//         while(tmp->val < tew.node->val){
//             tmp = tmp->prev;
//         }
//         tew.node->prev = tmp;
//         tew.node->next = tmp->next;
//         tmp->next->prev = tew.node;
//         tmp->next = tew.node;

//         if(tew.event == nullptr) cudaEventCreateWithFlags(&tew.event, cudaEventDisableTiming);
//         tew.t.onloadAsync(_stream);
//         cudaEventRecord(tew.event, _stream);
//         curr_size++;
//     }

//     void wait(const std::string& nm, cudaStream_t compute_stream) override {
//         auto it = loc_map.find(nm);
//         if (it == loc_map.end() || !it->second.event){
//             // std::cout<<"No event for "<<nm<<std::endl;
//             return;
//         } 
//         cudaStreamWaitEvent(compute_stream, it->second.event, 0);
//         // std::cout<<"READY TO USE: "<<nm<<std::endl;
//     }

//     bool is_done(const std::string& nm) override {
//         auto it = loc_map.find(nm);
//         if (it == loc_map.end() || !it->second.event){
//             // std::cout<<"No event for "<<nm<<std::endl;
//             return true;
//         } 
//         return cudaEventQuery(it->second.event) == cudaSuccess;
//         // cudaStreamWaitEvent(compute_stream, it->second.event, 0);
//         // std::cout<<"READY TO USE: "<<nm<<std::endl;
//     }
// };

// Belady-optimal (MRU-by-layer) cache: evict the GPU-resident expert whose
// next possible use is furthest away — i.e., (expert_layer − cur_layer + L) % L
// is maximised. Ties broken by lowest use count (LFU as secondary key).
// struct MRU_MoEManager : MoEManager {
//     unordered_map<std::string, TensorEventWrapper> loc_map;
//     unordered_map<std::string, int> use_count;
//     cudaStream_t _stream, _offload_stream;
//     int curr_size = 0, capacity, num_layers;
//     void* last_freed_gpu_buffer = nullptr;
//     std::vector<void*> _pending_cpu_free;

//     MRU_MoEManager(int capacity, int num_layers)
//         : capacity(capacity), num_layers(num_layers) {
//         cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
//         cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
//     }

//     void inform(const std::string nm, Tensor& t, int layer) override {
//         if (loc_map.count(nm)) return;
//         TensorEventWrapper tmp; tmp.t = t; tmp.layer = layer;
//         loc_map[nm] = std::move(tmp);
//         use_count[nm] = 0;
//     }

//     std::string pick_evict(int cur_layer, const std::vector<int>& to_ignore) {
//         std::string best; int best_dist = -1, best_freq = INT_MAX;
//         for (auto& [nm, tew] : loc_map) {
//             if (tew.t.on_cpu()) continue;
//             if (std::find(to_ignore.begin(), to_ignore.end(), tew.layer) != to_ignore.end()) continue;
//             int dist = (tew.layer - cur_layer + num_layers) % num_layers;
//             int freq = use_count[nm];
//             if (dist > best_dist || (dist == best_dist && freq < best_freq))
//                 best_dist = dist, best_freq = freq, best = nm;
//         }
//         return best;
//     }

//     void prepare(const std::string nm, float val = 0, const std::vector<int>& to_ignore = {}) override {
//         auto it = loc_map.find(nm);
//         if (it == loc_map.end()) return;
//         TensorEventWrapper& tew = it->second;
//         use_count[nm]++;
//         if (!tew.t.on_cpu()) return;

//         if (curr_size >= capacity) {
//             // Safe point to free old CPU staging buffers: all prior onload copies
//             // must be done before we can evict+reuse that GPU slot anyway.
//             cudaStreamSynchronize(_stream);
//             for (void* p : _pending_cpu_free) cudaFreeHost(p);
//             _pending_cpu_free.clear();

//             std::string evict_nm = pick_evict(tew.layer, to_ignore);
//             if (!evict_nm.empty()) {
//                 last_freed_gpu_buffer = loc_map[evict_nm].t.offloadAsync(_offload_stream, nullptr);
//                 curr_size--;
//             }
//         }

//         if (!tew.event) cudaEventCreateWithFlags(&tew.event, cudaEventDisableTiming);
//         void* old_cpu = tew.t.onloadAsync(_stream, last_freed_gpu_buffer);
//         last_freed_gpu_buffer = nullptr;
//         if (old_cpu) _pending_cpu_free.push_back(old_cpu);
//         cudaEventRecord(tew.event, _stream);
//         curr_size++;
//     }

//     void wait(const std::string& nm, cudaStream_t compute_stream) override {
//         auto it = loc_map.find(nm);
//         if (it == loc_map.end() || !it->second.event) return;
//         cudaStreamWaitEvent(compute_stream, it->second.event, 0);
//     }

//     bool is_done(const std::string& nm) override {
//         auto it = loc_map.find(nm);
//         if (it == loc_map.end() || !it->second.event) return true;
//         return cudaEventQuery(it->second.event) == cudaSuccess;
//     }

//     ~MRU_MoEManager() {
//         cudaStreamSynchronize(_stream);
//         for (void* p : _pending_cpu_free) cudaFreeHost(p);
//     }
// };

extern MoEManager* JoseMurinho;
