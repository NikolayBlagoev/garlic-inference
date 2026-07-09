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
        // cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
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

struct MRU_MoEManager : MoEManager{
    unordered_map<std::string, TensorEventWrapper> loc_map;
    cudaStream_t _stream;
    int elements_per_expert;
    LIST_NODE head, tail;
    int curr_size;
    int capacity;
    int num_layers;
    void* gpu_slot = nullptr;

    MRU_MoEManager(int capacity, int num_layers):
                capacity(capacity),num_layers(num_layers),curr_size(0),
                head(""), tail("") {
        cudaStreamCreateWithPriority(&_stream, cudaStreamNonBlocking,-3);
        // cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
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
            if(to_evict == nullptr) to_evict = tail.prev; 
            auto evict_it = loc_map.find(to_evict->nm);
            TensorEventWrapper& tew_evict = evict_it->second;

           
            void* freed_gpu = tew_evict.t.offloadAsync(_stream, nullptr);
            

            gpu_slot = freed_gpu;
            to_evict->next->prev = to_evict->prev;
            to_evict->prev->next = to_evict->next;
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

        tew.t.onloadAsync(_stream, gpu_slot);
        gpu_slot = nullptr;
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

extern MoEManager* JoseMurinho;
