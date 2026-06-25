#pragma once
#include "tensor.h"
#include <unordered_map>
#include <string.h>
#include <stdexcept>
#include <iostream>
using std::unordered_map;

// doubly linked list for LRU
struct LRU_NODE {
    LRU_NODE* prev;
    LRU_NODE* next;
    std::string nm;
    LRU_NODE(std::string nm) : nm(nm), prev(nullptr), next(nullptr) {}
};

struct TensorEventWrapper{
    cudaEvent_t event = nullptr;
    Tensor t;
    LRU_NODE* node = nullptr;
};
struct MoEManager{
    unordered_map<std::string, TensorEventWrapper> loc_map;

    cudaStream_t _stream, _offload_stream;
    int elements_per_expert;
    LRU_NODE head, tail;
    int curr_size;
    int capacity;

    MoEManager(int capacity):
                capacity(capacity),curr_size(0),
                head(""), tail("") {

        cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&_offload_stream, cudaStreamNonBlocking);
        head.next = &tail;
        tail.prev = &head;


    }

    void inform(const std::string nm, Tensor& t){
        auto it = loc_map.find(nm);
        if (it != loc_map.end()) return;
        TensorEventWrapper tmp;
        tmp.t = t;
        loc_map[nm] = std::move(tmp);
    }

    void prepare(const std::string nm){
        // std::cout<<"preparing "<<nm<<std::endl;
        auto it = loc_map.find(nm);
        if (it == loc_map.end()) return;
        TensorEventWrapper& tew = it->second;
        if(!tew.t.on_cpu()){
            if(!tew.node) return;
            tew.node->prev->next = tew.node->next;
            tew.node->next->prev = tew.node->prev;

            tew.node->prev = &head;
            tew.node->next = head.next;
            head.next->prev = tew.node;
            head.next = tew.node;
            return;
        }

        if(curr_size >= capacity - 2 && tail.prev != &head){
            LRU_NODE* to_evict = tail.prev;
            auto evict_it = loc_map.find(to_evict->nm);
            TensorEventWrapper& tew_evict = evict_it->second;
            tew_evict.t.offloadAsync(_offload_stream);
            // fix both sides of the doubly-linked list
            to_evict->prev->next = &tail;
            tail.prev = to_evict->prev;
            delete tew_evict.node;
            tew_evict.node = nullptr;
            if (tew_evict.event) {
                cudaEventDestroy(tew_evict.event);
                tew_evict.event = nullptr;
            }
            curr_size--;
        }

       
        LRU_NODE* new_node = new LRU_NODE(nm);
        tew.node = new_node;
        new_node->prev = &head;
        new_node->next = head.next;
        head.next->prev = new_node;
        head.next = new_node;

        cudaEventCreate(&tew.event);
        tew.t.onloadAsync(_stream);
        cudaEventRecord(tew.event, _stream);
        curr_size++;
    }

    
    void wait(const std::string& nm, cudaStream_t compute_stream){
        auto it = loc_map.find(nm);
        if (it == loc_map.end() || !it->second.event){
            // std::cout<<"No event for "<<nm<<std::endl;
            return;
        } 
        cudaStreamWaitEvent(compute_stream, it->second.event, 0);
        // std::cout<<"READY TO USE: "<<nm<<std::endl;
    }


};

extern MoEManager* JoseMurinho;
