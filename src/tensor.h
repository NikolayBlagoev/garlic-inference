#pragma once
#include "cuda-common.cuh"

#include <vector>
#include <numeric>
#include <stdexcept>
#ifdef FP8_AVAILABLE
#include <cuda_fp8.h>
#endif
#include <cuda_fp16.h>
#include <iostream>
#include <memory>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " — " << cudaGetErrorString(err) << "\n"; \
        std::exit(1); \
    } \
} while(0)

static cudaStream_t load_offload_stream = nullptr;
static cudaStream_t get_load_offload_stream(){
        if(!load_offload_stream){
            cudaStreamCreateWithFlags(&load_offload_stream, cudaStreamNonBlocking);
        }
        return load_offload_stream;
}


struct DataView {
    void* data;
    float* scale; // need this for scaling in case of float8
    float scale_val = 1.0f;
    bool on_cpu = false;
    bool initialized = false;
    long long num_elements;
    int device;
    cudaDataType_t dtype;

    DataView(){

    };

    DataView(cudaDataType_t dtype, long long num_elements, int device = 0, bool on_cpu = false, bool initialized = true, float scale_val = 1.0f) : 
        dtype(dtype), device(device), on_cpu(on_cpu), initialized(initialized), scale_val(scale_val), num_elements(num_elements){
        if(!on_cpu && initialized){
            cudaSetDevice(device);
            cudaMalloc(&data, num_elements * element_size(dtype));
            cudaMalloc(&scale, sizeof(float));
            cudaMemcpy(scale, &scale_val, sizeof(float), cudaMemcpyHostToDevice);
        }else if(on_cpu && initialized){
            data = malloc(num_elements * element_size(dtype));
            scale = &scale_val;
        }
    }

    // refactor to constructor?
    void gpu_initialize(){
        if(data || initialized) return;
        initialized = true;
        on_cpu = false;
        cudaMalloc(&data, num_elements * element_size(dtype));
        cudaMalloc(&scale, sizeof(float));
        cudaMemcpy(scale, &scale_val, sizeof(float), cudaMemcpyHostToDevice);
    }

    void update_scale(){
        if(on_cpu){
            scale_val = *scale;
            scale = &scale_val;
        }else{
            float* tmp = (float*) malloc(sizeof(float));
            cudaMemcpy(tmp,scale, sizeof(float), cudaMemcpyDeviceToHost);
            scale_val = *tmp;
            ::free(tmp);
        }
    }

    void update_scale(float val){
        scale_val = val;
        if(on_cpu){
            scale = &scale_val; // might be redundant tbh
        }else{
            cudaMemcpy(scale, &scale_val, sizeof(float), cudaMemcpyHostToDevice);
        }
            
    }

    void free(){
        if (!data) return;
        if (on_cpu) {        
            ::free(data);
                
        } else {
            cudaSetDevice(device);
            cudaFreeAsync(data, get_load_offload_stream());
            cudaFreeAsync(scale, get_load_offload_stream());
        }
        data = nullptr;
        scale = nullptr;
    }
    template<typename T>
    void set_data(std::vector<T>& buffer, size_t n_bytes){
        cudaSetDevice(device);
        cudaMemcpy(data, buffer.data(), n_bytes, cudaMemcpyHostToDevice);
    }
    ~DataView(){
        free();
    }

    static DataView make_like(DataView& o){
        return DataView(o.dtype, o.num_elements, o.device, o.on_cpu, o.initialized, o.scale_val);
    }

    static size_t element_size(cudaDataType_t dtype) {
            switch (dtype) {
                case CUDA_R_8I:
                case CUDA_R_8U:          return 1;
#ifdef FP8_AVAILABLE
                case CUDA_R_8F_E4M3:
                case CUDA_R_8F_E5M2:     return 1;
#endif
                case CUDA_R_16F:
                case CUDA_R_16BF:
                case CUDA_R_16I:
                case CUDA_R_16U:         return 2;
                case CUDA_R_32F:
                case CUDA_R_32I:
                case CUDA_R_32U:         return 4;
                case CUDA_R_64F:
                case CUDA_R_64I:
                case CUDA_R_64U:         return 8;
                default: throw std::runtime_error("unknown dtype");
            }
    }

};


// *For now support - float8, bfloat16, float32, and uint32*
// hurm... one day I might need non-contigious tensors...
// this will require restructuring of other stuff in the codebase
struct Tensor {
        std::shared_ptr<DataView>  _data;
        std::vector<int>        shape;
        // TODO: IN THE FUTURE I WANT STRIDE ADDED TOO TO SUPPORT DIFFERENT VIEWS OF THE SAME TENSOR
        // TODO: WHAT IF NOT CONTIGIOUS VIEW?

        // Default constructor
        Tensor() {}

        Tensor(std::vector<int> shape, DataView dv) : shape(shape){
            _data = std::make_shared<DataView>(dv);
            
        }

        Tensor(std::vector<int> shape, cudaDataType_t dtype, int device = 0, bool on_cpu = false, bool initialized = true, float scale_val = 1.0f)
            : shape(shape) {
            _data = std::make_shared<DataView>(dtype, num_elements(), device, on_cpu, initialized, scale_val);
        }

        // shallow copy operation
        // TODO: Revise when I have CPU offloading
        Tensor(const Tensor& o) {
            if (this != &o) {
                _data = o._data; 
                shape = o.shape; 
            }
        }


        // Ownership transfer :/
        // TODO: Revise when I have CPU offloading
        Tensor(Tensor&& o) noexcept {
            if (this != &o) {
                _data = o._data; 
                shape = std::move(o.shape); 
            }
        }

        Tensor& operator=(const Tensor& o) {
            if (this != &o) {
                _data = o._data;
                shape = o.shape;
            }
            return *this;
        }

        // Ownership transfer :/
        // TODO: Revise when I have CPU offloading
        Tensor& operator=(Tensor&& o) noexcept {
            if (this != &o) {
                _data = o._data; 
                shape = std::move(o.shape); 
            }
            return *this;
        }


        template<typename T>
        void set_data(std::vector<T>& buffer, size_t n_bytes){
            _data->set_data<T>(buffer, n_bytes);
        }


        void* data() const {
            return _data->data;
        }

        void gpu_initialize(){
            _data->gpu_initialize();
        }


        void free() {
            _data = nullptr;
        }

        ~Tensor(){
            free();
        }

        int num_elements() const {
            if(shape.empty()) return 0;
            int n = 1;
            for(int d : shape) n *= d;
            return n;
        }

        int ndim() const { return (int)shape.size(); }

        Tensor cast_to(cudaDataType_t new_dtype) const;
        
        cudaDataType_t dtype() const {
            return _data->dtype;
        }

        float* scale() const {
            return _data->scale;
        }

        float scale_val() const {
            return _data->scale_val;
        }

        bool initialized() const {
            return _data->initialized;
        }

        bool on_cpu() const {
            return _data->on_cpu;
        }

        int device() const {
            return _data->device;
        }

        static void list_values(const Tensor& t, int elms = -1){
            if(elms == -1){
                elms = t.num_elements();
            }
            if(t.dtype() == CUDA_R_32F){
                std::vector<float> h(elms);
                cudaMemcpy(h.data(), t._data->data, elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);
                
                for(int i = 0; i < elms; i++){
                    
                    std::cout << h[i]<<" ";
                }
                std::cout << std::endl;
            }else if(t.dtype() == CUDA_R_16BF){
                std::vector<__nv_bfloat16> h(elms);
                cudaMemcpy(h.data(), t._data->data, elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);
                
                for(int i = 0; i < elms; i++){
                    
                    std::cout << (float)h[i]<<" ";
                }
                std::cout << std::endl;

            }else if(t.dtype() == CUDA_R_8F_E4M3){
                std::vector<uint8_t> h(elms);
                cudaMemcpy(h.data(), t._data->data, elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);
                __nv_fp8_e4m3 fp8val;
                for(int i = 0; i < elms; i++){
                    fp8val.__x = h[i];
                    std::cout << (float)fp8val<<" ";
                }
                std::cout << std::endl;
            }else if(t.dtype() == CUDA_R_32U){
                std::vector<uint32_t> h(elms);
                cudaMemcpy(h.data(), t._data->data, elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);
                
                for(int i = 0; i < elms; i++){
                    
                    std::cout << h[i]<<" ";
                }
                std::cout << std::endl;
            }
        }

        Tensor make_copy(){
            if(!_data) return Tensor();
            Tensor t(shape, dtype(), device(), on_cpu(), initialized());
            if(!initialized()) return t;
            size_t bytes = (size_t) num_elements() * Tensor::element_size(dtype());
            if (on_cpu()) {
                memcpy(t._data->data, _data->data, bytes);
            } else {
                cudaSetDevice(device());
                CUDA_CHECK(cudaMemcpy(t._data->data, _data->data, bytes, cudaMemcpyDeviceToDevice));
            }
            return t;
        }

        void copy_into(Tensor& dst) {
            if (!_data || !initialized()) return;
            size_t bytes = (size_t) num_elements() * Tensor::element_size(dtype());
            if (!dst._data || !dst.initialized() || dst.num_elements() != num_elements() || dst.dtype() != dtype()) {
                dst = Tensor(shape, dtype(), device(), false, true);
            } else {
                dst.shape = shape; // after some refactoring shape now works as a view :)
            }
            cudaSetDevice(device());
            CUDA_CHECK(cudaMemcpy(dst._data->data, _data->data, bytes, cudaMemcpyDeviceToDevice));
        }

        void update_scale(float val){
            _data->update_scale(val);
        }

        void update_scale(){
            _data->update_scale();
        }
        

        static size_t element_size(cudaDataType_t dtype) {
            switch (dtype) {
                case CUDA_R_8I:
                case CUDA_R_8U:          return 1;
#ifdef FP8_AVAILABLE
                case CUDA_R_8F_E4M3:
                case CUDA_R_8F_E5M2:     return 1;
#endif
                case CUDA_R_16F:
                case CUDA_R_16BF:
                case CUDA_R_16I:
                case CUDA_R_16U:         return 2;
                case CUDA_R_32F:
                case CUDA_R_32I:
                case CUDA_R_32U:         return 4;
                case CUDA_R_64F:
                case CUDA_R_64I:
                case CUDA_R_64U:         return 8;
                default: throw std::runtime_error("unknown dtype");
            }
        }
};


