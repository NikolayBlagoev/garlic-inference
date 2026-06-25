#pragma once
#include "cuda-common.cuh"

#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <mutex>
#ifdef FP8_AVAILABLE
#include <cuda_fp8.h>
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

extern cudaStream_t load_offload_stream;
static cudaStream_t get_load_offload_stream(){
        if(!load_offload_stream){
            cudaStreamCreateWithFlags(&load_offload_stream, cudaStreamNonBlocking);
        }
        return load_offload_stream;
}
extern cudaStream_t compute_stream;
static cudaStream_t get_compute_stream(){
        if(!compute_stream){
            cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
        }
        return compute_stream;
}

extern cudaStream_t secondary_compute_stream;
static cudaStream_t get_secondary_stream(){
        if(!secondary_compute_stream){
            cudaStreamCreateWithFlags(&secondary_compute_stream, cudaStreamNonBlocking);
        }
        return secondary_compute_stream;
}

struct PinnedMemPool {

    std::vector<void*> free_list;
    int allocated, max_el;
    size_t buf_bytes;

    PinnedMemPool(int n, size_t bytes) : buf_bytes(bytes), allocated(0), max_el(n) {
        free_list.reserve(n);
        
    }

    void* acquire(cudaStream_t stream) {
        if(allocated < max_el){
            allocated += 1;
            void* p;
            cudaHostAlloc(&p, buf_bytes, cudaHostAllocDefault);
            cudaStreamSynchronize(stream);
            return p;
        }
        if (free_list.empty()) return nullptr;
        void* p = free_list.back();
        free_list.pop_back();
        return p;
    }

    void release(void* p) {
        
        free_list.push_back(p);
    }

    ~PinnedMemPool() {
        for (void* p : free_list) cudaFreeHost(p);
    }
};

extern PinnedMemPool* g_expert_pool;

struct DataView {
    void* data;
    bool on_cpu = false;
    bool initialized = false;
    long long num_elements;
    int device;
    bool pinned = false;
    cudaDataType_t dtype;

    DataView(){

    };

    DataView(cudaDataType_t dtype, long long num_elements, int device = 0, bool on_cpu = false, bool initialized = true) : 
        dtype(dtype), device(device), on_cpu(on_cpu), initialized(initialized), num_elements(num_elements){
        if(!on_cpu && initialized){
            cudaSetDevice(device);
            cudaMalloc(&data, num_elements * element_size(dtype));
        }else if(on_cpu && initialized){
            
            cudaMallocManaged(&data, num_elements * element_size(dtype), cudaMemAttachGlobal);
            // data = std::malloc(num_elements * element_size(dtype));
            // if (!data)
            //     throw std::runtime_error("malloc failed for CPU tensor");
        }
    }

    // refactor to constructor?
    void gpu_initialize(){
        if(data || initialized) return;
        initialized = true;
        on_cpu = false;
        cudaMalloc(&data, num_elements * element_size(dtype));
    }

    void offloadAsync(cudaStream_t stream){
        // if(!data || !initialized || on_cpu) return;
        // void* tmp = nullptr;
        // if (g_expert_pool) {
        //     tmp = g_expert_pool->acquire(stream);
        //     if (tmp){
        //         pinned = true;
        //     }
        // }
        
        // if(!tmp){
        //     pinned = false;
        //     tmp = std::malloc(num_elements * element_size(dtype));
        //     if (!tmp)
        //         throw std::runtime_error("malloc failed for CPU tensor");
        // }
        on_cpu = true;
        cudaMemLocation loc{ cudaMemLocationTypeHost, 0 };
        cudaError_t err = cudaMemPrefetchAsync(data, num_elements * element_size(dtype), loc, 0, stream);
        if (err != cudaSuccess) std::fprintf(stderr, "[offloadAsync] cudaMemPrefetchAsync: %s\n", cudaGetErrorString(err));
        // cudaMemcpyAsync(tmp, data, num_elements * element_size(dtype), cudaMemcpyDeviceToHost, stream);
        // cudaFreeAsync(data, stream);
        // data = tmp;
    }

    void onloadAsync(cudaStream_t stream){
        cudaMemLocation loc{ cudaMemLocationTypeDevice, device };
        cudaError_t err = cudaMemPrefetchAsync(data, num_elements * element_size(dtype), loc, 0, stream);
        if (err != cudaSuccess) std::fprintf(stderr, "[onloadAsync] cudaMemPrefetchAsync (device=%d): %s\n", device, cudaGetErrorString(err));
        // if(!data || !initialized || !on_cpu) return;
        // size_t sz = num_elements * element_size(dtype);
        // void* gpu_buf;
        // cudaMallocAsync(&gpu_buf, sz, stream);
        // cudaMemcpyAsync(gpu_buf, data, sz, cudaMemcpyHostToDevice, stream);
        // if(pinned){
        //     cudaLaunchHostFunc(stream, [](void* arg){ g_expert_pool->release(arg); }, data);
        // }else{
        //     cudaLaunchHostFunc(stream, [](void* arg){ std::free(arg); }, data);
        // }
        
        // data = gpu_buf;
        on_cpu = false;
    }

    void free(){
        if (!data) return;
        if (on_cpu) {
            std::free(data);
        } else {
            cudaSetDevice(device);
            cudaFreeAsync(data, get_load_offload_stream());
            
        }
        data = nullptr;
    }
    template<typename T>
    void set_data(std::vector<T>& buffer, size_t n_bytes, int offset = 0){
        if(on_cpu){
            cudaMemcpy(data + offset, buffer.data(), n_bytes, cudaMemcpyHostToHost);
        }else{
            cudaSetDevice(device);
            cudaMemcpy(data + offset, buffer.data(), n_bytes, cudaMemcpyHostToDevice);
        }
        
        
    }
    ~DataView(){
        free();
    }

    static DataView make_like(DataView& o){
        return DataView(o.dtype, o.num_elements, o.device, o.on_cpu, o.initialized);
    }


    static void list_values(const DataView& v, int elms = -1){
        if(elms == -1){
            elms = v.num_elements;
        }
        if(v.dtype == CUDA_R_32F){
            std::vector<float> h(elms);
            cudaMemcpy(h.data(), v.data, elms * DataView::element_size(v.dtype), cudaMemcpyDeviceToHost);

            for(int i = 0; i < elms; i++){

                std::cout << h[i]<<" ";
            }
            std::cout << std::endl;
        }
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
        std::shared_ptr<DataView>  _scale;
        std::vector<int>            shape;
        std::vector<int>            shape_scale;
        uint64_t                    offset;
        // TODO: IN THE FUTURE I WANT STRIDE ADDED TOO TO SUPPORT DIFFERENT VIEWS OF THE SAME TENSOR
        // TODO: WHAT IF NOT CONTIGIOUS VIEW?

        // Default constructor
        Tensor() {}

        Tensor(std::vector<int> shape, DataView dv, uint64_t offset = 0) : shape(shape), offset(offset){
            _data = std::make_shared<DataView>(dv);
            
        }

        Tensor(std::vector<int> shape, std::shared_ptr<DataView> dv, uint64_t offset = 0) : shape(shape), offset(offset){
            _data = dv;
        }



        Tensor(std::vector<int> shape, cudaDataType_t dtype, int device = 0, bool on_cpu = false, bool initialized = true, uint64_t offset = 0)
            : shape(shape), offset(offset) {
            _data = std::make_shared<DataView>(dtype, num_elements(), device, on_cpu, initialized);
        }

        // shallow copy operation
        // TODO: Revise when I have CPU offloading
        Tensor(const Tensor& o) {
            if (this != &o) {
                _data = o._data; 
                _scale = o._scale; 
                shape = o.shape; 
                shape_scale = o.shape_scale;
                offset = o.offset;
            }
        }


        // Ownership transfer :/
        // TODO: Revise when I have CPU offloading
        Tensor(Tensor&& o) noexcept {
            if (this != &o) {
                _data = o._data; 
                _scale = o._scale;
                shape = std::move(o.shape); 
                shape_scale = std::move(o.shape_scale);
                offset = o.offset;
            }
        }

        Tensor& operator=(const Tensor& o) {
            if (this != &o) {
                _data = o._data;
                _scale = o._scale;
                shape = o.shape;
                shape_scale = o.shape_scale;
                offset = o.offset;
            }
            return *this;
        }

        // Ownership transfer :/
        // TODO: Revise when I have CPU offloading
        Tensor& operator=(Tensor&& o) noexcept {
            if (this != &o) {
                _data = o._data; 
                _scale = o._scale;
                shape = std::move(o.shape); 
                shape_scale = std::move(o.shape_scale);
                offset = o.offset;
            }
            return *this;
        }


        template<typename T>
        void set_data(std::vector<T>& buffer, size_t n_bytes){
            _data->set_data<T>(buffer, n_bytes);
        }


        void* data() const {
            return static_cast<char*>(_data->data) + offset;
        }

        void gpu_initialize(){
            _data->gpu_initialize();
        }


        void free() {
            _data = nullptr;
            _scale = nullptr;
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
                cudaMemcpy(h.data(), t.data(), elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);

                for(int i = 0; i < elms; i++){

                    std::cout << h[i]<<" ";
                }
                std::cout << std::endl;
            }else if(t.dtype() == CUDA_R_16BF){
                std::vector<__nv_bfloat16> h(elms);
                cudaMemcpy(h.data(), t.data(), elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);

                for(int i = 0; i < elms; i++){

                    std::cout << (float)h[i]<<" ";
                }
                std::cout << std::endl;

            }else if(t.dtype() == CUDA_R_16F){
                std::vector<__half> h(elms);
                cudaMemcpy(h.data(), t.data(), elms * Tensor::element_size(t.dtype()),  cudaMemcpyDeviceToHost);

                for(int i = 0; i < elms; i++){

                    std::cout << (float)h[i]<<" ";
                }
                std::cout << std::endl;

            }else if(t.dtype() == CUDA_R_8F_E4M3){
                std::vector<uint8_t> h(elms);
                cudaMemcpy(h.data(), t.data(), elms * Tensor::element_size(t.dtype()), t.on_cpu() ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost);
                __nv_fp8_e4m3 fp8val;
                for(int i = 0; i < elms; i++){
                    fp8val.__x = h[i];
                    std::cout << (float)fp8val<<" ";
                }
                std::cout << std::endl;
            }else if(t.dtype() == CUDA_R_32U){
                std::vector<uint32_t> h(elms);
                cudaMemcpy(h.data(), t.data(), elms * Tensor::element_size(t.dtype()), cudaMemcpyDeviceToHost);

                for(int i = 0; i < elms; i++){

                    std::cout << h[i]<<" ";
                }
                std::cout << std::endl;
            }
        }

        Tensor make_copy(){
            if(!_data) return Tensor();
            Tensor t(shape, dtype(), device(), on_cpu(), initialized(), offset);
            if(!initialized()) return t;
            size_t bytes = (size_t) num_elements() * Tensor::element_size(dtype());
            if (on_cpu()) {
                memcpy(t._data->data, _data->data, bytes);
            } else {
                cudaSetDevice(device());
                CUDA_CHECK(cudaMemcpyAsync(t._data->data, _data->data, bytes, cudaMemcpyDeviceToDevice, get_compute_stream()));
            }
            return t;
        }

        void copy_into(Tensor& dst) {
            if (!_data || !initialized()) return;
            size_t bytes = (size_t) num_elements() * Tensor::element_size(dtype());
            if (!dst._data || !dst.initialized() || dst.num_elements() != num_elements() || dst.dtype() != dtype()) {
                dst = Tensor(shape, dtype(), device(), false, true, 0);
            } else {
                dst.shape = shape;
                dst.offset = 0;
            }
            cudaSetDevice(device());
            CUDA_CHECK(cudaMemcpyAsync(dst._data->data, data(), bytes, cudaMemcpyDeviceToDevice, get_compute_stream()));
        }

        void update_scale(std::vector<int> shape, DataView dv){
            _scale = std::make_shared<DataView>(dv);
            shape_scale = shape;
        }

        void update_scale(std::vector<int> shape){
            shape_scale = shape;
            _scale = std::make_shared<DataView>(CUDA_R_32F, num_elements_scale(), device(), false, initialized());
        }
        // Returns true if a new allocation was made (caller should zero extra rows if needed).
        bool lazy_update_scale(std::vector<int> new_shape){
            if (_scale && shape_scale == new_shape) return false;
            update_scale(new_shape);
            return true;
        }
        // template<typename T>
        // void set_data(std::vector<T>& buffer, size_t n_bytes){
        //     _data->set_data<T>(buffer, n_bytes);
        // }
        template<typename T>
        void update_scale(std::vector<int> shape, std::vector<T>& buffer, size_t n_bytes){
            shape_scale = shape;
            _scale = std::make_shared<DataView>(CUDA_R_32F, num_elements_scale(), device(), false, initialized());
            _scale->set_data<T>(buffer, n_bytes);

        }

        float* scale() const{
            if(!_scale) return nullptr;
            return (float*) _scale->data;
        }

        int ndim_scale() const { return (int)shape_scale.size(); }
        
        int num_elements_scale() const {
            if(shape_scale.empty()) return 0;
            int n = 1;
            for(int d : shape_scale) n *= d;
            return n;
        }
        
        void offloadAsync(cudaStream_t stream){
            _data->offloadAsync(stream);
        }

        void onloadAsync(cudaStream_t stream){
            _data->onloadAsync(stream);
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


