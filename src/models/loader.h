#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include "../external/json.hpp"
#include <unordered_map>
using json = nlohmann::json;
namespace fs = std::filesystem;

// had to break down and use AI for the SafeTensorReader:


struct WeightContainer {
    cudaDataType_t dtype;
    std::vector<int> shape;
    size_t start, end;
};

struct SafeTensorReader {
    std::unordered_map<std::string, WeightContainer> weights;
    std::string path_;
    size_t data_base_;
    SafeTensorReader() {}
    SafeTensorReader(const std::string& path) : path_(path) {
        data_base_ = 0;
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open: " + path);

        uint64_t hlen = 0;
        f.read(reinterpret_cast<char*>(&hlen), 8);

        std::string hs(hlen, '\0');
        f.read(hs.data(), (std::streamsize)hlen);
        data_base_ = 8 + hlen;

        json h = json::parse(hs);
        for (auto& [name, info] : h.items()) {
            if (name == "__metadata__") continue;
            WeightContainer m;
            
            m.dtype = SafeTensorReader::sft_to_cuda(info["dtype"].get<std::string>());
            for (auto& d : info["shape"]) m.shape.push_back(d.get<int>());
            m.start = info["data_offsets"][0].get<size_t>();
            m.end = info["data_offsets"][1].get<size_t>();
            // std::cout<< name << " " << info["dtype"].get<std::string>() << m.end << " "<< m.start<< std::endl;
            weights[name] = std::move(m);
        }
    }


  

    Tensor load(const std::string& name, int device) const {
        auto it = weights.find(name);
        if (it == weights.end())
            throw std::runtime_error("tensor not found in " + path_ + ": " + name);
        const WeightContainer& m = it->second;

        size_t nbytes = m.end - m.start;
        std::vector<char> buf(nbytes);
        // std::cout << "LINE 62" << std::endl;
        {
            std::ifstream f(path_, std::ios::binary);
            f.seekg((std::streamoff)(data_base_ + m.start));
            f.read(buf.data(), (std::streamsize)nbytes);
            if (!f) throw std::runtime_error("read error in " + path_ + " for " + name);
        }
        // std::cout << "LINE 69 " << m.dtype << " " << CUDA_R_8F_E4M3 << std::endl;

        Tensor t(m.shape, m.dtype, device);
        t.set_data<char>(buf, nbytes);
        if(t.dtype() == CUDA_R_16F){
            t = t.cast_to(CUDA_R_16BF);
            // std::cout<< t.dtype() << " "<< CUDA_R_16BF << std::endl;
        }
        
#ifdef FP8_AVAILABLE
        if (t.dtype() == CUDA_R_8F_E4M3) {
            std::string scale_name = name + "_scale_inv";
            auto sit = weights.find(scale_name);
            if (sit != weights.end()){
                const WeightContainer& sm = sit->second;
                
                nbytes = sm.end - sm.start;
                
                std::vector<char> buf(nbytes);
                
                {
                    std::ifstream f(path_, std::ios::binary);
                    f.seekg((std::streamoff)(data_base_ + sm.start));
                    f.read(buf.data(), (std::streamsize)nbytes);
                    if (!f) throw std::runtime_error("read error in " + path_ + " for " + name);
                }
                // std::cout << "LINE 99" << std::endl;
                // std::cout << sm.shape[0] << "x" << sm.shape[1] << " " << m.shape[0] << "x" << m.shape[1] << std::endl;
                // Convert F16 → F32 (model stores scales as F16).
                size_t n_elems = nbytes / ((sm.dtype == CUDA_R_16F) ? 2 : 4);
                std::vector<char> scale_f32(n_elems * 4);
                if(sm.dtype == CUDA_R_16F){
                    const __half* src = reinterpret_cast<const __half*>(buf.data());
                    float* dst = reinterpret_cast<float*>(scale_f32.data());
                    for (size_t i = 0; i < n_elems; ++i)
                        dst[i] = __half2float(src[i]);
                }else{
                    std::copy(buf.begin(), buf.end(), scale_f32.begin());
                }
                // MN-major SFB: CUTLASS expects [K/128, N/128] with N fast,
                // but the model stores [N/128, K/128] (K fast) — transpose on load.
                t.update_scale<char>(sm.shape, scale_f32, n_elems * 4);
                
            }
               
                
            
        }
#endif
        return t;
    }

    static cudaDataType_t sft_to_cuda(const std::string& s){
        if(s == "F32") return CUDA_R_32F;
        else if(s == "BF16") return CUDA_R_16BF;
        else if(s == "F16") return CUDA_R_16F;
        else if(s == "F8_E4M3") return CUDA_R_8F_E4M3;
        
    }

};


struct ModelLoader {
    std::unordered_map<std::string, std::string> weight_to_file;
    std::unordered_map<std::string, SafeTensorReader> file_to_shard;
    ModelLoader(const std::string& hf_dir) {
        
        std::string safe_tensor_idx = hf_dir + "/model.safetensors.index.json";
        std::ifstream f(safe_tensor_idx);
        if(f){
            json data = json::parse(f);
            for(auto& [k,v] : data["weight_map"].items()){
                weight_to_file[k] = hf_dir + "/" + v.get<std::string>();
            }
        }else{
            // std::cout << "one file" << std::endl;
            safe_tensor_idx = hf_dir + "/model.safetensors";
            SafeTensorReader rdr(safe_tensor_idx);
            for (auto& [k, v] : rdr.weights){
                weight_to_file[k] = safe_tensor_idx;
            }   
            file_to_shard[safe_tensor_idx] = std::move(rdr);
        }
        
    }

    Tensor load(const std::string& name, int device_) {
        std::string& fl_nm = weight_to_file[name];
        
        auto sf_it = file_to_shard.find(fl_nm);
        
        if (sf_it == file_to_shard.end()) {
            // std::cout << "  opening shard: " << fl_nm << std::endl;
            SafeTensorReader rdr(fl_nm);
            file_to_shard[fl_nm] = std::move(rdr);
            sf_it = file_to_shard.find(fl_nm);
        }
        // std::cout << "loading " << name << std::endl;
        return sf_it->second.load(name, device_);
    }
};
