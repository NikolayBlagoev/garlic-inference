#pragma once
#include <cuda_runtime.h>
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
        {
            std::ifstream f(path_, std::ios::binary);
            f.seekg((std::streamoff)(data_base_ + m.start));
            f.read(buf.data(), (std::streamsize)nbytes);
            if (!f) throw std::runtime_error("read error in " + path_ + " for " + name);
        }

        Tensor t(m.shape, m.dtype, device);
        t.set_data<char>(buf, nbytes);

        return t;
    }

    static cudaDataType_t sft_to_cuda(const std::string& s){
        if(s == "F32") return CUDA_R_32F;
        else if(s == "BF16") return CUDA_R_16BF;
        else if(s == "F8_E4M3") return CUDA_R_8F_E4M3;
        
    }

};


struct ModelLoader {
    std::unordered_map<std::string, std::string> weight_to_file;
    std::unordered_map<std::string, SafeTensorReader> file_to_shard;
    ModelLoader(const std::string& hf_dir) {
        std::string safe_tensor_idx = hf_dir + "/model.safetensors.index.json";
        std::ifstream f(safe_tensor_idx);
        json data = json::parse(f);
        for(auto& [k,v] : data["weight_map"].items()){
            weight_to_file[k] = hf_dir + "/" + v.get<std::string>();
        }
    }

    Tensor load(const std::string& name, int device_) {
        std::string& fl_nm = weight_to_file[name];
        
        auto sf_it = file_to_shard.find(fl_nm);
        
        if (sf_it == file_to_shard.end()) {
            std::cout << "  opening shard: " << fl_nm << "\n";
            SafeTensorReader rdr(fl_nm);
            file_to_shard[fl_nm] = std::move(rdr);
            sf_it = file_to_shard.find(fl_nm);
        }
        return sf_it->second.load(name, device_);
    }
};
