#pragma once
#include "tensor.h"
#include <memory>
#include <string>
#include <vector>

struct KVCache;
struct FlashAttnEngine;

struct LanguageConfig {
    std::string hf_dir;

    virtual ~LanguageConfig() = default;
};

struct LanguageModel {
    virtual ~LanguageModel() = default;

    virtual Tensor forward(const Tensor& input_ids, Tensor& position_ids,
                    std::vector<KVCache>& kvcache,
                    FlashAttnEngine& engine){}
};
