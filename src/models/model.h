#pragma once
#include "generic_model.h"
#include <memory>
#include <string>

struct AutoConfig {
    static std::unique_ptr<LanguageConfig> from_pretrained(const std::string& hf_dir);
};

struct AutoModel {
    static std::unique_ptr<LanguageModel> from_pretrained(const std::string& hf_dir, int device = 0);
};
