#include "model.h"
#include "qwen3/qwen3.h"
#include "qwen3_moe/qwen3_moe.h"
#include "../external/json.hpp"
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

std::unique_ptr<LanguageConfig> AutoConfig::from_pretrained(const std::string& hf_dir) {
    std::ifstream f(hf_dir + "/config.json");
    json cfg = json::parse(f);
    std::string architecture = cfg.at("architectures").at(0).get<std::string>();

    if (architecture.find("Qwen3MoeForCausalLM") != std::string::npos){
        return std::make_unique<Qwen3MoeConfig>(Qwen3MoeConfig::from_pretrained(hf_dir));
    }
    if (architecture.find("Qwen3ForCausalLM") != std::string::npos){
        return std::make_unique<Qwen3Config>(Qwen3Config::from_pretrained(hf_dir));
    }
    
}

std::unique_ptr<LanguageModel> AutoModel::from_pretrained(const std::string& hf_dir, int device) {
    std::unique_ptr<LanguageConfig> config = AutoConfig::from_pretrained(hf_dir);

    if (auto* moe_config = dynamic_cast<Qwen3MoeConfig*>(config.get())){
        return Qwen3Moe::from_pretrained(*moe_config, device);
    }
    if (auto* qwen3_config = dynamic_cast<Qwen3Config*>(config.get())){
        return Qwen3::from_pretrained(*qwen3_config, device);
    }
    
}
