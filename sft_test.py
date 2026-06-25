import safetensors
from safetensors import safe_open
# with safe_open("qwen3-4b-fp8/model.safetensors", framework="pt") as f:
with safe_open("qwen3-30b-fp8/model-00001-of-00004.safetensors", framework="pt") as f:
    for k in f.keys():
        # if "mlp.exp" in k:
        #     continue
        # print(k, f.get_tensor(k).shape, f.get_tensor(k).dtype)
        if ".0.mlp.experts.0." in k:
            print(k, f.get_tensor(k).shape, f.get_tensor(k).dtype)
            print(f.get_tensor(k))
            