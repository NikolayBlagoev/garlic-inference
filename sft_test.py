import safetensors
from safetensors import safe_open
with safe_open("qwen3-4b-fp8/model.safetensors", framework="pt") as f:
    for k in f.keys():
        print(k, f.get_tensor(k).shape, f.get_tensor(k).dtype)
        if "scale" in k:
            print(f.get_tensor(k))
            exit()