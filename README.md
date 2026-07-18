# Garlic Inference: A simple engine for LLM inference in pure C++ & CUDA

A simple C++ and CUDA inference engine for LLMs. Mostly a playground to test some ideas

Currently supports:

- [x] Flash Attention
- [x] Paged KV cache
- [x] Async load/offload
- [x] Qwen3 dense models
- [x] Weights in float 8 execution
- [X] Qwen3 MoE models
- [X] MoE caching
- [x] Better profiling
- [ ] Turboquant


## Some performance reports:

|                    | RTX 5060ti    | RTX 4090      |
| ------------------ | ------------- | ------------- |
| Qwen3 30B A3B fp8  | 53.3 tok/s    | 81.3 tok/s    |


## BUILD AND RUN:

```bash
cmake -B build && cmake --build build
./build/runner [MODEL DIRECTORY]
```

You should download a model first (for now only Qwen3 and Qwen3 MoE models are supported). I recommend trying with: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

## ABOUT GARLIC:

![alt text](imgs/garlic.jpg)
