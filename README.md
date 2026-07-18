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
cmake -B build -DRUN_TEST=qwen3-moe && cmake --build build && ./build/runner
```

## ABOUT GARLIC:

![alt text](imgs/garlic.jpg)
