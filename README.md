# Garlic Inference: A simple engine for LLM inference in pure C++ & CUDA

A simple C++ and CUDA inference engine for LLMs. Mostly a playground to test some ideas

Currently supports:

- [x] Flash Attention
- [x] Paged KV cache
- [x] Async load/offload
- [x] Qwen3 dense models
- [ ] Weights in float 8 execution
- [ ] Qwen3 MoE models
- [ ] MoE caching
- [ ] Better profiling
- [ ] Turboquant


Runs a Qwen3 4B model in Bfloat16 precision at 42.5 tokens/second

## BUILD AND RUN:

```bash
cmake -B build -DRUN_TEST=qwen3 && cmake --build build && ./build/runner
```

## ABOUT GARLIC:

![alt text](imgs/garlic.jpg)
