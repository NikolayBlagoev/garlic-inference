cmake_minimum_required(VERSION 3.18)
project(nbml CUDA CXX)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
if("${RUN_TEST}" STREQUAL "qwen3")
    # cmake -B build -DRUN_TEST=qwen3 && cmake --build build && ./build/runner
    add_executable(runner
    qwen_test.cpp
    src/models/qwen3.cpp
    src/tokenizers/bpe.cpp
    src/src-cuda/simple-ops.cu
    src/src-cuda/argmax.cu
    src/src-cuda/mat-ops.cu
    src/src-cuda/tensor.cu
    src/src-cuda/norm.cu
    src/src-cuda/rope.cu
    src/src-cuda/fattn.cu
    src/src-cuda/embedding.cu
    src/src-cuda/activations.cu
    src/src-cuda/kvcache.cu
    
)
endif()


# cmake -B build -DRUN_TEST=qwen3 && cmake --build build && ./build/runner
set_target_properties(runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_link_libraries(runner PRIVATE cublas cublasLt cudnn nvrtc)
target_include_directories(runner PRIVATE src src/src-cuda src/tokenizers src/models src/external src/external/cudnn-frontend/include ${CUDAToolkit_INCLUDE_DIRS})