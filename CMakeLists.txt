cmake_minimum_required(VERSION 3.18)
project(nbml CUDA CXX)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


file(GLOB_RECURSE MODELS ${CMAKE_SOURCE_DIR}/src/models/*.cpp)
file(GLOB_RECURSE CUDA_KERNELS ${CMAKE_SOURCE_DIR}/src/src-cuda/*.cu)
file(GLOB_RECURSE CPU_KERNELS ${CMAKE_SOURCE_DIR}/src/src-cpu/*.cpp)

add_executable(runner
        inference.cpp
        src/tokenizers/bpe.cpp
        ${MODELS}
        ${CUDA_KERNELS}
        ${CPU_KERNELS}
)

# export PATH=/usr/local/cuda-13.2/bin/:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64/:$LD_LIBRARY_PATH
# cmake -B build -DRUN_TEST=qwen3 && cmake --build build && ./build/runner
set_target_properties(runner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "120a"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)



# maximum optimziation to c++ code
target_compile_options(runner PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)

target_link_libraries(runner PRIVATE cublas cublasLt cudnn nvrtc pthread nvidia-ml)
target_compile_options(runner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>)


# there's some warnings... Annoying me...
target_compile_options(runner PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-w>
    $<$<COMPILE_LANGUAGE:CUDA>:-w>)


file(GLOB_RECURSE _header_files
    ${CMAKE_SOURCE_DIR}/src/models/*.h ${CMAKE_SOURCE_DIR}/src/models/*.hpp
    ${CMAKE_SOURCE_DIR}/src/src-cuda/*.cuh ${CMAKE_SOURCE_DIR}/src/src-cuda/*.h)
set(SUBDIR_INCLUDE_DIRS "")
foreach(_hdr ${_header_files})
    get_filename_component(_dir ${_hdr} DIRECTORY)
    list(APPEND SUBDIR_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES SUBDIR_INCLUDE_DIRS)

target_include_directories(runner PRIVATE
    src src/src-cuda src/tokenizers src/models
    ${SUBDIR_INCLUDE_DIRS}
    src/moe-managers
    src/src-cpu
    src/external
    src/external/cudnn-frontend/include
    ${CUDAToolkit_INCLUDE_DIRS})