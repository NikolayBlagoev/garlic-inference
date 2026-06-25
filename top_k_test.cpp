#include <cuda_runtime.h>
#include "cuda-common.cuh"
#include "argmax.cuh"




#include <thread>
int main() {
    int k = 4;
    Tensor t({1,40}, CUDA_R_16BF, 0);
    std::vector<__nv_bfloat16> data = {0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9,0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9,0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9,0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9};
    cudaMemcpy(t.data(), data.data(), (size_t)40 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    Tensor out({1,k}, CUDA_R_32U, 0);
    Tensor scores({1,k}, CUDA_R_32F, 0);
    topk(t, out, scores, k);
    Tensor::list_values(out,k);
    Tensor::list_values(scores,k);
    return 0;
}