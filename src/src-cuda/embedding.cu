#include "embedding.cuh"


__global__ void embedding_kernel_f16_f32(
    const uint32_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ input_embeds,
    int embed_dim,
    const int neat){
    
    const __nv_bfloat16* src = w + (long long) input_ids[blockIdx.x] * embed_dim; // take this id in the look up table
    float* dst = input_embeds + (long long) blockIdx.x * embed_dim; // which token to process

    for (int i = threadIdx.x * 4; i + 3 < neat; i += blockDim.x * 4){ // iterate column copy in 4 (better loading)
        float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src + i));
        float2 x2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src + i + 2));
        *reinterpret_cast<float4*>(dst + i) = make_float4(x1.x, x1.y, x2.x, x2.y);
        
    }
    if (threadIdx.x != 0) return;
    // the non neat (should be one thread)
    for (int j = neat; j < embed_dim; j += 1){
        dst[j] = ((float) src[j]);

    }
        
}

__global__ void embedding_kernel_bf16_bf16(
    const uint32_t* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ input_embeds,
    int embed_dim,
    const int neat){

    const __nv_bfloat16* src = w + (long long) input_ids[blockIdx.x] * embed_dim; // take this id in the look up table
    __nv_bfloat16* dst = input_embeds + (long long) blockIdx.x * embed_dim; // which token to process

    for (int i = threadIdx.x * 4; i + 3 < neat; i += blockDim.x * 4){ // iterate column copy in 4 (better loading)
        //TODO  - write and load at once rather than twice
        *reinterpret_cast<__nv_bfloat162*>(dst + i) = *reinterpret_cast<const __nv_bfloat162*>(src + i);
        *reinterpret_cast<__nv_bfloat162*>(dst + i + 2) = *reinterpret_cast<const __nv_bfloat162*>(src + i + 2);
    
        
    }
    if (threadIdx.x != 0) return;
    // the non neat (should be one thread)
    for (int j = neat; j < embed_dim; j += 1){
        dst[j] = (src[j]);

    }
        
}


__global__ void embedding_kernel_f16_f16(
    const uint32_t* __restrict__ input_ids,
    const  __half* __restrict__ w,
     __half* __restrict__ input_embeds,
    int embed_dim,
    const int neat){

    const  __half* src = w + (long long) input_ids[blockIdx.x] * embed_dim; // take this id in the look up table
     __half* dst = input_embeds + (long long) blockIdx.x * embed_dim; // which token to process

    for (int i = threadIdx.x * 4; i + 3 < neat; i += blockDim.x * 4){ // iterate column copy in 4 (better loading)
        //TODO  - write and load at once rather than twice
        *reinterpret_cast< __half2*>(dst + i) = *reinterpret_cast<const  __half2*>(src + i);
        *reinterpret_cast< __half2*>(dst + i + 2) = *reinterpret_cast<const  __half2*>(src + i + 2);
    
        
    }
    if (threadIdx.x != 0) return;
    // the non neat (should be one thread)
    for (int j = neat; j < embed_dim; j += 1){
        dst[j] = src[j];

    }
        
}


#ifdef FP8_AVAILABLE
// adopted from something I saw online once in a dream
__global__ void embedding_kernel_f8_f32(
    const uint32_t* __restrict__ input_ids,
    const __nv_fp8_e4m3* __restrict__ w,
    const float* scale_w, // can be nullptr
    float* __restrict__ input_embeds,
    int embed_dim,
    const int neat){

    const __nv_fp8_e4m3* src = w + (long long) input_ids[blockIdx.x] * embed_dim; // take this id in the look up table
    float* dst = input_embeds + (long long) blockIdx.x * embed_dim; // which token to process
    float scale = 1.0f;
    if(scale_w != nullptr) scale = __ldg(scale_w);
    for (int i = threadIdx.x * 4; i + 3 < neat; i += blockDim.x * 4){ // iterate column copy in 4 (better loading)
        uint32_t raw = *reinterpret_cast<const uint32_t*>(src + i);
        __nv_fp8_e4m3 x1,x2,x3,x4;
        x1.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x2.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x3.__x = raw & 0xFF; // load bottom most 2 bytes
        raw = raw >> 8;
        x4.__x = raw & 0xFF; // load bottom most 2 bytes
        *reinterpret_cast<float4*>(dst + i) = make_float4(((float) x1) * scale, ((float) x2) * scale, ((float) x3) * scale, ((float) x4) * scale);
        
    }
    if (threadIdx.x != 0) return;
    // the non neat (should be one thread)
    for (int j = neat; j < embed_dim; j += 1){
        dst[j] = ((float) src[j]) * scale;

    }
        
}
#endif

void embedding_forward(Tensor& input_embeds, const Tensor& input_ids, const Tensor& w) {
    const int total_num_tokens = input_ids.num_elements();
    const int embed_dim = w.shape[1];
    const int neat = (embed_dim / 4) * 4;
    const int threads = 256;
    const int blocks = total_num_tokens;


    if (input_embeds.dtype() == CUDA_R_32F && w.dtype() == CUDA_R_16BF) {
        
        embedding_kernel_f16_f32<<<blocks, threads, 0, get_compute_stream()>>>(
            static_cast<const uint32_t*>(input_ids.data()),
            static_cast<const __nv_bfloat16*>(w.data()),
            static_cast<float*>(input_embeds.data()),
            embed_dim,
            neat);

    } else if (input_embeds.dtype() == CUDA_R_16BF && w.dtype() == CUDA_R_16BF) {

        embedding_kernel_bf16_bf16<<<blocks, threads, 0, get_compute_stream()>>>(
            static_cast<const uint32_t*>(input_ids.data()),
            static_cast<const __nv_bfloat16*>(w.data()),
            static_cast<__nv_bfloat16*>(input_embeds.data()),
            embed_dim,
            neat);

    } else if (input_embeds.dtype() == CUDA_R_16F && w.dtype() == CUDA_R_16F) {

        embedding_kernel_f16_f16<<<blocks, threads, 0, get_compute_stream()>>>(
            static_cast<const uint32_t*>(input_ids.data()),
            static_cast<const __half*>(w.data()),
            static_cast<__half*>(input_embeds.data()),
            embed_dim,
            neat);
        
    }
}
