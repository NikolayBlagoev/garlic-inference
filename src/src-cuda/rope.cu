#include "rope.cuh"

__global__ void _rope_init(float* __restrict__ A, float base, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float double_N = (float)(N * 2);
    if (i + 3 < N) {
        float i1 = 1.0f / __powf(base, (i + 0) * 2.0f / double_N);
        float i2 = 1.0f / __powf(base, (i + 1) * 2.0f / double_N);
        float i3 = 1.0f / __powf(base, (i + 2) * 2.0f / double_N);
        float i4 = 1.0f / __powf(base, (i + 3) * 2.0f / double_N);
        *reinterpret_cast<float4*>(A + i) = make_float4(i1, i2, i3, i4);
    } else {
        for (; i < N; i++)
            A[i] = 1.0f / __powf(base, i * 2.0f / double_N);
    }
}

void initialize_rope(float base, Tensor& t) {
    int N = t.num_elements();
    int threads = 256;
    int blocks = (N / 4 + threads - 1) / threads;
    _rope_init<<<blocks, threads>>>(static_cast<float*>(t._data->data), base, N);
}



//   def forward(self, x, position_ids):
//      inv_freq_expanded = self.inv_freq[None, :, None].float().expand(B, -1, 1).to(x.device)
//      position_ids_expanded = position_ids[:, None, :].float()
//      device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
//      with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
//          freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
//          emb = torch.cat((freqs, freqs), dim=-1)
//          cos = emb.cos() * self.attention_scaling
//          sin = emb.sin() * self.attention_scaling
//      return cos.to(dtype=x.dtype()), sin.to(dtype=x.dtype())

//
// cos_out and sin_out are [B, S, head_dim]

__global__ void rope_emb_kernel(
    float* __restrict__ cos_out,
    float* __restrict__ sin_out,
    const uint32_t* __restrict__   position_ids,
    const float* __restrict__ inv_freq,
    float attention_factor, int B, 
    int S, int head_dim){

    int half_point = head_dim / 2;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int lane = idx & 31;
    int warp = idx >> 5;

    if (warp >= B * S) return;
    // view position ids as B * S
    int seq_idx = warp % S;
    int batch_idx = warp / S;
    int offset = batch_idx * S + seq_idx;
    int cs_offset = offset * head_dim;

    for (int i = lane; i < half_point; i += 32) {
        float cos_v, sin_v;

        __sincosf((float)position_ids[offset] * inv_freq[i], &sin_v, &cos_v);
        cos_v *= attention_factor;
        sin_v *= attention_factor;

        // Since we concatenate we repeat at + N:
        // emb = torch.cat((freqs, freqs), dim=-1)
        // cos = emb.cos() * self.attention_scaling
        // sin = emb.sin() * self.attention_scaling

        cos_out[cs_offset + i] = cos_v;
        cos_out[cs_offset + i + half_point] = cos_v;
        sin_out[cs_offset + i] = sin_v;
        sin_out[cs_offset + i + half_point] = sin_v;
    }
}



void rope_forward(
    Tensor& x,
    Tensor& position_ids,
    Tensor& inv_freq,
    Tensor& cos_emb,
    Tensor& sin_emb){
    int B = x.shape[0];
    int S = x.shape[1];
    // head_dim = 2 * inv_freq.shape[0] since inv_freq holds [head_dim/2] freqs
    int head_dim = 2 * inv_freq.shape[0];


    int threads = 256;
    int warps = threads / 32;
    int blocks = (position_ids.num_elements() + warps - 1) / warps;
    rope_emb_kernel<<<blocks, threads>>>(
                    static_cast<float*>(cos_emb.data()),
                    static_cast<float*>(sin_emb.data()),
                    static_cast<const uint32_t*>(position_ids.data()), 
                    static_cast<const float*>(inv_freq.data()), 
                    1.0f, B, S, head_dim);
    
    
}

// cos/sin are always float; x can be float or bfloat16
template<typename T>
__global__ void apply_rotary_pos_emb_kernel(
    T* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int B, int S, int num_heads, int head_dim){
    int half_point = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= B * num_heads * S * half_point) return;

    int b_idx = (int) (idx / (S * num_heads * half_point));
    int s_idx = (int) (idx / (num_heads * half_point)) % S;
    int h_idx = (int) (idx / (half_point)) % num_heads;
    int d_idx = idx % half_point;

    long long offset = ((b_idx*S + s_idx) * num_heads + h_idx) * head_dim + d_idx;
    float x1 = (float) x[offset];
    float x2 = (float) x[offset + half_point];
    float c = cos[(b_idx * S + s_idx) * head_dim + d_idx];
    float s = sin[(b_idx * S + s_idx) * head_dim + d_idx];
    x[offset]            = (T)(x1 * c - x2 * s);
    x[offset + half_point] = (T)(x2 * c + x1 * s);
}

void apply_rotary_pos_emb(Tensor& x, const Tensor& cos, const Tensor& sin){
    int B = x.shape[0];
    int S = x.shape[1];
    int num_heads = x.shape[2];
    int head_dim = x.shape[3];
    int threads = 256;
    int blocks = (x.num_elements() / 2 + threads - 1) / threads;
    const float* cos_ptr = (const float*) cos.data();
    const float* sin_ptr = (const float*) sin.data();
    if (x.dtype() == CUDA_R_32F){
        apply_rotary_pos_emb_kernel<float><<<blocks,threads>>>(
            (float*) x.data(), cos_ptr, sin_ptr, B, S, num_heads, head_dim);
    } else if (x.dtype() == CUDA_R_16BF){
        apply_rotary_pos_emb_kernel<__nv_bfloat16><<<blocks,threads>>>(
            (__nv_bfloat16*) x.data(), cos_ptr, sin_ptr, B, S, num_heads, head_dim);
    }
}