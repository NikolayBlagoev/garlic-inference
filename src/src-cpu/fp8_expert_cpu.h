#pragma once
// ---------------------------------------------------------------------------
// fp8_expert_cpu.h — CPU kernel for "cold" MoE experts (weights not in VRAM).
//
// Pure C++/x86 module: NO CUDA headers. The CUDA-side glue lives in
// cpu_expert_runtime.h. This module computes, for one expert e and T tokens:
//
//      h = bf16( bf16(up(x)) * bf16(silu(gate(x))) )      (matches the GPU
//      y = bf16( down(h) )                                  fused kernel's
//                                                           rounding exactly)
//
// where gate/up are [I, H] and down is [H, I] FP8-E4M3 matrices with
// 128x128 block scales (float), identical layout and scale indexing to
// fp8_grouped_groupwise_* / fp8_gemv_groupwise_* in src/src-cuda/mat-ops.cu:
//
//      scale_row  = n * scale_N / N          (integer division, repo formula)
//      scale(k)   = S[scale_row * scale_K + (k / 128)]
//      fp32 accumulate inside each 128-wide k-group, then acc += group*scale
//
// Core primitive: register-tiled GEMV. One thread owns a contiguous range of
// output rows n; for each row it streams the FP8 row once and FMA-accumulates
// against up to MR=4 activation vectors held in L1 (so multi-token dispatch
// costs ~1x weight bandwidth, not T x). FP8->FP32 decode is branch-free bit
// manipulation (shift into the fp32 exponent field + one multiply by 2^120),
// no lookup table on the hot path.
//
// Parallelism: a persistent worker pool. Each expert is split into row-chunk
// subtasks over two dependent stages (stage 1: fused gate/up rows of I;
// stage 2: down rows of H). Workers steal chunks across *all* in-flight
// experts, so both the "8 cold experts x 1 token" decode case and the
// "1 cold expert" case use every core. Per output row the summation order is
// fixed, so results are bitwise deterministic w.r.t. thread count.
//
// Requirements: x86-64 with AVX2+FMA (runtime-checked; scalar fallback is
// provided for correctness). Optional AVX-512F path if compiled in.
// Weights must remain valid and immutable while jobs are in flight (true in
// garlic-inference: expert host buffers are written once at load and never
// freed or mutated — "offload" merely flips a pointer back to them).
// ---------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <cstddef>

namespace garlic_cpu {

// Host-side view of one expert's packed weights. All pointers reference
// memory owned elsewhere (pinned host buffers + scale mirrors); this struct
// is trivially copyable and immutable after registration.
struct ExpertWeights {
    const uint8_t* w_gate = nullptr;  // [I, H] FP8-E4M3, row-major
    const uint8_t* w_up   = nullptr;  // [I, H]
    const uint8_t* w_down = nullptr;  // [H, I]
    const float*   s_gate = nullptr;  // [sg_rows, sg_cols] f32, row-major
    const float*   s_up   = nullptr;  // [su_rows, su_cols]
    const float*   s_down = nullptr;  // [sd_rows, sd_cols]
    int H = 0;                        // hidden size
    int I = 0;                        // moe intermediate size
    int sg_rows = 0, sg_cols = 0;     // gate scale shape (up must match)
    int su_rows = 0, su_cols = 0;
    int sd_rows = 0, sd_cols = 0;

    bool valid() const {
        return w_gate && w_up && w_down && s_gate && s_up && s_down
            && H > 0 && I > 0
            && sg_rows == su_rows && sg_cols == su_cols;
    }
};

// The k-group width used by the GPU kernels (kGemvSK in mat-ops.cu).
constexpr int kGroupK = 128;

// ---------------------------------------------------------------------------
// One unit of submitted work: the full FFN of one expert over T tokens.
// The caller owns every buffer and must keep them alive until finished()==1.
//   x_bf16 : [T, H] bf16 activations (raw uint16 bit patterns)
//   y_bf16 : [T, H] bf16 output (may alias x_bf16 — stage 2 only reads h)
//   xf     : [T, H] f32 scratch  (filled by submit(): bf16 -> f32 once)
//   h      : [T, I] f32 scratch  (intermediate, values are bf16-rounded)
// ---------------------------------------------------------------------------
struct ExpertJob {
    const ExpertWeights* w = nullptr;
    const uint16_t* x_bf16 = nullptr;
    uint16_t*       y_bf16 = nullptr;
    float*          xf     = nullptr;
    float*          h      = nullptr;
    int T = 0;

    // --- internal scheduling state (managed by ExpertExecutor) ---
    int c1 = 0, c2 = 0;               // chunk counts per stage
    int chunk1 = 0, chunk2 = 0;       // rows per chunk
    int n1_claim = 0, n1_done = 0;    // guarded by executor mutex
    int n2_claim = 0, n2_done = 0;
    std::atomic<int> finished{0};     // 1 when y_bf16 fully written

    bool done() const { return finished.load(std::memory_order_acquire) != 0; }
    void reset() {
        c1 = c2 = chunk1 = chunk2 = 0;
        n1_claim = n1_done = n2_claim = n2_done = 0;
        finished.store(0, std::memory_order_relaxed);
    }
};

// ---------------------------------------------------------------------------
// Persistent fork-join executor with cross-expert chunk stealing.
// submit() is non-blocking; poll job.done() (or wait_all()) for completion.
// Thread-safe for a single producer thread (the inference driver).
// ---------------------------------------------------------------------------
class ExpertExecutor {
public:
    // nthreads <= 0 selects max(1, hardware_concurrency - reserve).
    explicit ExpertExecutor(int nthreads = 0, int reserve = 2);
    ~ExpertExecutor();

    ExpertExecutor(const ExpertExecutor&) = delete;
    ExpertExecutor& operator=(const ExpertExecutor&) = delete;

    // Converts x_bf16 -> job.xf on the calling thread (cheap: T*H elements),
    // then enqueues the two GEMV stages. job must outlive completion.
    void submit(ExpertJob& job);

    // Block until every submitted job has finished (test/shutdown helper;
    // the inference loop polls done() instead).
    void wait_all();

    int num_threads() const { return nthreads_; }

private:
    struct Impl;
    Impl* impl_;
    int nthreads_;
};

// ---------------------------------------------------------------------------
// Single-shot entry points (no pool) — used by tests and as a tiny fallback.
// ---------------------------------------------------------------------------

// Full expert on the calling thread using the best available SIMD path.
void expert_ffn_fp8(const ExpertWeights& w,
                    const uint16_t* x_bf16, uint16_t* y_bf16, int T,
                    float* xf /*[T*H]*/, float* h /*[T*I]*/);

// Scalar double-accumulation reference (ground truth for tests).
void expert_ffn_fp8_ref(const ExpertWeights& w,
                        const uint16_t* x_bf16, uint16_t* y_bf16, int T);

// --- exposed for unit tests -------------------------------------------------
float    fp8_e4m3_to_float(uint8_t b);   // software decode (handles NaN)
float    bf16_to_f32(uint16_t b);
uint16_t f32_to_bf16(float f);           // round-to-nearest-even
const char* active_isa();                // "avx512" | "avx2" | "scalar"

} // namespace garlic_cpu
