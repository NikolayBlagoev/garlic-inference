#pragma once
// ---------------------------------------------------------------------------
// cpu_expert_runtime.h — bridges garlic-inference's MoE forward pass to the
// pure-CPU cold-expert kernel (fp8_expert_cpu.h).
//
// Data flow per layer (decode step):
//
//   moe_gather ─► [host sync of expert_offsets — already in the repo]
//        │
//        ├─ hot experts (weights in VRAM / in flight) ─► GPU fused GEMV path
//        │
//        └─ cold experts (Tensor::on_cpu() == true, n_e <= threshold):
//             1. D2H  gathered[rows(e)]  ─► pinned x_stage      (xfer stream)
//             2. cudaStreamSynchronize(xfer)  — tens of µs for a few KB
//             3. ExpertExecutor::submit()  — workers chew rows in parallel,
//                reading FP8 weights DIRECTLY from the pinned host buffers
//                the loader already created (zero weight movement)
//             4. main thread polls job.done() inside the same loop that polls
//                GPU expert readiness; on completion: H2D y_stage ─► the same
//                gathered rows (xfer stream) + cudaEventRecord
//             5. finish_layer(): compute stream waits every H2D event
//        ▼
//   moe_scatter  (unchanged — weighted combine over all slots)
//
// Why this is safe:
//  * The pinned host copy of an expert is written once at load and NEVER
//    mutated or freed (offloadAsync just flips the data pointer back to it),
//    so CPU reads race nothing — even if the LRU manager is simultaneously
//    H2D-ing the same buffer to warm the cache for future tokens.
//  * Block scales live ONLY in VRAM (Tensor::update_scale hardcodes
//    on_cpu=false), so register_expert() mirrors them to host once.
//  * CPU results land in rows of `gathered` disjoint from every GPU expert's
//    output rows; the compute stream is fenced on the H2D events before
//    moe_scatter reads them.
//
// Env knobs:
//   GARLIC_CPU_MOE=0            disable (default: enabled)
//   GARLIC_CPU_MOE_THRESHOLD=N  max tokens per expert for the CPU path (8)
//   GARLIC_CPU_MOE_WARM=0       don't also prefetch CPU-computed experts (1)
//   GARLIC_CPU_THREADS=N        worker count (hardware_concurrency - 2)
//   GARLIC_CPU_ISA=...          cap kernel ISA (debug)
// ---------------------------------------------------------------------------

#include "../tensor.h"
#include "fp8_expert_cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

struct CpuMoeRuntime {
    // ---- configuration -----------------------------------------------------
    int H = 0, I = 0, L = 0, E = 0;
    int  token_threshold = 8;
    bool enabled = true;
    bool warm_cache = false;    // also prefetch CPU-handled experts for later

    garlic_cpu::ExpertExecutor* exec = nullptr;

    // ---- per-expert host views ---------------------------------------------
    std::vector<garlic_cpu::ExpertWeights> views;   // [L*E]
    std::vector<std::vector<float>> scale_store;    // owns host scale mirrors

    // ---- per-layer working state --------------------------------------------
    static constexpr int kMaxSlots = 4;
    struct Slot {
        garlic_cpu::ExpertJob job;
        int         expert     = -1;
        uint32_t    row_begin  = 0;
        int         n_e        = 0;
        bool        staged     = false;   // D2H issued, job not yet submitted
        bool        h2d_issued = false;
        cudaEvent_t ev         = nullptr;
    };
    Slot slots[kMaxSlots];
    int  n_slots = 0;

    uint16_t* x_stage = nullptr;          // pinned [kMaxSlots * threshold * H]
    uint16_t* y_stage = nullptr;          // pinned [kMaxSlots * threshold * H]
    std::vector<float> xf_scratch;        // [kMaxSlots * threshold * H]
    std::vector<float> h_scratch;         // [kMaxSlots * threshold * I]
    cudaStream_t xfer_stream = nullptr;

    // -------------------------------------------------------------------------
    static int env_int(const char* k, int dflt) {
        const char* v = std::getenv(k);
        return v ? std::atoi(v) : dflt;
    }

    void init(int H_, int I_, int L_, int E_) {
        H = H_; I = I_; L = L_; E = E_;
        enabled         = true;
        token_threshold = 8;
        if (!enabled) return;

        views.assign((size_t)L * E, garlic_cpu::ExpertWeights{});
        scale_store.clear();
        scale_store.reserve((size_t)L * E * 3);

        int nthr = 0;
        exec = new garlic_cpu::ExpertExecutor(nthr, /*reserve=*/2);

        const size_t stage_elems = (size_t)kMaxSlots * token_threshold * H;
        CUDA_CHECK(cudaHostAlloc((void**)&x_stage, stage_elems * sizeof(uint16_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&y_stage, stage_elems * sizeof(uint16_t), cudaHostAllocDefault));
        xf_scratch.resize(stage_elems);
        h_scratch.resize((size_t)kMaxSlots * token_threshold * I);
        cudaStreamCreateWithPriority(&xfer_stream, cudaStreamNonBlocking, -2);
        for (int i = 0; i < kMaxSlots; ++i)
            cudaEventCreateWithFlags(&slots[i].ev, cudaEventDisableTiming);
        std::printf("[cpu-moe] enabled: %d workers, isa=%s, threshold=%d, warm=%d\n",
                    exec->num_threads(), garlic_cpu::active_isa(),
                    token_threshold, (int)warm_cache);
    }

    // Called once per expert right after loading (weights may be GPU- or
    // host-resident; only host-resident ones become CPU-eligible).
    void register_expert(int layer, int e,
                         const Tensor& g, const Tensor& u, const Tensor& d) {
        if (!enabled) return;
#ifdef FP8_AVAILABLE
        if (g.dtype() != CUDA_R_8F_E4M3 || u.dtype() != CUDA_R_8F_E4M3 ||
            d.dtype() != CUDA_R_8F_E4M3) return;
#else
        return;
#endif
        // Host base of the packed gate|up|down allocation. Stable forever:
        // on_cpu -> data IS the pinned host buffer; after any onload, that
        // same pointer is retained as pin_memory.
        DataView* dv = g._data.get();
        const uint8_t* base = nullptr;
        if (dv->on_cpu)               base = (const uint8_t*)dv->data;
        else if (dv->pin_memory)      base = (const uint8_t*)dv->pin_memory;
        if (!base) return;                       // GPU-only expert (layers 0-1)

        if (!g._scale || !u._scale || !d._scale) return;
        if (g.ndim_scale() != 2 || u.ndim_scale() != 2 || d.ndim_scale() != 2) return;

        auto mirror_scale = [&](const Tensor& t) -> const float* {
            scale_store.emplace_back((size_t)t.num_elements_scale());
            std::vector<float>& hostv = scale_store.back();
            CUDA_CHECK(cudaMemcpy(hostv.data(), t.scale(),
                                  hostv.size() * sizeof(float), cudaMemcpyDeviceToHost));
            return hostv.data();
        };

        garlic_cpu::ExpertWeights w;
        w.w_gate = base + g.offset;
        w.w_up   = base + u.offset;
        w.w_down = base + d.offset;
        w.s_gate = mirror_scale(g);
        w.s_up   = mirror_scale(u);
        w.s_down = mirror_scale(d);
        w.I = g.shape[0];  w.H = g.shape[1];               // gate: [I, H]
        w.sg_rows = g.shape_scale[0]; w.sg_cols = g.shape_scale[1];
        w.su_rows = u.shape_scale[0]; w.su_cols = u.shape_scale[1];
        w.sd_rows = d.shape_scale[0]; w.sd_cols = d.shape_scale[1];

        // Sanity: dims consistent with model config and GPU kernel assumptions.
        if (w.H != H || w.I != I) return;
        if (d.shape[0] != H || d.shape[1] != I) return;
        if ((long long)w.sg_cols * garlic_cpu::kGroupK < w.H) return;
        if ((long long)w.sd_cols * garlic_cpu::kGroupK < w.I) return;
        if (!w.valid()) return;

        views[(size_t)layer * E + e] = w;
    }

    bool eligible(int layer, int e, int n_e, cudaDataType_t act_dtype) const {
        return enabled
            && act_dtype == CUDA_R_16BF
            && n_e > 0 && n_e <= token_threshold
            && n_slots < kMaxSlots
            && views[(size_t)layer * E + e].valid();
    }

    void begin_layer() { n_slots = 0; }

    // Stage activation rows for expert e. Precondition (holds in the repo):
    // the compute stream has been synchronized after moe_gather, so
    // `gathered` is fully written. Returns slot index, or -1.
    int stage(int layer, int e, uint32_t row_begin, int n_e, const Tensor& gathered) {
        if (n_slots >= kMaxSlots) return -1;
        const int s = n_slots++;
        Slot& sl = slots[s];
        sl.expert = e; sl.row_begin = row_begin; sl.n_e = n_e;
        sl.staged = true; sl.h2d_issued = false;

        const size_t off_elems = (size_t)s * token_threshold * H;
        const uint8_t* src = (const uint8_t*)gathered.data() + (size_t)row_begin * H * 2;
        CUDA_CHECK(cudaMemcpyAsync(x_stage + off_elems, src,
                                   (size_t)n_e * H * 2, cudaMemcpyDeviceToHost, xfer_stream));

        garlic_cpu::ExpertJob& j = sl.job;
        j.w      = &views[(size_t)layer * E + e];
        j.x_bf16 = x_stage + off_elems;
        j.y_bf16 = y_stage + off_elems;
        j.xf     = xf_scratch.data() + off_elems;
        j.h      = h_scratch.data() + (size_t)s * token_threshold * I;
        j.T      = n_e;
        return s;
    }

    // One sync for all staged D2H copies (a few KB), then hand jobs to the pool.
    void launch_jobs() {
        if (n_slots == 0) return;
        CUDA_CHECK(cudaStreamSynchronize(xfer_stream));
        for (int s = 0; s < n_slots; ++s) {
            if (!slots[s].staged) continue;
            exec->submit(slots[s].job);
            slots[s].staged = false;
        }
    }

    // Non-blocking: if the job finished and results are not yet on the GPU,
    // issue the H2D + event. Returns true once the H2D has been issued.
    bool poll_and_flush(int s, const Tensor& gathered) {
        Slot& sl = slots[s];
        if (sl.h2d_issued) return true;
        if (!sl.job.done()) return false;
        uint8_t* dst = (uint8_t*)gathered.data() + (size_t)sl.row_begin * H * 2;
        CUDA_CHECK(cudaMemcpyAsync(dst, sl.job.y_bf16,
                                   (size_t)sl.n_e * H * 2, cudaMemcpyHostToDevice, xfer_stream));
        CUDA_CHECK(cudaEventRecord(sl.ev, xfer_stream));
        sl.h2d_issued = true;
        return true;
    }

    // Block (CPU-side) until every slot's results are in flight, then fence
    // the compute stream on their H2D events. Called just before moe_scatter.
    void finish_layer(const Tensor& gathered, cudaStream_t compute) {
        for (int s = 0; s < n_slots; ++s) {
            while (!poll_and_flush(s, gathered))
                std::this_thread::yield();
            CUDA_CHECK(cudaStreamWaitEvent(compute, slots[s].ev, 0));
        }
    }
};

// Single global instance (C++17 inline variable).
inline CpuMoeRuntime g_cpu_moe;
