#include <iostream>
#ifdef USE_CUTLASS_FP8
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"


#include "tools/util/include/cutlass/util/packed_stride.hpp"
#endif
namespace garlic{
#ifdef USE_CUTLASS_FP8
    using namespace cute;

#define CUTLASS_CHECK(status)                                                                    \
{                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
        exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
}

    // A matrix configuration
    using ElementA            = cutlass::float_e4m3_t;                          // Element type for A matrix operand
    using LayoutA             = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB            = cutlass::float_e4m3_t;                          // Element type for B matrix operand
    using LayoutB             = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using ElementC            = cutlass::bfloat16_t;                          // Element type for C and D matrix operands
    using LayoutC             = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
    constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    using ElementD           = ElementC;
    using LayoutD            = LayoutC;
    constexpr int AlignmentD = AlignmentC;

    // MMA type
    using ElementAccumulator = float;                                           // Element Accumulator will also be our scale factor type
    using ElementCompute = float;


    // MMA and Cluster Tile Shapes
    // Shape of the tile
    using CooperativeMmaTileShape_MNK = Shape<_128,_128,_128>;                          

    // Shape of the threadblocks in a cluster
    using ClusterShape_MNK = Shape<_1,_1,_1>;

    constexpr int ScaleGranularityM = 128;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 128;

    using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
    using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand


    template <class TileShape>
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutC, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    template <class TileShape, class Schedule>
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
        ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
        ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue<TileShape>::SharedStorage))>,
        Schedule
    >::CollectiveOp;

    template <class TileShape, class Schedule>
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop<TileShape, Schedule>,
        CollectiveEpilogue<TileShape>,
        void>;

    using CooperativeGemm = cutlass::gemm::device::GemmUniversalAdapter<
        GemmKernel<CooperativeMmaTileShape_MNK, cutlass::gemm::KernelScheduleSm120Blockwise>>;


    using StrideA = typename CooperativeGemm::GemmKernel::StrideA;
    using StrideB = typename CooperativeGemm::GemmKernel::StrideB;
    using StrideC = typename CooperativeGemm::GemmKernel::StrideC;
    using StrideD = typename CooperativeGemm::GemmKernel::StrideD;

    
    void gemm_fp8_groupwise_bf16_sm120(
        void* y,
        const void* W,
        float* sfa,
        const void* x,
        float* sfb,
        int N, int K, int M, int L){
        using namespace cute;

        // TMA epilogue requires M (CUTLASS N dimension) to be a multiple of 8 for BF16.
        constexpr int M_align = 8;
        int M_padded = (M + M_align - 1) & ~(M_align - 1);
        bool needs_pad = (M_padded > M);

        // A = W (weight, N x K RowMajor), B = x (input, M x K stored as ColumnMajor K x M)
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, L));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M_padded, K, L));

        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M_padded, L));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M_padded, L));
        LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(N, M_padded, K, L));
        LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(N, M_padded, K, L));
        //
        // When padded, GEMM writes to a temporary N x M_padded output; real M cols are copied back.
        size_t padded_y_bytes = needs_pad ? (size_t)N * M_padded * sizeof(ElementD) : 0;

        // Build args with a placeholder D pointer; we'll set the real one after workspace is ready.
        auto make_args = [&](ElementD* y_ptr) {
            typename CooperativeGemm::Arguments a{
                cutlass::gemm::GemmUniversalMode::kGemm,
                    {N, M_padded, K, L},
                    {
                        (const ElementA*) W,    stride_A,
                        (const ElementB*) x,    stride_B,
                        sfa,                   layout_SFA,
                        sfb,                   layout_SFB
                    },
                    {
                        {},
                        nullptr,   stride_C,
                        y_ptr,     stride_D
                    }
            };
            a.epilogue.thread.alpha = 1.0f;
            a.epilogue.thread.beta  = 0.0f;
            return a;
        };

        CooperativeGemm gemm;
        size_t cutlass_ws = CooperativeGemm::get_workspace_size(make_args(nullptr));
        size_t total_ws   = cutlass_ws + padded_y_bytes;

        static uint8_t* workspace   = nullptr;
        static size_t   ws_capacity = 0;
        if (total_ws > ws_capacity) {
            cudaFree(workspace);
            cudaMalloc(&workspace, total_ws);
            ws_capacity = total_ws;
        }

        ElementD* y_ptr = needs_pad
            ? reinterpret_cast<ElementD*>(workspace + cutlass_ws)
            : reinterpret_cast<ElementD*>(y);

        auto args = make_args(y_ptr);

        CUTLASS_CHECK(gemm.can_implement(args));
        CUTLASS_CHECK(gemm.initialize(args, workspace));
        CUTLASS_CHECK(gemm.run());

        if (needs_pad) {
            // Copy the N x M real columns out of the N x M_padded padded output.
            cudaMemcpy2D(
                y,                                    // dst
                (size_t)M        * sizeof(ElementD),  // dst pitch (bytes per row)
                y_ptr,                                // src
                (size_t)M_padded * sizeof(ElementD),  // src pitch
                (size_t)M        * sizeof(ElementD),  // width to copy
                (size_t)N,                            // N rows
                cudaMemcpyDeviceToDevice);
        }
    }


#endif

}