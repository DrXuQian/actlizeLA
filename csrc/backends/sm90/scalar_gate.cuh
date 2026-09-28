#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace gdn::sm90 {
struct NoGateFactors {
    CUTE_DEVICE void operator()(int, float, float, int) const {}
};
// Natural-log increments, NOT multipliers or cumulative gates. Prefix resets
// at each C64 tile; the resident state carries all earlier chunk decay.
template <int Rows, int Dim, class Gate, class Work, class Pipeline,
          class PipeState, class SharedTensor, class PublishFactors = NoGateFactors>
CUTE_DEVICE void load_scalar_gate(Gate const* gate, int heads, Work const& work,
                                 int block, Pipeline& pipeline,
                                 PipeState& stage, SharedTensor shared,
                                 PublishFactors publish_factors = {}) {
    static_assert(Rows == 64 && Dim == 128);
    int lane = int(threadIdx.x) & 31;
    int row = block * Rows + lane;
    int64_t token = work.tok_offset + row;
    float lo = row < work.seq_len ? float(gate[token * heads + work.o_head_idx()]) : 0.f;
    float hi = row + 32 < work.seq_len ? float(gate[(token + 32) * heads + work.o_head_idx()]) : 0.f;
    CUTE_UNROLL
    for (int distance = 1; distance < 32; distance *= 2) {
        float a = __shfl_up_sync(0xffffffffu, lo, distance);
        float b = __shfl_up_sync(0xffffffffu, hi, distance);
        if (lane >= distance) { lo += a; hi += b; }
    }
    hi += __shfl_sync(0xffffffffu, lo, 31);
    constexpr float log2e = 1.4426950408889634f;
    lo *= log2e; hi *= log2e;
    pipeline.producer_acquire(stage);
    shared(lane, 0, stage.index()) = lo;
    shared(lane + 32, 0, stage.index()) = hi;
    publish_factors(lane, lo, hi, stage.index());
    // PipelineAsync counts every producer lane (32), not TMA bytes.
    cutlass::arch::fence_view_async_shared();
    pipeline.producer_commit(stage);
    ++stage;
}
} // namespace gdn::sm90
