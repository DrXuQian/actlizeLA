#include "target.cuh"
#include "launch.h"
#include <cute/tensor.hpp>
#include "kda/sm90/device/device_universal.hpp"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "value_types.cuh"
#include "configuration.cuh"
#include <climits>
#include <stdexcept>

namespace gdn::sm90 {
using namespace cute;
using namespace kda::sm90::kernel;
using BF16 = cutlass::bfloat16_t;

template <class Gate, bool Initial>
void run(Arguments const& a, cudaStream_t stream) {
    using Types = ValueKernelTypes<Gate,Initial,ConfigurationTraits::ValueTile,
                                  ConfigurationTraits::AuxRegisters>;
    using Kernel = std::conditional_t<ConfigurationTraits::Tuned,
        typename Types::Kernel, typename Types::Builder::Kernel>;
    using Operation = cutlass::device::Universal<Kernel>;
    typename Operation::Arguments args{};
    args.problem_size.total_seqlen = int64_t(a.batch) * a.length;
    args.problem_size.num_seqs = a.batch;
    args.problem_size.num_qk_heads = a.qk_heads;
    args.problem_size.num_v_heads = a.v_heads;
    args.problem_size.head_size = 128;
    args.problem_size.sequence_length = a.length;
    auto& m = args.mainloop;
    m.ptr_Q = static_cast<BF16 const*>(a.q); m.dQ = {int64_t(a.qk_heads)*128, _1{}, 128};
    m.ptr_K = static_cast<BF16 const*>(a.k); m.dK = m.dQ;
    m.ptr_V = static_cast<BF16 const*>(a.v); m.dV = {int64_t(a.v_heads)*128, _1{}, 128};
    m.ptr_O = static_cast<BF16*>(a.output); m.dO = m.dV;
    m.ptr_Gate = static_cast<Gate const*>(a.gate);
    m.beta_ptr = static_cast<BF16 const*>(a.beta); m.beta_stride = {a.v_heads, 1};
    m.ptr_input_state = a.initial; m.ptr_output_state = a.final;
    m.scale = 0.08838834764831844f; // 1/sqrt(128), public GDN contract
    args.hw_info.sm_count = 0; // grid=B*Hv, no SM-dependent scratch/dispatch
    Operation op;
    if (op.get_workspace_size(args) != 0 || op.can_implement(args) != cutlass::Status::kSuccess)
        throw std::runtime_error("GDN fused SM90 invalid kernel arguments/workspace");
    if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess ||
        op.run(stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("GDN fused SM90 initialize/launch failed");
}

void launch(Arguments const& a, cudaStream_t stream) {
    if (a.batch <= 0 || a.length <= 0 || a.qk_heads <= 0 || a.v_heads <= 0 ||
        a.v_heads % a.qk_heads || int64_t(a.batch)*a.length > INT32_MAX ||
        int64_t(a.batch)*a.v_heads > INT32_MAX)
        throw std::invalid_argument("GDN fused SM90 invalid shape/GVA/grid");
    if (a.gate_fp32) {
        if (a.initial) run<float, true>(a, stream); else run<float, false>(a, stream);
    } else {
        if (a.initial) run<BF16, true>(a, stream); else run<BF16, false>(a, stream);
    }
}
} // namespace gdn::sm90
