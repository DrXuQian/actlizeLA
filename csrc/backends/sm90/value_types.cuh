// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "scalar_gdn_state.cuh"

namespace gdn::sm90 {
template<class Gate, bool Initial, int ValueTile = 128, int AuxRegs = 104>
struct ValueKernelTypes {
    using BF16 = cutlass::bfloat16_t;
    using OriginalOptions = std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementGateGmem,Gate>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kElementBetaGmem,BF16>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kInitStateFromInput,cute::bool_constant<Initial>>>;
    using SplitOptions = decltype(std::tuple_cat(OriginalOptions{},std::tuple<
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kValueTile,cute::Int<ValueTile>>,
        kda::sm90::kernel::Option<kda::sm90::kernel::Tag::kAuxRegisters,cute::Int<AuxRegs>>>{}));
    using Options = std::conditional_t<ValueTile == 128,OriginalOptions,SplitOptions>;
    using Stride = cute::tuple<int64_t,cute::_1,int32_t>;
    using Builder = kda::sm90::kernel::FlatBuilderKdaFwd<BF16,float,float,
        cute::Shape<cute::_64,cute::_64,cute::_128>,Stride,Stride,Stride,Stride,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,Options>;
    using Collective = ScalarGdnState<typename Builder::CollectiveMainloop,true>;
    using Scheduler = std::conditional_t<ValueTile == 128,typename Builder::TileScheduler,
                                         ValueTileScheduler<ValueTile>>;
    using Kernel = kda::sm90::kernel::FlatKernelTmaWarpSpecializedKdaFwd<Collective,Scheduler,Options>;
};
} // namespace gdn::sm90
