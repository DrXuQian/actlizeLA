// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "kerutils/common/cute_ext.hpp"

namespace gdn::sm90 {

// Shared by the real consumer and host ownership proof. Logical coordinates
// are independent of the parent's physical KK layout and stage pitch.
template<class Tensor>
CUTE_HOST_DEVICE auto inverse64_row_tiles(Tensor const& matrix,int row_half) {
    using namespace cute;
    auto quadrants=flat_divide(matrix,Shape<_32,_32>{});
    auto d=local_tile(quadrants(_,_,_1{},_1{}),Shape<_16,_32>{},make_coord(row_half,_0{}));
    auto c=kerutils::select_tensor<1,0>(quadrants(_,_,_1{},_0{}));
    auto a=kerutils::select_tensor<1,0>(quadrants(_,_,_0{},_0{}));
    auto o=local_tile(quadrants(_,_,_1{},_0{}),Shape<_16,_32>{},make_coord(row_half,_0{}));
    return make_tuple(d,c,a,o);
}

template<class Element,class Tensor,class Convert>
CUTE_DEVICE void inverse64_local_reduction(Tensor const& matrix,int barrier_id,Convert convert) {
    using namespace cute;
    using SmemLayout=typename Tensor::layout_type;
    static_assert(rank(SmemLayout{})==2 && size<0>(SmemLayout{})==64 && size<1>(SmemLayout{})==64);
    constexpr bool column_major=stride<0>(SmemLayout{})==1;
    using Mma=SM80_16x8x16_F32F16F16F32_TN;
    using First=decltype(make_tiled_mma(Mma{},Layout<Shape<_1,_1>>{},Shape<_16,_32,_32>{}));
    using Second=decltype(make_tiled_mma(Mma{},Layout<Shape<_1,_1>>{},Shape<_16,_32,_16>{}));
    using CopyD=std::conditional_t<column_major,SM75_U16x8_LDSM_T,SM75_U32x4_LDSM_N>;
    using CopyB=std::conditional_t<column_major,SM75_U32x4_LDSM_N,SM75_U16x8_LDSM_T>;
    using CopyO=std::conditional_t<column_major,SM90_U16x8_STSM_T,SM90_U32x4_STSM_N>;
    int warp=(int(threadIdx.x)>>5)&3, lane=int(threadIdx.x)&31;
    First first;Second second;
    auto t1=first.get_thread_slice(lane);
    auto t2=second.get_thread_slice(lane);
    auto copy_d=make_tiled_copy_A(Copy_Atom<CopyD,Element>{},first);
    auto copy_c=make_tiled_copy_B(Copy_Atom<CopyB,Element>{},first);
    auto copy_a=make_tiled_copy_B(Copy_Atom<CopyB,Element>{},second);
    auto copy_o=make_tiled_copy_C(Copy_Atom<CopyO,Element>{},second);
    auto output=make_fragment_like<Element>(partition_fragment_C(second,Shape<_16,_32>{}));
    if(warp<2) {
        auto tiles=inverse64_row_tiles(matrix,warp);
        auto d=t1.partition_fragment_A(get<0>(tiles));
        auto c=t1.partition_fragment_B(get<1>(tiles));
        auto a=t2.partition_fragment_B(get<2>(tiles));
        auto dc=partition_fragment_C(first,Shape<_16,_32>{});
        auto td=copy_d.get_thread_slice(lane);
        auto tc=copy_c.get_thread_slice(lane);
        auto ta=copy_a.get_thread_slice(lane);
        copy(copy_d,td.partition_S(get<0>(tiles)),td.retile_D(d));
        copy(copy_c,tc.partition_S(get<1>(tiles)),tc.retile_D(c));
        clear(dc);gemm(first,d,c,dc);
        transform(dc,[](auto x){return -x;});
        auto dc_half=convert(dc,second);
        copy(copy_a,ta.partition_S(get<2>(tiles)),ta.retile_D(a));
        auto low=partition_fragment_C(second,Shape<_16,_32>{});
        auto high=partition_fragment_C(second,Shape<_16,_32>{});
        clear(low);clear(high);
        // Preserve the old two K16 FP32 partials and their separate FP16
        // roundings. A single K32 accumulation is NOT numerically equivalent.
        gemm(second,dc_half(_,_,_0{}),a(_,_,_0{}),low);
        gemm(second,dc_half(_,_,_1{}),a(_,_,_1{}),high);
        CUTE_UNROLL
        for(int i=0;i<size(output);++i) output(i)=Element(high(i))+Element(low(i));
    }
    // C and output alias. Both row warps must finish reading C first; the
    // other two warps still participate in the original 128-thread group.
    cutlass::arch::NamedBarrier::arrive_and_wait(128,barrier_id);
    if(warp<2) {
        auto tiles=inverse64_row_tiles(matrix,warp);
        auto store=copy_o.get_thread_slice(lane);
        copy(copy_o,store.retile_S(output),store.partition_D(get<3>(tiles)));
    }
    // The unchanged caller's final barrier retires publication before the
    // inverse-to-BF16 consumer. No partial sum crosses a warp or shared plane.
}
} // namespace gdn::sm90
