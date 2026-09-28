// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace gdn::sm90 {

// Convert in logical accumulator order BEFORE its register-operand retile.
// No aliasing assumption: each pair is explicitly an RNE numeric conversion.
template<class To, class Fragment>
CUTE_DEVICE auto convert_fragment(Fragment const& src) {
    using From = typename Fragment::value_type;
    constexpr int Count = decltype(cute::size(src))::value;
    cutlass::Array<From,Count> input;
    CUTE_UNROLL
    for (int i=0;i<Count;++i) input[i]=src(i);
    auto converted=cutlass::NumericArrayConverter<To,From,Count,
        cutlass::FloatRoundStyle::round_to_nearest>{}(input);
    auto dst=cute::make_fragment_like<To>(src);
    CUTE_UNROLL
    for (int i=0;i<Count;++i) dst(i)=converted[i];
    return dst;
}

} // namespace gdn::sm90
