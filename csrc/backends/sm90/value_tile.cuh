// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kda/sm90/kernel/tile_scheduler.hpp"

namespace gdn::sm90 {

// Keep public [B,T,Hv,128] and [B,Hv,K,V] strides. Only a CTA's V ownership
// changes. Both halves repeat QK/inverse, but never reduce or share state.
template<int ValueTile>
struct ValueTileScheduler : kda::sm90::kernel::IndividualTileScheduler {
    using Base = kda::sm90::kernel::IndividualTileScheduler;
    using Params = typename Base::Params;
    static_assert(ValueTile == 64);
    CUTE_DEVICE ValueTileScheduler(Params const& p) : Base(p) {}
    static dim3 get_grid_shape(Params const& p) {
        dim3 grid = Base::get_grid_shape(p);
        grid.y = 128 / ValueTile;
        return grid;
    }
    CUTE_HOST_DEVICE static constexpr int value_begin(int tile) { return tile*ValueTile; }
    template<class Problem>
    CUTE_DEVICE auto get_next_work(Params p, Problem const& problem) {
        auto work = Base::get_next_work(p,problem);
        work.value_offset = value_begin(int(blockIdx.y));
        return work;
    }
};

struct SingleStateOrder {
    CUTE_DEVICE void init(int) {}
    CUTE_DEVICE void ordered_or_wait(int) {}
    CUTE_DEVICE void notify_next_blocked(int) {}
};

// This public-stride helper is also checked independently by the host map
// proof, including batched tails where TMA stores must become predicated.
CUTE_HOST_DEVICE constexpr int64_t value_output_index(
    int64_t token, int heads, int head, int value_begin, int local_value) {
    return (token*heads+head)*128 + value_begin + local_value;
}
} // namespace gdn::sm90
