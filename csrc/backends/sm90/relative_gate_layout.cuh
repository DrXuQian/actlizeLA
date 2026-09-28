#pragma once
#include <cute/config.hpp>

namespace gdn::sm90 {
// One32-lane producer owns lo[0..31],hi[32..63]. The same alpha stage
// protects prefixes, prefix factors and this relative-decay channel.
CUTE_HOST_DEVICE constexpr int relative_gate_last_lane(int valid) { return (valid-1)&31; }
CUTE_HOST_DEVICE constexpr bool relative_gate_last_is_hi(int valid) { return valid>32; }
CUTE_HOST_DEVICE constexpr int relative_gate_index(int stage,int token) { return stage*64+token; }
} // namespace gdn::sm90
