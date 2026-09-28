// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cute/numeric/integral_constant.hpp>

namespace gdn::sm90 {
// Avoid length+63 overflow at the admitted positive int32 extent boundary.
CUTE_HOST_DEVICE constexpr int aux_chunk_count(int length) {
    return 1 + (length - 1) / 64;
}
// Positive sequence lengths are admitted by the public launcher. The final
// block stays dynamic, including an exactly-full final block. No tail policy
// or pipeline phase is changed by specializing the preceding full chunks.
template<class Body>
CUTE_HOST_DEVICE void for_each_aux_chunk(int length, Body&& body) {
    int chunks = aux_chunk_count(length);
    CUTE_NO_UNROLL
    for (int chunk=0; chunk<chunks-1; ++chunk) body(chunk, cute::Int<64>{});
    body(chunks-1, length-(chunks-1)*64);
}
} // namespace gdn::sm90
