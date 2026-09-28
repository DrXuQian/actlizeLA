#pragma once

#include "../../../include/gdn_qsa/backend_target.h"
#if !defined(GDN_QSA_PPU)
#error "PPU AIU primitives require the explicit legacy PPU target"
#endif

#include <cstddef>
#include <cstdio>

// Target primitives only. All scheduling, reset admission, register-state
// lifetimes, double buffering and Neumann expansion live in the original TUs.
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/arch/copy_ppu.hpp>
#include "gdn_qsa/ppu/shared_copy.cuh"
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace gdn_arch {

using Stream = hggcStream_t;
using MmaBf16 = cute::PPU0010_16x16x16_F32BF16BF16F32_TN;
using MmaF16 = cute::PPU0010_16x16x16_F16F16F16F16_TN;
template <int R, int C> using RowLayout = gdn_qsa::ppu::RowLayout<R, C>;
template <int R, int C> using ColumnLayout = gdn_qsa::ppu::ColumnLayout<R, C>;
template <class Kernel>
inline void set_smem(Kernel kernel, std::size_t bytes) {
    auto status = hggcFuncSetAttribute(
        kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
    if (status != hggcSuccess) {
        std::fprintf(stderr, "[PPU GDN] dynamic shared-memory admission failed: %s\n",
                     hggcGetErrorString(status));
    }
}
inline void copy_device(void* dst, void const* src, std::size_t bytes, Stream s) {
    hggcMemcpyAsync(dst, src, bytes, hggcMemcpyDeviceToDevice, s);
}

CUTE_DEVICE float bf16_to_float(cutlass::bfloat16_t x) {
    return static_cast<float>(x);
}
CUTE_DEVICE void async_copy16(void* dst, void const* src, bool pred) {
    cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
        *reinterpret_cast<cute::uint128_t const*>(src),
        *reinterpret_cast<cute::uint128_t*>(dst), pred);
}
template <int Threads>
CUTE_DEVICE void compute_barrier() {
    asm volatile("ppu.bar.sync 8, %0;" :: "n"(Threads) : "memory");
}

// Every native PPU accumulator has four columns on each of its two rows.
// Keep all original beta/state update loops; only their coordinate ABI varies.
CUTE_HOST_DEVICE auto row_coordinate(int a, int d, int upper_row) {
    return a + 2 * d + 4 * upper_row;
}

template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_a(Mma const& mma) {
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::A>{};
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_at(Mma const& mma) {
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::AT>{};
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_b(Mma const& mma) {
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::B>{};
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto load_c(Mma const& mma) {
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::C>{};
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto load_ct(Mma const& mma) {
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::CT>{};
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto store_c(Mma const& mma) {
    return gdn_qsa::ppu::SharedCopy<gdn_qsa::ppu::CopyRole::StoreC>{};
}

CUTE_DEVICE void result_to_b_words(uint32_t const* src, uint32_t* dst) {
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Result, FragmentMap::Operand, true>(src, dst);
}
CUTE_DEVICE void transpose_a_words(uint32_t const* src, uint32_t* dst) {
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Operand, FragmentMap::Operand, true>(src, dst);
}
CUTE_DEVICE void operand_to_result_words(uint32_t const* src, uint32_t* dst) {
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Operand, FragmentMap::Result, false>(src, dst);
}

// The original Neumann expansion stores all powers in A-register order.
// Preserve that invariant around the native PPU F16 MMA's C-register order.
CUTE_DEVICE void mma_f16_16x16(
    uint32_t* d, uint32_t const* a, uint32_t const* b, uint32_t const* c) {
    uint32_t acc[4], result[4];
    operand_to_result_words(c, acc);
    MmaF16::fma(result[0], result[1], result[2], result[3],
               a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
               acc[0], acc[1], acc[2], acc[3]);
    using gdn_qsa::ppu::FragmentMap;
    gdn_qsa::ppu::remap<FragmentMap::Result, FragmentMap::Operand, false>(result, d);
}

template <class A, class B, class C, class Store>
CUTE_DEVICE void dot_16x16(A const& a, B const& b, C& c, int lane, Store store) {
    using namespace cute;
    auto mma = make_tiled_mma(MmaBf16{}, Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    auto acc = partition_fragment_C(mma, Shape<_16, _16>{});
    clear(acc);
    auto copyA = copy_a<cutlass::bfloat16_t>(mma);
    auto copyB = copy_b<cutlass::bfloat16_t>(mma);
    auto thrA = copyA.get_slice(lane);
    auto thrB = copyB.get_slice(lane);
    auto at = local_tile(a, Shape<_16, _16>{}, make_coord(0, 0));
    auto bt = local_tile(b, Shape<_16, _16>{}, make_coord(0, 0));
    auto ra = thr.partition_fragment_A(at);
    auto rb = thr.partition_fragment_B(bt);
    CUTE_UNROLL
    for (int k = 0; k < size<1>(a) / 16; ++k) {
        copy(copyA, thrA.partition_S(local_tile(a, Shape<_16, _16>{}, make_coord(0, k))), ra);
        copy(copyB, thrB.partition_S(local_tile(b, Shape<_16, _16>{}, make_coord(0, k))), rb);
        gemm(thr, ra(_, _, Int<0>{}), rb(_, _, Int<0>{}), acc);
    }
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        auto rc = gdn_qsa::ppu::result_coord(lane, i);
        c(rc.row, rc.col) = store(acc(i));
    }
}

}  // namespace gdn_arch
