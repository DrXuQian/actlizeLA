#pragma once

#include "../../../include/gdn_qsa/backend_target.h"
#if defined(GDN_QSA_PPU)
#error "CUDA SM80 primitives cannot be included in a PPU AIU target"
#endif

#include <cstddef>
#include <cstdio>

// Target primitives only. All scheduling, reset admission, register-state
// lifetimes, double buffering and Neumann expansion live in the original TUs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace gdn_arch {

using Stream = cudaStream_t;
using MmaBf16 = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using MmaF16 = cute::SM80_16x8x16_F16F16F16F16_TN;
template <int R, int C>
using RowLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
    cute::make_shape(cute::Int<R>{}, cute::Int<C>{}), cute::LayoutLeft{}));
template <int R, int C>
using ColumnLayout = decltype(cute::tile_to_shape(
    cute::GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
    cute::make_shape(cute::Int<R>{}, cute::Int<C>{}), cute::LayoutRight{}));
template <class Kernel>
inline void set_smem(Kernel kernel, std::size_t bytes) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
}
inline void copy_device(void* dst, void const* src, std::size_t bytes, Stream s) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
}

CUTE_DEVICE float bf16_to_float(cutlass::bfloat16_t x) {
    return static_cast<float>(x);
}
CUTE_DEVICE void async_copy16(void* dst, void const* src, bool pred) {
    uint32_t address = cute::cast_smem_ptr_to_uint(dst);
    int const bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(address), "l"(src), "r"(bytes));
}
template <int Threads>
CUTE_DEVICE void compute_barrier() {
    asm volatile("bar.sync 8, %0;" :: "n"(Threads) : "memory");
}

// Every native PPU accumulator has four columns on each of its two rows.
// Keep all original beta/state update loops; only their coordinate ABI varies.
CUTE_HOST_DEVICE auto row_coordinate(int a, int d, int upper_row) {
    return cute::make_coord(cute::make_coord(a, upper_row), 0, d);
}

template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_a(Mma const& mma) {
    return cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>{}, mma);
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_at(Mma const& mma) {
    return cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>{}, mma);
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto copy_b(Mma const& mma) {
    return cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>{}, mma);
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto load_c(Mma const& mma) {
    return cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>{}, mma);
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto load_ct(Mma const& mma) {
    return cute::make_tiled_copy_C(cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>{}, mma);
}
template <class Element, class Mma>
CUTE_HOST_DEVICE auto store_c(Mma const& mma) {
    return cute::make_tiled_copy_C(cute::Copy_Atom<cute::AutoVectorizingCopy, Element>{}, mma);
}

CUTE_DEVICE void result_to_b_words(uint32_t const* src, uint32_t* dst) {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) cute::SM75_U32x1_MOVM_T::copy(src[i], dst[i]);
}
CUTE_DEVICE void transpose_a_words(uint32_t const* src, uint32_t* dst) {
    result_to_b_words(src, dst);
}
CUTE_DEVICE void operand_to_result_words(uint32_t const* src, uint32_t* dst) {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) dst[i] = src[i];
}

// The original Neumann expansion stores all powers in A-register order.
// Preserve that invariant around the native PPU F16 MMA's C-register order.
CUTE_DEVICE void mma_f16_16x16(
    uint32_t* d, uint32_t const* a, uint32_t const* b, uint32_t const* c) {
    MmaF16::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]);
    MmaF16::fma(d[2], d[3], a[0], a[1], a[2], a[3], b[2], b[3], c[2], c[3]);
}

template <class A, class B, class C, class Store>
CUTE_DEVICE void dot_16x16(A const& a, B const& b, C& c, int lane, Store store) {
    using namespace cute;
    auto mma = make_tiled_mma(MmaBf16{}, Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    cooperative_gemm(lane, mma, 1.0f, a, b, 0.0f, c,
        identity{}, identity{}, identity{}, store,
        SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{},
        SM75_U32x4_LDSM_N{}, AutoVectorizingCopy{});
}

}  // namespace gdn_arch
