// L209 -- numerical authority for the production 16 -> 32 -> 64 inverse.
// The direct 16x16 triangular solves remain FP32.  Each of the six block
// products rounds both operands to TF32 at the MMA boundary and accumulates in
// FP32, matching PPU0010 m16n16k8 semantics.  This is deliberately distinct
// from the all-FP32 L203 authority: the non-exact fixture quantifies that seam.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "cutlass/tfloat32.h"
#include "reference/ppu_chunked_gdn_inverse.hpp"

namespace {

constexpr int kN = 64;
constexpr int kElements = kN * kN;
constexpr int kExpectedBlockProducts = 6;
constexpr int kExpectedAiuGeneratedEntries =
    2 * 16 * 16 + 32 * 32;

float tf32(float x) {
  return float(cutlass::tfloat32_t(x));
}

void direct16(float const* lower, float* inverse, int begin) {
  for (int row = 0; row < 16; ++row) {
    for (int column = 0; column < 16; ++column) {
      float value = row == column ? 1.0f : 0.0f;
      for (int k = 0; k < row; ++k) {
        value -= lower[(begin + row) * kN + begin + k] *
                 inverse[(begin + k) * kN + begin + column];
      }
      inverse[(begin + row) * kN + begin + column] = value;
    }
  }
}

// C[M,N] = A[M,K] * B[N,K], with explicit element strides.  This is the same
// logical B convention as PpuChunkedGdnResidentMmaTf32::mma.
void tf32_product(
    float* output, int output_ld,
    float const* a, int stride_a_m, int stride_a_k,
    float const* b, int stride_b_n, int stride_b_k,
    int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      float value = 0.0f;
      for (int reduction = 0; reduction < k; ++reduction) {
        float const av = tf32(a[row * stride_a_m + reduction * stride_a_k]);
        float const bv = tf32(b[column * stride_b_n + reduction * stride_b_k]);
        value = std::fma(av, bv, value);
      }
      output[row * output_ld + column] = value;
    }
  }
}

void merge(
    float const* lower, float* inverse, int begin, int half,
    int& block_products) {
  std::array<float, 32 * 32> temp{};
  int const lower_begin = begin + half;

  // temp = D^-1 C.  Logical B[N,K] views C[K,N] through strides (1,64).
  tf32_product(temp.data(), half,
               inverse + lower_begin * kN + lower_begin, kN, 1,
               lower + lower_begin * kN + begin, 1, kN,
               half, half, half);
  ++block_products;

  std::array<float, 32 * 32> result{};
#if defined(L209_PLANT_B_STRIDE)
  int const b_n = kN;
  int const b_k = 1;
#else
  int const b_n = 1;
  int const b_k = kN;
#endif
  tf32_product(result.data(), half,
               temp.data(), half, 1,
               inverse + begin * kN + begin, b_n, b_k,
               half, half, half);
  ++block_products;

  for (int row = 0; row < half; ++row) {
    for (int column = 0; column < half; ++column) {
#if defined(L209_PLANT_MISSING_NEGATE)
      inverse[(lower_begin + row) * kN + begin + column] =
          result[row * half + column];
#else
      inverse[(lower_begin + row) * kN + begin + column] =
          -result[row * half + column];
#endif
    }
  }
}

void blocked_tf32(float const* lower, float* inverse, int& block_products) {
  std::fill(inverse, inverse + kElements, 0.0f);
  for (int begin = 0; begin < kN; begin += 16) {
    direct16(lower, inverse, begin);
  }
  merge(lower, inverse, 0, 16, block_products);
  merge(lower, inverse, 32, 16, block_products);
#if !defined(L209_PLANT_SKIP_FINAL_MERGE)
  merge(lower, inverse, 0, 32, block_products);
#endif
}

std::array<float, kElements> exact_lower() {
  std::array<float, kElements> lower{};
  for (int row = 1; row < kN; ++row) {
    // A bidiagonal unit-lower matrix keeps every generated block value a
    // signed power of two: TF32 conversion and FP32 accumulation are exact.
    lower[std::size_t(row * kN + row - 1)] =
        (row & 1) ? 0.5f : -0.5f;
  }
  return lower;
}

std::array<float, kElements> nonexact_lower() {
  std::array<float, kElements> lower{};
  for (int row = 1; row < kN; ++row) {
    for (int column = std::max(0, row - 5); column < row; ++column) {
      int const code = (row * 17 + column * 13) % 19 - 9;
      lower[std::size_t(row * kN + column)] = float(code) * 0.0007f;
    }
  }
  return lower;
}

int raw_bad(float const* a, float const* b) {
  int bad = 0;
  for (int i = 0; i < kElements; ++i) {
    bad += std::memcmp(a + i, b + i, sizeof(float)) != 0;
  }
  return bad;
}

float max_abs_diff(float const* a, float const* b) {
  float value = 0.0f;
  for (int i = 0; i < kElements; ++i) {
    value = std::max(value, std::fabs(a[i] - b[i]));
  }
  return value;
}

}  // namespace

int main() {
  auto const exact = exact_lower();
  auto const nonexact = nonexact_lower();
  std::array<float, kElements> exact_fp32{};
  std::array<float, kElements> exact_tf32{};
  std::array<float, kElements> nonexact_fp32{};
  std::array<float, kElements> nonexact_tf32{};

  bool const fp32_exact_ok =
      cutlass::linear_attention::detail::invert_unit_lower_blocked(
          exact.data(), exact_fp32.data(), kN, 16);
  bool const fp32_nonexact_ok =
      cutlass::linear_attention::detail::invert_unit_lower_blocked(
          nonexact.data(), nonexact_fp32.data(), kN, 16);
  int exact_products = 0;
  int nonexact_products = 0;
  blocked_tf32(exact.data(), exact_tf32.data(), exact_products);
  blocked_tf32(nonexact.data(), nonexact_tf32.data(), nonexact_products);

  int const exact_bad = raw_bad(exact_fp32.data(), exact_tf32.data());
  int const nonexact_bad = raw_bad(nonexact_fp32.data(), nonexact_tf32.data());
  float const exact_diff =
      max_abs_diff(exact_fp32.data(), exact_tf32.data());
  float const nonexact_diff =
      max_abs_diff(nonexact_fp32.data(), nonexact_tf32.data());
  float const exact_residual =
      cutlass::linear_attention::detail::unit_lower_inverse_residual(
          exact.data(), exact_tf32.data(), kN);
  float const nonexact_residual =
      cutlass::linear_attention::detail::unit_lower_inverse_residual(
          nonexact.data(), nonexact_tf32.data(), kN);

  bool const ok = fp32_exact_ok && fp32_nonexact_ok &&
                  exact_products == kExpectedBlockProducts &&
                  nonexact_products == kExpectedBlockProducts &&
                  exact_bad == 0 && exact_residual == 0.0f &&
                  nonexact_bad >= kExpectedAiuGeneratedEntries &&
                  nonexact_diff > 0.0f && nonexact_diff < 2.0e-5f &&
                  nonexact_residual < 2.0e-5f;
  std::printf(
      "[l209] %s: decomposition=16->32->64 block_products=%d/%d "
      "exact_raw_bad=%d/4096 exact_residual=%g "
      "exact_max_abs=%g nonexact_raw_bad=%d/4096 aiu_generated=%d max_abs=%g residual=%g "
      "authority=TF32-input+FP32-FMA\n",
      ok ? "PASS" : "FAIL", exact_products, nonexact_products, exact_bad,
      exact_residual, exact_diff, nonexact_bad, kExpectedAiuGeneratedEntries,
      nonexact_diff, nonexact_residual);
  return ok ? 0 : 1;
}
