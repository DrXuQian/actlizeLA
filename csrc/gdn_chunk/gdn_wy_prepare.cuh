#pragma once

#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_stage_address.cuh"
#include "gdn_qsa/ppu/wy_prepare_rows.cuh"

namespace gdn_qsa::wy {
struct PrepareStorage {
  alignas(128) BF16 k[Chunk * Dim], v[Chunk * Dim];
  alignas(128) float lower[Chunk * Chunk], inverse[Chunk * Chunk];
  union {
    alignas(128) float temp[4][16 * 16];
    alignas(128) BF16 packed[4][2 * 16 * 16];  // solve scratch is dead during W/U
  };
  float prefix[Chunk], beta[Chunk];
};

// Shared arithmetic authority for legacy and tiled prepare variants.
// Same four-warp solve, TF32 high/residual products and BF16 boundaries.
template <bool Address = false, int RowCache = 0>
__device__ __forceinline__ int64_t prepare_inverse(Inputs const& p, Workspace const& ws, PrepareStorage& sm) {
  static_assert(RowCache >= 0 && RowCache <= 2 && (!RowCache || Address));
  int const tid = int(threadIdx.x), warp = unsigned(threadIdx.x) >> 5, lane = unsigned(threadIdx.x) & 31;
  int const ct = int(blockIdx.x) % p.shape.chunks();
  int const bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int const qh = p.shape.q_head(h);
  int64_t const group = p.shape.group(b, h, ct);
  if constexpr (Address) {
    state_stage<Chunk, Dim, ParallelThreads>(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
                                           int64_t(p.shape.q_heads) * Dim, valid);
    state_stage<Chunk, Dim, ParallelThreads>(sm.v, p.v + p.shape.input(b, first, h, p.shape.value_heads),
                                           int64_t(p.shape.value_heads) * Dim, valid);
  } else {
    stage<Chunk, Dim, ParallelThreads>(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
                                      int64_t(p.shape.q_heads) * Dim, valid);
    stage<Chunk, Dim, ParallelThreads>(sm.v, p.v + p.shape.input(b, first, h, p.shape.value_heads),
                                      int64_t(p.shape.value_heads) * Dim, valid);
  }
  if (tid < Chunk) {
    int64_t const gi = (int64_t(b) * p.shape.sequence + first + min(tid, valid - 1))
                      * p.shape.value_heads + h;
    float g = tid < valid ? gate(p, gi) : 0.0f;
    CUTE_UNROLL
    for (int offset = 1; offset < 32; offset *= 2) {
      float const other = __shfl_up_sync(0xffffffffu, g, offset);
      if (lane >= offset) g += other;
    }
    sm.prefix[tid] = g;
    sm.beta[tid] = tid < valid ? float(p.beta[gi]) : 0.0f;
  }
  commit_wait();
  if (tid >= 32 && tid < Chunk) sm.prefix[tid] += sm.prefix[31];
  __syncthreads();
  if (tid < Chunk) ws.gates[group * Chunk + tid] = sm.prefix[tid];

  // Ten lower triangular 16x16 products; upper entries are never read.
  for (int tile = warp; tile < 16; tile += 4) {
    int const br = tile / 4, bc = tile % 4;
    if (bc > br) continue;
    float acc[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t a[4], bt[4];
      load<Chunk, Dim>(sm.k, br * 16, k, a);
      load<Chunk, Dim>(sm.k, bc * 16, k, bt);
      bf16_mma(acc, a, bt);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const r = br * 16 + rc.row, c = bc * 16 + rc.col;
      sm.lower[r * Chunk + c] = r > c
          ? acc[s] * sm.beta[r] * expf(sm.prefix[r] - sm.prefix[c]) : 0.0f;
    }
  }
  __syncthreads();

  // Four independent unit-lower 16x16 diagonal solves in FP32. Each active
  // lane owns one entire inverse column; no CTA barrier per scalar row.
  float column[16];
  CUTE_UNROLL
  for (int r = 0; r < 16; ++r) {
    float x = r == (lane % 16) ? 1.0f : 0.0f;
    CUTE_UNROLL
    for (int k = 0; k < r; ++k)
      x -= sm.lower[(warp * 16 + r) * Chunk + warp * 16 + k] * column[k];
    column[r] = x;
    if (lane < 16) sm.inverse[(warp * 16 + r) * Chunk + warp * 16 + lane] = x;
  }
  __syncthreads();
  // Block forward substitution: R_ij = -R_ii sum(A_ip R_pj).
  #pragma unroll 1
  for (int gap = 1; gap < 4; ++gap) {
    int const br = warp + gap, bc = warp;
    if (br < 4) {
      float acc[8] = {};
      #pragma unroll 1
      for (int k = bc; k < br; ++k)
        tf32_product(acc, sm.lower + br * 16 * Chunk + k * 16, Chunk,
                     sm.inverse + k * 16 * Chunk + bc * 16, Chunk);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.temp[warp][rc.row * 16 + rc.col] = acc[s];
      }
      __syncwarp();
      float merged[8] = {};
      tf32_product(merged, sm.inverse + br * 16 * Chunk + br * 16, Chunk, sm.temp[warp], 16);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.inverse[(br * 16 + rc.row) * Chunk + bc * 16 + rc.col] = -merged[s];
      }
    }
    __syncthreads();
  }
  // All inverse-merge readers have retired at the preceding CTA barrier.
  // Reuse the dead temp scratch, not live prefix/beta/inverse storage.
  float warp_rows[PrepareRowsPlan::Slots];
  if constexpr (RowCache == 1) {
    static_assert(sizeof(sm.temp[0]) >= Chunk * sizeof(float));
    if (PrepareRowsPlan::shared_writer(unsigned(tid)))
      sm.temp[0][tid] = expf(sm.prefix[tid]);
  } else if constexpr (RowCache == 2) {
    CUTE_UNROLL
    for (unsigned slot = 0; slot < PrepareRowsPlan::Slots; ++slot)
      warp_rows[slot] = expf(sm.prefix[PrepareRowsPlan::producer_row(unsigned(lane), slot)]);
  }
  // lower is dead; reuse its storage for the BF16 inverse consumed by W/U.
  BF16* inv = reinterpret_cast<BF16*>(sm.lower);
  if constexpr (Address) {
    #pragma unroll 1
    for (unsigned it = 0; it < PrepareElementPlan::InverseIterations; ++it) {
      unsigned const i = PrepareElementPlan::inverse(unsigned(tid), it);
      unsigned const r = i / Chunk, c = i % Chunk;
      inv[state_shared_offset<Chunk, Chunk>(r, c)] = BF16(r >= c ? sm.inverse[i] : 0.0f);
    }
    if constexpr (RowCache == 1) __syncthreads();  // row producers -> all K-column consumers
    #pragma unroll 1
    for (unsigned it = 0; it < PrepareElementPlan::ValueIterations; ++it) {
      unsigned const r = PrepareElementPlan::row(it), c = PrepareElementPlan::column(unsigned(tid));
      unsigned const at = state_shared_offset<Chunk, Dim>(r, c);
      if constexpr (RowCache != 0) {
        float factor;
        if constexpr (RowCache == 1) factor = sm.temp[0][r];
        else {
          // All 32 lanes participate in every iteration. Selecting a register
          // before the shuffle avoids a dynamically indexed register array.
          float const owned = r < PrepareRowsPlan::Warp ? warp_rows[0] : warp_rows[1];
          factor = __shfl_sync(PrepareRowsPlan::FullMask, owned, PrepareRowsPlan::lane(r));
        }
        // Do NOT precombine beta*factor: the old multiplication order matters.
        sm.k[at] = BF16(float(sm.k[at]) * sm.beta[r] * factor);
      } else {
        sm.k[at] = BF16(float(sm.k[at]) * sm.beta[r] * expf(sm.prefix[r]));
      }
      sm.v[at] = BF16(float(sm.v[at]) * sm.beta[r]);
    }
  } else {
    for (int i = tid; i < Chunk * Chunk; i += ParallelThreads) {
      int const r = i / Chunk, c = i % Chunk;
      inv[swizzle<Chunk, Chunk>(r, c)] = BF16(r >= c ? sm.inverse[i] : 0.0f);
    }
    for (int i = tid; i < Chunk * Dim; i += ParallelThreads) {
      int const r = i / Dim, c = i % Dim, at = swizzle<Chunk, Dim>(r, c);
      sm.k[at] = BF16(float(sm.k[at]) * sm.beta[r] * expf(sm.prefix[r]));
      sm.v[at] = BF16(float(sm.v[at]) * sm.beta[r]);
    }
  }
  __syncthreads();
  return group;
}
}  // namespace gdn_qsa::wy
