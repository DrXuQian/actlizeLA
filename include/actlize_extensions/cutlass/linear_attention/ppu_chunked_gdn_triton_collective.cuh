/***************************************************************************************************
 * Copyright (c) 2026 actlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Four-kernel forward DAG matching flash-linear-attention's post-cumsum GDN
 * boundary:
 *   1. KKT + solve_tril
 *   2. recompute W and U together
 *   3. serial H recurrence with the FP32 state resident across chunks
 *   4. O with QH and causal QK recomputed at the point of use
 *
 * This is deliberately independent from the legacy A/W/P collective seam.
 * gamma_log2_cumsum is already the output of Triton's first (cumsum) kernel,
 * so that preprocessing launch remains outside this ABI on both paths.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh"

#if defined(__CUDACC__) && !defined(__HGGCCC__)
#define ACTLIZE_GDN_TRITON_DEVICE __device__ __forceinline__
#else
#define ACTLIZE_GDN_TRITON_DEVICE CUTLASS_DEVICE
#endif

namespace cutlass::linear_attention {

template <class Traits_, class Arguments_>
struct PpuChunkedGdnTritonCollectiveBf16C64D128BV64
    : PpuChunkedGdnCollectiveBf16C64D128BV64<Traits_, Arguments_> {
  using Base =
      PpuChunkedGdnCollectiveBf16C64D128BV64<Traits_, Arguments_>;
  using Traits = Traits_;
  using Arguments = Arguments_;
  using Params = Arguments;
  using Element = typename Base::Element;
  using BaseSharedStorageView = typename Base::SharedStorageView;

  static constexpr int kChunk = Base::kChunk;
  static constexpr int kHeadK = Base::kHeadK;
  static constexpr int kHeadV = Base::kHeadV;
  static constexpr int kValueBlock = Base::kValueBlock;
  static constexpr int kValueBlocks = Base::kValueBlocks;
  static constexpr int kThreadCount = Base::kThreadCount;
  static constexpr int kScoreElements = Base::kScoreElements;
  static constexpr int kWElements = Base::kWElements;
  static constexpr int kValueTileElements = Base::kValueTileElements;
  static constexpr int kStateTileElements = kHeadK * kValueBlock;

  // One record is exactly Triton's five post-cumsum tensors for one
  // (sequence, V-head, chunk): A, W, U, H-at-chunk-start, and Vnew.  P/QK is
  // intentionally absent; the O kernel recomputes it on chip.
  struct alignas(16) ChunkWorkspace {
    Element inverse[kScoreElements];               // A    [C,C]
    Element w[kWElements];                         // W    [C,K]
    Element u[kChunk * kHeadV];                    // U    [C,V]
    Element h_start[kHeadK * kHeadV];              // H    [K,V]
    Element v_new[kChunk * kHeadV];                // Vnew [C,V]
  };

  static constexpr int kChunkWorkspaceBytes = int(sizeof(ChunkWorkspace));
  static_assert(kChunkWorkspaceBytes == 90112,
                "Triton-aligned A/W/U/H/Vnew seam must be exactly 88 KiB");

  // KKT uses the proved global-dot and blocked inverse, but no longer computes
  // QK/P or W.  The 32-KiB arena is alternately the AIU mainloop, strict lower
  // matrix, FP32 inverse and final BF16 A.
  struct alignas(32) KktSharedStorage {
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[32 * 1024];

    ACTLIZE_GDN_TRITON_DEVICE operator BaseSharedStorageView() {
      return BaseSharedStorageView{nullptr, gamma, beta, phase};
    }
  };

  // W and U share one generated BF16 C64x64 operand.  A is consumed directly
  // from its global seam, as in Triton's recompute_w_u kernel.
  struct alignas(32) WuSharedStorage {
    float gamma[kChunk];
    float beta[kChunk];
    Element operand[kScoreElements];
  };

  // On PPU the recurrent state is two FP32 accumulator fragments and therefore
  // does not appear here.  The one BF16 tile is only a tensor-fragment layout
  // conversion for WH, then is reused for gated Vnew.  NVIDIA's independent
  // scalar oracle needs an explicit state array because it has no PPU fragment.
  struct alignas(32) HSharedStorage {
    float gamma[kChunk];
    float beta[kChunk];
    Element operand[kScoreElements];
#if !defined(__HGGCCC__)
    float state[kStateTileElements];
#endif
  };

  struct alignas(32) OSharedStorage {
    float gamma[kChunk];
    // Kept beside gamma because the common scalar loader is deliberately the
    // one authority for the caller ABI.  O does not consume beta afterwards.
    float beta[kChunk];
    Element causal[kScoreElements];
  };

  static_assert(sizeof(KktSharedStorage) == 33280 &&
                    sizeof(WuSharedStorage) == 8704 &&
                    sizeof(OSharedStorage) == 8704,
                "Triton-aligned stage-local shared ledgers changed");
#if defined(__HGGCCC__)
  static_assert(sizeof(HSharedStorage) == 8704,
                "PPU H state must remain register-resident");
#else
  static_assert(sizeof(HSharedStorage) == 41472,
                "CUDA scalar H authority must own one explicit K128xBV64 state");
#endif

  CUTLASS_HOST_DEVICE static constexpr std::int64_t workspace_index(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_idx) {
    return ((std::int64_t(work.sequence_idx) * params.problem.num_v_heads +
             work.v_head_idx) *
                work.chunk_count +
            chunk_idx);
  }

 private:
  template <class Storage>
  ACTLIZE_GDN_TRITON_DEVICE static int load_scalars(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_idx, Storage& shared, int thread_idx) {
    int const chunk_begin = chunk_idx * kChunk;
    int const valid = work.token_count - chunk_begin < kChunk
                          ? work.token_count - chunk_begin
                          : kChunk;
    for (int row = thread_idx; row < kChunk; row += kThreadCount) {
      if (row < valid) {
        std::int64_t const token =
            std::int64_t(work.token_begin) + chunk_begin + row;
        std::int64_t const gh =
            token * params.problem.num_v_heads + work.v_head_idx;
        shared.gamma[row] = params.gamma_log2_cumsum[gh];
        shared.beta[row] = params.beta[gh];
      } else {
        shared.gamma[row] = 0.0f;
        shared.beta[row] = 0.0f;
      }
    }
    __syncthreads();
    return valid;
  }

  ACTLIZE_GDN_TRITON_DEVICE static void scalar_wu(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, ChunkWorkspace& dst,
      WuSharedStorage& shared, int thread_idx) {
    for (int index = thread_idx; index < kWElements; index += kThreadCount) {
      int const row = index / kHeadK;
      int const feature = index % kHeadK;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        Element const scaled_k = Base::to_bf16(
            j < valid
                ? shared.beta[j] * Base::exp2_gate(shared.gamma[j]) *
                      Base::to_float(params.k[Base::qk_offset(
                          params, work, chunk_begin + j, feature)])
                : 0.0f);
        sum += Base::to_float(dst.inverse[row * kChunk + j]) *
               Base::to_float(scaled_k);
      }
      dst.w[index] = Base::to_bf16(sum);
    }
    for (int index = thread_idx; index < kChunk * kHeadV;
         index += kThreadCount) {
      int const row = index / kHeadV;
      int const value = index % kHeadV;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        Element const scaled_v = Base::to_bf16(
            j < valid
                ? shared.beta[j] * Base::to_float(params.v[Base::vo_offset(
                      params, work, chunk_begin + j, value)])
                : 0.0f);
        sum += Base::to_float(dst.inverse[row * kChunk + j]) *
               Base::to_float(scaled_v);
      }
      dst.u[index] = Base::to_bf16(sum);
    }
    __syncthreads();
  }

#if defined(__HGGCCC__)
  using ResidentMma = typename Base::ResidentMma;
  using StateFragment = typename ResidentMma::Accumulator;

  ACTLIZE_GDN_TRITON_DEVICE static void ppu0010_wu(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, ChunkWorkspace& dst,
      WuSharedStorage& shared, int thread_idx) {
#pragma unroll
    for (int feature_base = 0; feature_base < kHeadK;
         feature_base += kChunk) {
      for (int index = thread_idx; index < kScoreElements;
           index += kThreadCount) {
        int const feature = index / kChunk;
        int const row = index % kChunk;
        shared.operand[index] = Base::to_bf16(
            row < valid
                ? shared.beta[row] * Base::exp2_gate(shared.gamma[row]) *
                      Base::to_float(params.k[Base::qk_offset(
                          params, work, chunk_begin + row,
                          feature_base + feature)])
                : 0.0f);
      }
      __syncthreads();
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      ResidentMma::mma(
          accum,
          dst.inverse, kChunk, 1,
          shared.operand, kChunk, 1,
          thread_idx, valid, kChunk, valid);
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int feature, float value) {
            dst.w[row * kHeadK + feature_base + feature] =
                Base::to_bf16(value);
          },
          valid, kChunk);
      __syncthreads();
    }

#pragma unroll
    for (int value_base = 0; value_base < kHeadV;
         value_base += kValueBlock) {
      for (int index = thread_idx; index < kScoreElements;
           index += kThreadCount) {
        int const value = index / kChunk;
        int const row = index % kChunk;
        shared.operand[index] = Base::to_bf16(
            row < valid
                ? shared.beta[row] * Base::to_float(params.v[Base::vo_offset(
                      params, work, chunk_begin + row, value_base + value)])
                : 0.0f);
      }
      __syncthreads();
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      ResidentMma::mma(
          accum,
          dst.inverse, kChunk, 1,
          shared.operand, kChunk, 1,
          thread_idx, valid, kValueBlock, valid);
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float product) {
            dst.u[row * kHeadV + value_base + value] =
                Base::to_bf16(product);
          },
          valid, kValueBlock);
      __syncthreads();
    }
  }

  ACTLIZE_GDN_TRITON_DEVICE static void load_state_fragment(
      StateFragment& state, Params const& params,
      PpuChunkedGdnWorkTileInfo const& work, int feature_base,
      int thread_idx) {
    ResidentMma::for_each_c_coordinate(
        thread_idx, [&](int slot, int feature, int value) {
          auto const domain = cute::idx2crd(slot, cute::shape(state));
          state(domain) =
              params.initial_state == nullptr
                  ? 0.0f
                  : params.initial_state[Base::state_offset(
                        params, work, feature_base + feature,
                        work.value_begin + value)];
        });
  }

  ACTLIZE_GDN_TRITON_DEVICE static void publish_state_fragment(
      StateFragment const& state, ChunkWorkspace& chunk,
      Element* operand, int feature_base, int value_base,
      int thread_idx) {
    ResidentMma::for_each_c_coordinate(
        thread_idx, [&](int slot, int feature, int value) {
          auto const domain = cute::idx2crd(slot, cute::shape(state));
          Element const rounded = Base::to_bf16(state(domain));
          operand[feature * kValueBlock + value] = rounded;
          chunk.h_start[(feature_base + feature) * kHeadV +
                        value_base + value] = rounded;
        });
  }

  ACTLIZE_GDN_TRITON_DEVICE static void update_state_fragment(
      StateFragment& state, StateFragment const& update,
      float decay, int thread_idx) {
    ResidentMma::for_each_c_coordinate(
        thread_idx, [&](int slot, int, int) {
          auto const domain = cute::idx2crd(slot, cute::shape(state));
          state(domain) = decay * state(domain) + update(domain);
        });
  }

  ACTLIZE_GDN_TRITON_DEVICE static void store_state_fragment(
      StateFragment const& state, Params const& params,
      PpuChunkedGdnWorkTileInfo const& work, int feature_base,
      int thread_idx) {
    if (params.final_state == nullptr) return;
    ResidentMma::for_each_c_coordinate(
        thread_idx, [&](int slot, int feature, int value) {
          auto const domain = cute::idx2crd(slot, cute::shape(state));
          params.final_state[Base::state_offset(
              params, work, feature_base + feature,
              work.value_begin + value)] = state(domain);
        });
  }

  ACTLIZE_GDN_TRITON_DEVICE static void ppu0010_h(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      ChunkWorkspace* workspace, HSharedStorage& shared, int thread_idx) {
    StateFragment state0 = ResidentMma::make_accumulator();
    StateFragment state1 = ResidentMma::make_accumulator();
    load_state_fragment(state0, params, work, 0, thread_idx);
    load_state_fragment(state1, params, work, kChunk, thread_idx);

    std::int64_t const qk_pitch =
        std::int64_t(params.problem.num_qk_heads) * kHeadK;
    for (int chunk_idx = 0; chunk_idx < work.chunk_count; ++chunk_idx) {
      int const chunk_begin = chunk_idx * kChunk;
      int const valid =
          load_scalars(params, work, chunk_idx, shared, thread_idx);
      ChunkWorkspace& chunk =
          workspace[workspace_index(params, work, chunk_idx)];

      auto wh = ResidentMma::make_accumulator();
      ResidentMma::clear(wh);
      publish_state_fragment(
          state0, chunk, shared.operand, 0, work.value_begin, thread_idx);
      __syncthreads();
      ResidentMma::mma(
          wh,
          chunk.w, kHeadK, 1,
          shared.operand, 1, kValueBlock,
          thread_idx, valid, kValueBlock, kChunk);
      __syncthreads();
      publish_state_fragment(
          state1, chunk, shared.operand, kChunk, work.value_begin, thread_idx);
      __syncthreads();
      ResidentMma::mma(
          wh,
          chunk.w + kChunk, kHeadK, 1,
          shared.operand, 1, kValueBlock,
          thread_idx, valid, kValueBlock, kChunk);
      __syncthreads();

      float const gamma_last = shared.gamma[valid - 1];
      ResidentMma::for_each_c_coordinate(
          thread_idx, [&](int slot, int row, int value) {
            if (row >= valid) return;
            auto const domain = cute::idx2crd(slot, cute::shape(wh));
            float const raw =
                Base::to_float(chunk.u[row * kHeadV + work.value_begin + value]) -
                wh(domain);
            chunk.v_new[row * kHeadV + work.value_begin + value] =
                Base::to_bf16(raw);
            shared.operand[row * kValueBlock + value] = Base::to_bf16(
                Base::exp2_gate(gamma_last - shared.gamma[row]) * raw);
          });
      __syncthreads();

      auto update0 = ResidentMma::make_accumulator();
      ResidentMma::clear(update0);
      Element const* const k0 =
          params.k + Base::qk_offset(params, work, chunk_begin, 0);
      ResidentMma::mma(
          update0,
          k0, 1, qk_pitch,
          shared.operand, 1, kValueBlock,
          thread_idx, kChunk, kValueBlock, valid);
      update_state_fragment(
          state0, update0, Base::exp2_gate(gamma_last), thread_idx);

      auto update1 = ResidentMma::make_accumulator();
      ResidentMma::clear(update1);
      Element const* const k1 =
          params.k + Base::qk_offset(params, work, chunk_begin, kChunk);
      ResidentMma::mma(
          update1,
          k1, 1, qk_pitch,
          shared.operand, 1, kValueBlock,
          thread_idx, kChunk, kValueBlock, valid);
      update_state_fragment(
          state1, update1, Base::exp2_gate(gamma_last), thread_idx);
    }

    store_state_fragment(state0, params, work, 0, thread_idx);
    store_state_fragment(state1, params, work, kChunk, thread_idx);
  }

  ACTLIZE_GDN_TRITON_DEVICE static void ppu0010_o(
      Params const& params,
      PpuChunkedGdnValueChunkWorkTileInfo const& output_work,
      ChunkWorkspace const& chunk, OSharedStorage& shared,
      int valid, int thread_idx) {
    PpuChunkedGdnWorkTileInfo const& work = output_work.head;
    int const chunk_begin = output_work.chunk_idx * kChunk;
    int const value_base = work.value_begin;
    std::int64_t const qk_pitch =
        std::int64_t(params.problem.num_qk_heads) * kHeadK;

    auto qh = ResidentMma::make_accumulator();
    ResidentMma::clear(qh);
#pragma unroll
    for (int feature_base = 0; feature_base < kHeadK;
         feature_base += kChunk) {
      Element const* const q =
          params.q + Base::qk_offset(params, work, chunk_begin, feature_base);
      Element const* const h =
          chunk.h_start + feature_base * kHeadV + value_base;
      ResidentMma::mma(
          qh,
          q, qk_pitch, 1,
          h, 1, kHeadV,
          thread_idx, valid, kValueBlock, kChunk);
    }

    {
      auto qk = ResidentMma::make_accumulator();
      ResidentMma::clear(qk);
#pragma unroll
      for (int feature_base = 0; feature_base < kHeadK;
           feature_base += kChunk) {
        Element const* const q =
            params.q + Base::qk_offset(params, work, chunk_begin, feature_base);
        Element const* const k =
            params.k + Base::qk_offset(params, work, chunk_begin, feature_base);
        ResidentMma::mma(
            qk,
            q, qk_pitch, 1,
            k, qk_pitch, 1,
            thread_idx, valid, valid, kChunk);
      }
      ResidentMma::for_each_c_coordinate(
          thread_idx, [&](int slot, int row, int column) {
            if (row >= valid || column >= valid) return;
            auto const domain = cute::idx2crd(slot, cute::shape(qk));
            shared.causal[row * kChunk + column] = Base::to_bf16(
                row >= column
                    ? qk(domain) * Base::exp2_gate(
                          shared.gamma[row] - shared.gamma[column])
                    : 0.0f);
          });
    }
    __syncthreads();

    auto causal = ResidentMma::make_accumulator();
    ResidentMma::clear(causal);
    ResidentMma::mma(
        causal,
        shared.causal, kChunk, 1,
        chunk.v_new + value_base, 1, kHeadV,
        thread_idx, valid, kValueBlock, valid);
    ResidentMma::for_each_c_coordinate(
        thread_idx, [&](int slot, int row, int value) {
          if (row >= valid) return;
          auto const domain = cute::idx2crd(slot, cute::shape(qh));
          float const result = params.scale *
              (Base::exp2_gate(shared.gamma[row]) * qh(domain) +
               causal(domain));
          params.output[Base::vo_offset(
              params, work, chunk_begin + row, value_base + value)] =
              Base::to_bf16(result);
        });
  }
#endif  // defined(__HGGCCC__)

  ACTLIZE_GDN_TRITON_DEVICE static void scalar_h(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      ChunkWorkspace* workspace, HSharedStorage& shared, int thread_idx) {
#if !defined(__HGGCCC__)
    for (int index = thread_idx; index < kStateTileElements;
         index += kThreadCount) {
      int const feature = index / kValueBlock;
      int const value = index % kValueBlock;
      shared.state[index] =
          params.initial_state == nullptr
              ? 0.0f
              : params.initial_state[Base::state_offset(
                    params, work, feature, work.value_begin + value)];
    }
    __syncthreads();
    for (int chunk_idx = 0; chunk_idx < work.chunk_count; ++chunk_idx) {
      int const chunk_begin = chunk_idx * kChunk;
      int const valid =
          load_scalars(params, work, chunk_idx, shared, thread_idx);
      ChunkWorkspace& chunk =
          workspace[workspace_index(params, work, chunk_idx)];
      for (int index = thread_idx; index < kStateTileElements;
           index += kThreadCount) {
        int const feature = index / kValueBlock;
        int const value = index % kValueBlock;
        chunk.h_start[feature * kHeadV + work.value_begin + value] =
            Base::to_bf16(shared.state[index]);
      }
      __syncthreads();
      float const gamma_last = shared.gamma[valid - 1];
      for (int index = thread_idx; index < kValueTileElements;
           index += kThreadCount) {
        int const row = index / kValueBlock;
        int const value = index % kValueBlock;
        if (row >= valid) continue;
        float wh = 0.0f;
        for (int feature = 0; feature < kHeadK; ++feature) {
          wh += Base::to_float(chunk.w[row * kHeadK + feature]) *
                Base::to_float(chunk.h_start[
                    feature * kHeadV + work.value_begin + value]);
        }
        float const raw =
            Base::to_float(chunk.u[row * kHeadV + work.value_begin + value]) - wh;
        chunk.v_new[row * kHeadV + work.value_begin + value] =
            Base::to_bf16(raw);
        shared.operand[index] = Base::to_bf16(
            Base::exp2_gate(gamma_last - shared.gamma[row]) * raw);
      }
      __syncthreads();
      float const decay = Base::exp2_gate(gamma_last);
      for (int index = thread_idx; index < kStateTileElements;
           index += kThreadCount) {
        int const feature = index / kValueBlock;
        int const value = index % kValueBlock;
        float update = 0.0f;
        for (int row = 0; row < valid; ++row) {
          update += Base::to_float(params.k[Base::qk_offset(
                        params, work, chunk_begin + row, feature)]) *
                    Base::to_float(shared.operand[
                        row * kValueBlock + value]);
        }
        shared.state[index] = decay * shared.state[index] + update;
      }
      __syncthreads();
    }
    if (params.final_state != nullptr) {
      for (int index = thread_idx; index < kStateTileElements;
           index += kThreadCount) {
        int const feature = index / kValueBlock;
        int const value = index % kValueBlock;
        params.final_state[Base::state_offset(
            params, work, feature, work.value_begin + value)] =
            shared.state[index];
      }
    }
#else
    (void)params;
    (void)work;
    (void)workspace;
    (void)shared;
    (void)thread_idx;
#endif
  }

  ACTLIZE_GDN_TRITON_DEVICE static void scalar_o(
      Params const& params,
      PpuChunkedGdnValueChunkWorkTileInfo const& output_work,
      ChunkWorkspace const& chunk, OSharedStorage& shared,
      int valid, int thread_idx) {
    PpuChunkedGdnWorkTileInfo const& work = output_work.head;
    int const chunk_begin = output_work.chunk_idx * kChunk;
    for (int index = thread_idx; index < kValueTileElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      int const value = index % kValueBlock;
      if (row >= valid) continue;
      float qh = 0.0f;
      for (int feature = 0; feature < kHeadK; ++feature) {
        qh += Base::to_float(params.q[Base::qk_offset(
                  params, work, chunk_begin + row, feature)]) *
              Base::to_float(chunk.h_start[
                  feature * kHeadV + work.value_begin + value]);
      }
      float causal = 0.0f;
      for (int column = 0; column <= row; ++column) {
        float qk = 0.0f;
        for (int feature = 0; feature < kHeadK; ++feature) {
          qk += Base::to_float(params.q[Base::qk_offset(
                    params, work, chunk_begin + row, feature)]) *
                Base::to_float(params.k[Base::qk_offset(
                    params, work, chunk_begin + column, feature)]);
        }
        Element const p = Base::to_bf16(
            qk * Base::exp2_gate(
                     shared.gamma[row] - shared.gamma[column]));
        causal += Base::to_float(p) * Base::to_float(chunk.v_new[
            column * kHeadV + work.value_begin + value]);
      }
      params.output[Base::vo_offset(
          params, work, chunk_begin + row, work.value_begin + value)] =
          Base::to_bf16(params.scale *
              (Base::exp2_gate(shared.gamma[row]) * qh + causal));
    }
    __syncthreads();
  }

 public:
  static constexpr PpuChunkedGdnStatus argument_status(Arguments const& args) {
    return Base::argument_status(args);
  }

  ACTLIZE_GDN_TRITON_DEVICE static void run_kkt_solve(
      Params const& params, PpuChunkedGdnPrepareWorkTileInfo const& prepare,
      ChunkWorkspace* workspace, KktSharedStorage& shared) {
    int const thread_idx = int(threadIdx.x);
    if (!prepare.valid || workspace == nullptr) return;
    PpuChunkedGdnWorkTileInfo const& work = prepare.head;
    int const chunk_begin = prepare.chunk_idx * kChunk;
    BaseSharedStorageView view = shared;
    int const valid =
        Base::load_chunk_scalars(params, work, prepare.chunk_idx, view, thread_idx);
    Base::global_dot(
        params, work, chunk_begin, valid,
        detail::PpuChunkedGdnGlobalDotKind::kStrictLowerKk,
        view, thread_idx);
    Base::solve_inverse(view, thread_idx);
    ChunkWorkspace& dst =
        workspace[workspace_index(params, work, prepare.chunk_idx)];
    Element const* const inverse = Base::inverse_bf16(view);
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      dst.inverse[index] = inverse[index];
    }
  }

  ACTLIZE_GDN_TRITON_DEVICE static void run_wu(
      Params const& params, PpuChunkedGdnPrepareWorkTileInfo const& prepare,
      ChunkWorkspace* workspace, WuSharedStorage& shared) {
    int const thread_idx = int(threadIdx.x);
    if (!prepare.valid || workspace == nullptr) return;
    PpuChunkedGdnWorkTileInfo const& work = prepare.head;
    int const chunk_begin = prepare.chunk_idx * kChunk;
    int const valid =
        load_scalars(params, work, prepare.chunk_idx, shared, thread_idx);
    ChunkWorkspace& dst =
        workspace[workspace_index(params, work, prepare.chunk_idx)];
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_wu(params, work, chunk_begin, valid, dst, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("Triton-aligned W/U requires ppu0010");
#else
    scalar_wu(params, work, chunk_begin, valid, dst, shared, thread_idx);
#endif
  }

  ACTLIZE_GDN_TRITON_DEVICE static void run_h(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      ChunkWorkspace* workspace, HSharedStorage& shared) {
    int const thread_idx = int(threadIdx.x);
    if (!work.valid || workspace == nullptr ||
        work.value_count != kValueBlock) {
      return;
    }
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_h(params, work, workspace, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("Triton-aligned H requires ppu0010");
#else
    scalar_h(params, work, workspace, shared, thread_idx);
#endif
  }

  ACTLIZE_GDN_TRITON_DEVICE static void run_o(
      Params const& params,
      PpuChunkedGdnValueChunkWorkTileInfo const& output_work,
      ChunkWorkspace const* workspace, OSharedStorage& shared) {
    int const thread_idx = int(threadIdx.x);
    if (!output_work.valid || workspace == nullptr) return;
    PpuChunkedGdnWorkTileInfo const& work = output_work.head;
    int const valid = load_scalars(
        params, work, output_work.chunk_idx, shared, thread_idx);
    ChunkWorkspace const& chunk =
        workspace[workspace_index(params, work, output_work.chunk_idx)];
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_o(params, output_work, chunk, shared, valid, thread_idx);
#elif defined(__HGGC_ARCH__)
    CUTE_INVALID_CONTROL_PATH("Triton-aligned O requires ppu0010");
#else
    scalar_o(params, output_work, chunk, shared, valid, thread_idx);
#endif
  }
};

}  // namespace cutlass::linear_attention

#undef ACTLIZE_GDN_TRITON_DEVICE
