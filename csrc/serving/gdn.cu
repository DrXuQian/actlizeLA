// Recovery recurrence and compact records follow Flare gdn_recovery.py.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0
#include "abi.h"
#include "aux_common.cuh"
#include "gdn_transaction_v2.h"
#include "runtime.cuh"
#include "prefill_schedule.h"
#include <climits>

namespace actlize_la::serving {
__device__ bool valid_range(Gdn const &p, int r) {
  return p.starts[r] >= 0 && p.starts[r + 1] >= p.starts[r] &&
         p.starts[r + 1] <= p.tokens &&
         p.starts[r + 1] - p.starts[r] <= p.width;
}
__global__ void validate_gdn(Gdn p, ala_gdn_transaction_v2 guard) {
  int const r = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= p.requests)
    return;
  int error = Ok, live = *p.live;
  if (live < 0 || live > p.requests)
    error = Bounds;
  else if (r < live) {
    if (!valid_range(p, r))
      error = Bounds;
    else if ((p.checkpoint_slots && p.checkpoint_slots[r] != 0) ||
             (p.checkpoint_offsets && p.checkpoint_offsets[r] != 0))
      error = Unsupported;
    else {
      auto src = p.initial_slots[int64_t(r) * p.initial_stride];
      auto dst = p.final_slots[int64_t(r) * p.final_stride];
      if (src < 0 || src >= p.slots || dst < 0 || dst >= p.slots)
        error = Slot;
      // Null decode/verify is inactive; prefill may read zero and write live.
      bool const active = src >= 0 && src < p.slots && dst >= 0 &&
                          dst < p.slots && p.starts[r + 1] > p.starts[r] &&
                          (p.mode == ChunkPrefill ? dst > 0 : src > 0);
      bool const reads = active && src > 0;
      bool const writes = active && p.mode != Verify && dst > 0;
      for (int other = 0; other < live; ++other)
        if (other != r && valid_range(p, other) &&
            p.starts[other + 1] > p.starts[other]) {
          auto os = p.initial_slots[int64_t(other) * p.initial_stride];
          auto od = p.final_slots[int64_t(other) * p.final_stride];
          bool const oa = os >= 0 && os < p.slots && od >= 0 && od < p.slots &&
                          (p.mode == ChunkPrefill ? od > 0 : os > 0);
          bool const other_reads = oa && os > 0;
          bool const other_writes = oa && p.mode != Verify && od > 0;
          if (writes &&
              ((other_writes && dst == od) || (other_reads && dst == os)))
            error = Conflict;
          if (reads && other_writes && src == od)
            error = Conflict;
        }
    }
  }
  if (!error && r < live && guard.header.version == 2) {
    for (int j = 0; j < live; ++j)
      if (j != r && valid_range(p, j) && p.starts[j + 1] > p.starts[j] &&
          p.starts[r + 1] > p.starts[r] && p.starts[r] < p.starts[j + 1] &&
          p.starts[j] < p.starts[r + 1])
        error = Conflict;
    int dep = guard.upstream_errors.data
                  ? aux::at<int32_t>(guard.upstream_errors, r)
                  : 0;
    if (!error && dep)
      error = dep > 0 && dep <= Unsupported ? dep : Bounds;
    else if (!error && guard.epochs.data && p.starts[r + 1] > p.starts[r] &&
             p.initial_slots[int64_t(r) * p.initial_stride] > 0) {
      int64_t epoch = aux::at<int64_t>(guard.epochs, r),
              ready = aux::at<int64_t>(guard.ready_epochs, r),
              used = aux::at<int64_t>(guard.consumed_epochs, r);
      if (epoch <= 0 || epoch <= ready || epoch <= used || ready > used)
        error = History;
    }
  }
  p.errors[r] = error;
  // V2 inactive/empty rows retain any outstanding record window. Clearing its
  // lengths while leaving ready_epoch pending would make later commit
  // impossible.
  bool const keep_records =
      p.mode == Verify && guard.header.version == 2 &&
      (error ||
       ((r >= live || !valid_range(p, r) || p.starts[r + 1] == p.starts[r] ||
         p.initial_slots[int64_t(r) * p.initial_stride] == 0) &&
        aux::at<int64_t>(guard.ready_epochs, r) >
            aux::at<int64_t>(guard.consumed_epochs, r)));
  if (p.mode == Verify && !keep_records) {
    bool const active = error == Ok && r < live &&
                        p.initial_slots[int64_t(r) * p.initial_stride] > 0;
    p.recorded_lengths[r] = active ? p.starts[r + 1] - p.starts[r] : 0;
    p.recorded_slots[r] =
        active ? p.initial_slots[int64_t(r) * p.initial_stride] : 0;
  }
}

template <bool TokenParallel = false>
__global__ void prepare_gdn(Gdn p) {
  int const r = blockIdx.x, h = blockIdx.y,
            lane = TokenParallel ? threadIdx.x % 32 : threadIdx.x;
  if (r >= *p.live || p.errors[r] || !valid_range(p, r))
    return;
  if (p.mode != ChunkPrefill &&
      p.initial_slots[int64_t(r) * p.initial_stride] == 0)
    return;
  if (p.mode == ChunkPrefill && p.final_slots[int64_t(r) * p.final_stride] == 0)
    return;
  int const first = p.starts[r], length = p.starts[r + 1] - first;
  // A whole warp still owns exactly one normalized head/token, including
  // the original four elements/lane and XOR reduction order. Only independent
  // tokens move to other warps/blocks; decode and verify instantiate false.
  int const begin = TokenParallel ? prefill_schedule::first(
      blockIdx.z, prefill_schedule::PrepareWarps, int(threadIdx.x) / 32) : 0;
  int const step = TokenParallel ? prefill_schedule::stride(
      gridDim.z, prefill_schedule::PrepareWarps) : 1;
  for (int t = begin; t < length; t += step) {
    int64_t const token = first + t;
    if (h < 2 * p.hk) {
      float x[4], square = 0;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        x[d] =
            read_bf(p.mixed, token * p.mixed_stride + h * 128 + lane * 4 + d);
        square += x[d] * x[d];
      }
      float factor = rsqrtf(warp_sum(square) + 1e-6f);
      // Native chunk output already owns Q scale. Decode owns it here once.
      if (h < p.hk && p.mode != ChunkPrefill)
        factor *= 0.08838834764831845f;
      auto dst = h < p.hk ? p.queries : p.keys;
      int head = h < p.hk ? h : h - p.hk;
#pragma unroll
      for (int d = 0; d < 4; ++d)
        dst[((int64_t(r) * p.hk + head) * p.width + t) * 128 + lane * 4 + d] =
            x[d] * factor;
    } else if (lane == 0) {
      int const head = h - 2 * p.hk;
      float const g = log_decay(read_bf(p.a, token * p.a_stride + head),
                                p.a_log[head], p.dt_bias[head]);
      int64_t const at = (int64_t(r) * p.hv + head) * p.width + t;
      p.decays[at] = p.mode == ChunkPrefill ? g : expf(g);
      static_cast<BFloat *>(p.betas)[at] =
          bf(1.0f / (1.0f + expf(-read_bf(p.b, token * p.b_stride + head))));
    }
  }
}

// One warp owns one V row, distributed across K. FP32 live H never leaves
// registers during verification. No per-token H snapshots and no atomics.
__global__ void recurrent_gdn(Gdn p) {
  int const r = blockIdx.x, h = blockIdx.y,
            v = blockIdx.z * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (r >= *p.live || p.errors[r] || !valid_range(p, r))
    return;
  int const first = p.starts[r], length = p.starts[r + 1] - first;
  auto const src = p.initial_slots[int64_t(r) * p.initial_stride];
  auto const dst = p.final_slots[int64_t(r) * p.final_stride];
  if ((src == 0 && p.mode != ChunkPrefill) ||
      (dst == 0 && p.mode == ChunkPrefill)) {
    if (!lane)
      for (int t = 0; t < length; ++t)
        static_cast<BFloat *>(
            p.output)[(int64_t(first + t) * p.hv + h) * 128 + v] = 0;
    return;
  }
  float state[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    state[k] = src ? p.state[src * p.state_stride +
                             (int64_t(h) * 128 + v) * 128 + lane * 4 + k]
                   : 0;
  int const kh = h / (p.hv / p.hk);
  for (int t = 0; t < length; ++t) {
    int64_t const gate = (int64_t(r) * p.hv + h) * p.width + t;
    float const decay =
        p.mode == ChunkPrefill ? expf(p.decays[gate]) : p.decays[gate];
    float const beta = read_bf(p.betas, gate);
    float key[4], query[4], dot = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int64_t const at =
          ((int64_t(r) * p.hk + kh) * p.width + t) * 128 + lane * 4 + k;
      key[k] = p.keys[at];
      query[k] = p.queries[at];
      state[k] = __fmul_rn(state[k], decay);
      dot += state[k] * key[k];
    }
    dot = warp_sum(dot);
    float const delta = (read_bf(p.mixed, int64_t(first + t) * p.mixed_stride +
                                              2 * p.hk * 128 + h * 128 + v) -
                         dot) *
                        beta;
    float out = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state[k] = __fadd_rn(state[k], __fmul_rn(delta, key[k]));
      out += state[k] * query[k];
    }
    out = warp_sum(out);
    if (!lane) {
      if (p.mode == ChunkPrefill)
        out *= 0.08838834764831845f;
      static_cast<BFloat *>(
          p.output)[(int64_t(first + t) * p.hv + h) * 128 + v] = bf(out);
      if (p.mode == Verify)
        p.delta[gate * 128 + v] = delta;
    }
  }
  if (p.mode != Verify && dst > 0 && length > 0) {
#pragma unroll
    for (int k = 0; k < 4; ++k)
      p.state[dst * p.state_stride + (int64_t(h) * 128 + v) * 128 + lane * 4 +
              k] = state[k];
  }
}

__global__ void validate_commit(Commit p, ala_gdn_transaction_v2 guard) {
  int const r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= p.requests)
    return;
  int error = Ok, live = *p.live;
  if (live < 0 || live > p.requests)
    error = Bounds;
  else if (r < live) {
    auto const slot = p.slots[int64_t(r) * p.slot_stride];
    int const count = p.counts[r];
    if (slot < 0 || slot >= p.state_slots)
      error = Slot;
    else if (count < 0 || count > p.width || count > p.recorded_lengths[r] ||
             (count > 0 && slot != p.recorded_slots[r]))
      error = Prefix;
    for (int j = 0; j < live; ++j)
      if (j != r && slot > 0 && slot == p.slots[int64_t(j) * p.slot_stride] &&
          count > 0 && p.counts[j] > 0)
        error = Conflict;
  }
  if (!error && r < live && guard.header.version == 2) {
    int dep = guard.upstream_errors.data
                  ? aux::at<int32_t>(guard.upstream_errors, r)
                  : 0;
    if (dep)
      error = dep > 0 && dep <= Unsupported ? dep : Bounds;
    else if (p.recorded_lengths[r] > 0) {
      int64_t epoch = aux::at<int64_t>(guard.epochs, r);
      if (p.slots[int64_t(r) * p.slot_stride] != p.recorded_slots[r] ||
          epoch <= 0 || epoch != aux::at<int64_t>(guard.ready_epochs, r) ||
          epoch <= aux::at<int64_t>(guard.consumed_epochs, r))
        error = History;
    }
  }
  p.errors[r] = error;
}
__global__ void commit_gdn(Commit p) {
  int const r = blockIdx.x, h = blockIdx.y;
  int const at = blockIdx.z * 256 + threadIdx.x, v = at / 128, k = at % 128;
  if (r >= *p.live || p.errors[r])
    return;
  auto const slot = p.slots[int64_t(r) * p.slot_stride];
  int const count = p.counts[r];
  if (!slot || !count)
    return;
  int64_t const dst = slot * p.state_stride + h * 16384 + at;
  float state = p.state[dst];
  for (int t = 0; t < count; ++t) {
    int64_t const gate = (int64_t(r) * p.hv + h) * p.width + t;
    float const key =
        p.keys[((int64_t(r) * p.hk + h / (p.hv / p.hk)) * p.width + t) * 128 +
               k];
    state = __fmul_rn(state, p.decays[gate]);
    state = __fadd_rn(state, __fmul_rn(p.delta[gate * 128 + v], key));
  }
  p.state[dst] = state;
}
bool valid(Gdn const &p) {
  return p.mixed && p.a && p.b && p.a_log && p.dt_bias && p.state && p.starts &&
         p.live && p.initial_slots && p.final_slots && p.output && p.queries &&
         p.keys && p.decays && p.betas && p.errors && p.requests > 0 &&
         p.tokens > 0 && p.slots > 1 && p.hk > 0 && p.hv > 0 &&
         p.hv % p.hk == 0 && int64_t(2) * p.hk + p.hv <= 65535 && p.width > 0 &&
         p.width <= INT_MAX / 128 && p.mode >= Decode &&
         p.mode <= ChunkPrefill &&
         p.mixed_stride >= int64_t(2 * p.hk + p.hv) * 128 &&
         p.a_stride >= p.hv && p.b_stride >= p.hv &&
         p.state_stride >= int64_t(p.hv) * 16384 && p.initial_stride > 0 &&
         p.final_stride > 0 && (p.mode != Decode || p.width == 1) &&
         (p.mode != Verify ||
          (p.width <= 8 && p.delta && p.recorded_lengths && p.recorded_slots));
}
int launch_prepare(Gdn const &p, Stream s) {
#if ALA_PREFILL_TOKEN_PARALLEL && ALA_PPU_TARGET == 100
  if (p.mode == ChunkPrefill) {
    prepare_gdn<true><<<dim3(p.requests, 2 * p.hk + p.hv,
                            prefill_schedule::blocks(p.width)),
                         prefill_schedule::PrepareThreads, 0, s>>>(p);
    return last_error();
  }
#endif
  prepare_gdn<false><<<dim3(p.requests, 2 * p.hk + p.hv), 32, 0, s>>>(p);
  return last_error();
}
int prepare(Gdn const &p, Stream s) {
  validate_gdn<<<(p.requests + 127) / 128, 128, 0, s>>>(p, {});
  if (auto rc = last_error())
    return rc;
  return launch_prepare(p, s);
}
int prepare_checked(Gdn const &p, Stream s,
                    ala_gdn_transaction_v2 const &guard) {
  validate_gdn<<<(p.requests + 127) / 128, 128, 0, s>>>(p, guard);
  if (int rc = last_error())
    return rc;
  return launch_prepare(p, s);
}
} // namespace actlize_la::serving

extern "C" int ala_serving_version() { return actlize_la::serving::AbiVersion; }
extern "C" int ala_serving_target() {
#ifdef ALA_PPU_TARGET
  return ALA_PPU_TARGET;
#else
  return 1200;
#endif
}
extern "C" int ala_serving_struct_size(int which) {
  using namespace actlize_la::serving;
  return which == 0   ? int(sizeof(Gdn))
         : which == 1 ? int(sizeof(Commit))
         : which == 2 ? int(sizeof(Prefill))
         : which == 3 ? int(sizeof(ConvHistory))
                      : -1;
}
extern "C" int ala_gdn_run(actlize_la::serving::Gdn const *p, void *stream) {
  using namespace actlize_la::serving;
  if (!p || !valid(*p))
    return -1;
  auto s = static_cast<Stream>(stream);
  if (auto rc = prepare(*p, s))
    return rc;
  recurrent_gdn<<<dim3(p->requests, p->hv, 32), 128, 0, s>>>(*p);
  return last_error();
}
extern "C" int ala_gdn_commit(actlize_la::serving::Commit const *p,
                              void *stream) {
  using namespace actlize_la::serving;
  if (!p || !p->state || !p->delta || !p->keys || !p->decays || !p->slots ||
      !p->recorded_slots || !p->counts || !p->live || !p->recorded_lengths ||
      !p->errors || p->requests <= 0 || p->state_slots < 2 || p->hk <= 0 ||
      p->hv <= 0 || p->hv > 65535 || p->hv % p->hk || p->width < 1 ||
      p->width > 8 || p->state_stride < int64_t(p->hv) * 16384 ||
      p->slot_stride < 1)
    return -1;
  auto s = static_cast<Stream>(stream);
  validate_commit<<<(p->requests + 127) / 128, 128, 0, s>>>(*p, {});
  if (auto rc = last_error())
    return rc;
  commit_gdn<<<dim3(p->requests, p->hv, 64), 256, 0, s>>>(*p);
  return last_error();
}

namespace actlize_la::serving {
bool valid_dependency(ala_gdn_transaction_v2 const *p, int requests) {
  return p && p->header.version == 2 && p->header.struct_bytes == sizeof(*p) &&
         p->requests == requests &&
         aux::shape(p->upstream_errors, ALA_I32, {requests}) &&
         aux::absent(p->epochs) && aux::absent(p->ready_epochs) &&
         aux::absent(p->consumed_epochs);
}
bool valid_transaction(ala_gdn_transaction_v2 const *p, int requests) {
  if (!p || p->header.version != 2 || p->header.struct_bytes != sizeof(*p) ||
      p->requests != requests)
    return false;
  for (auto t : {p->epochs, p->ready_epochs, p->consumed_epochs})
    if (!aux::shape(t, ALA_I64, {requests}))
      return false;
  return (aux::absent(p->upstream_errors) ||
          aux::shape(p->upstream_errors, ALA_I32, {requests})) &&
         aux::disjoint({p->epochs, p->upstream_errors},
                       {p->ready_epochs, p->consumed_epochs});
}
__global__ void stamp_epoch(ala_gdn_transaction_v2 p, int const *live,
                            int const *errors, int const *lengths,
                            bool consumed, int const *starts = nullptr,
                            int64_t const *slots = nullptr,
                            int64_t slot_stride = 1) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < p.requests && r < *live && !errors[r] && lengths[r] > 0 &&
      (!starts ||
       (starts[r + 1] > starts[r] && slots[int64_t(r) * slot_stride] > 0)))
    aux::at<int64_t>(consumed ? p.consumed_epochs : p.ready_epochs, r) =
        aux::at<int64_t>(p.epochs, r);
}
} // namespace actlize_la::serving
extern "C" int ala_gdn_transaction_v2_size() {
  return sizeof(ala_gdn_transaction_v2);
}
extern "C" int ala_gdn_checked_v2(actlize_la::serving::Gdn const *p,
                                  ala_gdn_transaction_v2 const *guard,
                                  void *stream) {
  using namespace actlize_la::serving;
  if (!p || !valid(*p) || p->mode == Verify ||
      !valid_dependency(guard, p->requests))
    return -1;
  auto s = static_cast<Stream>(stream);
  if (int rc = prepare_checked(*p, s, *guard))
    return rc;
  recurrent_gdn<<<dim3(p->requests, p->hv, 32), 128, 0, s>>>(*p);
  return last_error();
}
extern "C" int ala_gdn_verify_v2(actlize_la::serving::Gdn const *p,
                                 ala_gdn_transaction_v2 const *guard,
                                 void *stream) {
  using namespace actlize_la::serving;
  if (!p || !valid(*p) || p->mode != Verify ||
      !valid_transaction(guard, p->requests))
    return -1;
  auto s = static_cast<Stream>(stream);
  validate_gdn<<<(p->requests + 127) / 128, 128, 0, s>>>(*p, *guard);
  if (int e = last_error())
    return e;
  prepare_gdn<<<dim3(p->requests, 2 * p->hk + p->hv), 32, 0, s>>>(*p);
  if (int e = last_error())
    return e;
  recurrent_gdn<<<dim3(p->requests, p->hv, 32), 128, 0, s>>>(*p);
  if (int e = last_error())
    return e;
  stamp_epoch<<<(p->requests + 127) / 128, 128, 0, s>>>(
      *guard, p->live, p->errors, p->recorded_lengths, false, p->starts,
      p->initial_slots, p->initial_stride);
  return last_error();
}
extern "C" int ala_gdn_commit_v2(actlize_la::serving::Commit const *p,
                                 ala_gdn_transaction_v2 const *guard,
                                 void *stream) {
  using namespace actlize_la::serving;
  if (!p || !p->state || !p->delta || !p->keys || !p->decays || !p->slots ||
      !p->recorded_slots || !p->counts || !p->live || !p->recorded_lengths ||
      !p->errors || p->requests <= 0 || p->state_slots < 2 || p->hk <= 0 ||
      p->hv <= 0 || p->hv > 65535 || p->hv % p->hk || p->width < 1 ||
      p->width > 8 || p->state_stride < int64_t(p->hv) * 16384 ||
      p->slot_stride < 1 || !valid_transaction(guard, p->requests))
    return -1;
  auto s = static_cast<Stream>(stream);
  validate_commit<<<(p->requests + 127) / 128, 128, 0, s>>>(*p, *guard);
  if (int e = last_error())
    return e;
  commit_gdn<<<dim3(p->requests, p->hv, 64), 256, 0, s>>>(*p);
  if (int e = last_error())
    return e;
  stamp_epoch<<<(p->requests + 127) / 128, 128, 0, s>>>(
      *guard, p->live, p->errors, p->recorded_lengths, true);
  return last_error();
}
