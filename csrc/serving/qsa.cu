// Streaming/paged extension of qsa_indexer_kernel.cu and qsa_core.cu.
// Compression rounding and speculative ring transaction follow Flare/b12x:
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0
#include "abi.h"
#include "qsa.h"
#include "runtime.cuh"

namespace actlize_la::serving {
constexpr int Ratio = 4;
__device__ int physical(Qsa const &p, int r, int pos, bool compressed = false) {
  int const size = compressed ? p.page_size / 4 : p.page_size;
  int const page = pos / size;
  if (page < 0 || page >= p.table_width)
    return -1;
  int x = (compressed ? p.compressed_table
                      : p.main_table)[int64_t(r) * p.table_width + page];
  return x >= 0 && x < p.pages ? x : -1;
}
__device__ bool active(Qsa const &p, int row) {
  auto r = p.request_ids[row];
  return r >= 0 && r < p.requests && r < *p.live && !p.errors[r] &&
         row >= p.starts[r] && row < p.starts[r + 1] &&
         p.raw_slots[r * p.raw_slot_stride] >= 0 && p.positions[row] >= 0 &&
         p.positions[row] < p.lengths[r];
}
// Empty/inactive packed rows may retain old slot/page metadata on graph replay.
// They have no memory ownership this round and must not create false conflicts.
__device__ bool nonempty_request(Qsa const &p, int r) {
  int const start = p.starts[r], end = p.starts[r + 1];
  auto const slot = p.raw_slots[int64_t(r) * p.raw_slot_stride];
  return start >= 0 && end > start && end <= p.tokens && slot >= 0 &&
         slot < p.raw_slot_count && p.request_ids[start] == r;
}
__global__ void validate_qsa(Qsa p) {
  int const r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= p.requests)
    return;
  int error = Ok, live = *p.live;
  if (live < 0 || live > p.requests)
    error = Bounds;
  else if (r < live) {
    int const start = p.starts[r], end = p.starts[r + 1];
    auto slot = p.raw_slots[int64_t(r) * p.raw_slot_stride];
    if (start < 0 || end < start || end > p.tokens || p.lengths[r] < 0 ||
        int64_t(p.lengths[r]) > int64_t(p.table_width) * p.page_size)
      error = Bounds;
    else if (slot >= p.raw_slot_count)
      error = Slot;
    else if (slot >= 0 && end > start && p.request_ids[start] >= 0) {
      int64_t first = p.positions[start];
      if (first < 0 || first >= p.lengths[r]) {
        p.errors[r] = Bounds;
        return;
      }
      int64_t prior = p.anchors[slot * p.anchor_stride];
      int accepted = p.accepted_previous[r];
      if (first < 0 || accepted < 0 || (!p.prefilling[r] && accepted > 8) ||
          first != prior + accepted ||
          (prior == -1 && (first != 0 || accepted != 1)))
        error = History;
      for (int j = 0; j < live; ++j)
        if (j != r && nonempty_request(p, j) &&
            slot == p.raw_slots[int64_t(j) * p.raw_slot_stride])
          error = Conflict;
      for (int row = start; row < end; ++row) {
        if (p.request_ids[row] < 0) {
          if (first + row - start < p.lengths[r])
            error = Bounds;
          continue;
        }
        auto pos = p.positions[row];
        if (p.request_ids[row] != r || pos != first + row - start || pos < 0 ||
            pos >= p.lengths[r])
          error = Bounds;
        int ph = physical(p, r, int(pos));
        if (ph < 0 || p.slot_mapping[row] !=
                          int64_t(ph) * p.page_size + pos % p.page_size)
          error = Page;
        for (int axis = 0; axis < 3; ++axis)
          if (p.rope_positions[int64_t(row) * 3 + axis] < 0 ||
              p.rope_positions[int64_t(row) * 3 + axis] >= p.rope_rows)
            error = Bounds;
        if ((pos + 1) % Ratio == 0 &&
            physical(p, r, int(pos / Ratio), true) < 0)
          error = Page;
      }
      // All previous complete groups are visible history; page ownership is
      // established by the caller, but invalid/missing physical pages fail.
      for (int page = 0; page < (p.lengths[r] + p.page_size - 1) / p.page_size;
           ++page) {
        int mp = p.main_table[int64_t(r) * p.table_width + page];
        int cp = p.compressed_table[int64_t(r) * p.table_width + page];
        if (mp < 0 || mp >= p.pages || cp < 0 || cp >= p.pages)
          error = Page;
      }
      for (int64_t pos = first / Ratio * Ratio; pos < first; ++pos) {
        int ri = int(pos % p.ring_size);
        if (p.raw_tags[slot * p.tag_slot_stride + ri] != pos)
          error = History;
        for (int a = 0; a < 3; ++a) {
          auto rp = p.raw_rope[slot * p.rope_slot_stride + ri * 3 + a];
          if (rp < 0 || rp >= p.rope_rows)
            error = History;
        }
      }
      // Writable boundary pages must be exclusive. Sharing older immutable
      // pages is fine; appending to a page another request can read needs COW.
      if (!error)
        for (int lp = int(first / p.page_size);
             lp < (p.lengths[r] + p.page_size - 1) / p.page_size; ++lp) {
          int mp = p.main_table[int64_t(r) * p.table_width + lp],
              cp = p.compressed_table[int64_t(r) * p.table_width + lp];
          for (int l = 0; l < (p.lengths[r] + p.page_size - 1) / p.page_size;
               ++l)
            if (l != lp)
              if (mp == p.main_table[int64_t(r) * p.table_width + l] ||
                  cp == p.compressed_table[int64_t(r) * p.table_width + l])
                error = Conflict;
          for (int j = 0; j < live; ++j)
            if (j != r && nonempty_request(p, j)) {
              int pages =
                  (max(0, min(p.lengths[j], p.table_width * p.page_size)) +
                   p.page_size - 1) /
                  p.page_size;
              for (int l = 0; l < pages; ++l)
                if (mp == p.main_table[int64_t(j) * p.table_width + l] ||
                    cp == p.compressed_table[int64_t(j) * p.table_width + l])
                  error = Conflict;
            }
        }
    }
  }
  p.errors[r] = error;
}

// Same Gemma RMSNorm + BF16 boundary and partial mRoPE as b12x. This is
// ONLY the indexer's required encoder, not a replacement for model RoPE/norm.
__device__ void encode_index(float (&x)[4], float const *weight,
                             int64_t const (&position)[3], Qsa const &p,
                             int lane) {
  float square = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    square += x[j] * x[j];
  float const norm = rsqrtf(warp_sum(square) / 128.0f + p.norm_epsilon);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    x[j] = round_bf(x[j] * norm * (1.0f + weight[lane * 4 + j]));
  float partner[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    partner[j] = __shfl_xor_sync(0xffffffffU, x[j], 8);
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int const pair = (lane * 4 + j) % 32;
      int const axis = (pair % 3 == 1 && pair < 33)   ? 1
                       : (pair % 3 == 2 && pair < 30) ? 2
                                                      : 0;
      auto const at = position[axis] * 32 + pair;
      float co = p.rope_bf16 ? read_bf(p.cosine, at)
                             : static_cast<float const *>(p.cosine)[at];
      float si = p.rope_bf16 ? read_bf(p.sine, at)
                             : static_cast<float const *>(p.sine)[at];
      float left = x[j] * co;
      float right = (lane < 8 ? -partner[j] : partner[j]) * si;
      x[j] = p.rope_bf16 ? round_bf(round_bf(left) + round_bf(right))
                         : left + right;
    }
  }
}
__global__ void encode_queries(Qsa p) {
  int const row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
  bool live = active(p, row);
  float x[4];
  int64_t positions[3] = {0, 0, 0};
  if (live)
    for (int j = 0; j < 3; ++j)
      positions[j] = p.rope_positions[int64_t(row) * 3 + j];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    x[j] = live ? read_bf(p.index_query, int64_t(row) * p.index_stride +
                                             h * 128 + lane * 4 + j)
                : 0;
  encode_index(x, p.query_weight, positions, p, lane);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    p.encoded_query[(int64_t(row) * 4 + h) * 128 + lane * 4 + j] =
        live ? x[j] : 0;
}
__global__ void append_kv(Qsa p) {
  int const row = blockIdx.x, h = blockIdx.y;
  if (!active(p, row))
    return;
  auto slot = p.slot_mapping[row];
  for (int d = threadIdx.x; d < 256; d += blockDim.x) {
    int64_t const dst = (slot * p.kv_heads + h) * 256 + d;
    static_cast<BFloat *>(p.key_cache)[dst] = static_cast<BFloat const *>(
        p.new_key)[int64_t(row) * p.key_stride + h * 256 + d];
    static_cast<BFloat *>(p.value_cache)[dst] = static_cast<BFloat const *>(
        p.new_value)[int64_t(row) * p.value_stride + h * 256 + d];
  }
}
__global__ void compress_completed(Qsa p) {
  int const row = blockIdx.x, lane = threadIdx.x;
  if (!active(p, row))
    return;
  auto pos = p.positions[row];
  if ((pos + 1) % 4)
    return;
  int const r = int(p.request_ids[row]), start = p.starts[r];
  auto first = p.positions[start],
       slot = p.raw_slots[int64_t(r) * p.raw_slot_stride];
  float x[4] = {};
  int64_t rope[3] = {};
  for (int t = 0; t < 4; ++t) {
    auto src = pos - 3 + t;
    bool current = src >= first;
    auto input = current ? int64_t(start) + src - first : src % p.ring_size;
#pragma unroll
    for (int d = 0; d < 4; ++d)
      x[d] += current ? read_bf(p.raw_key, input * p.raw_stride + lane * 4 + d)
                      : read_bf(p.raw_ring, slot * p.ring_slot_stride +
                                                input * 128 + lane * 4 + d);
    if (!t)
      for (int a = 0; a < 3; ++a)
        rope[a] = current
                      ? p.rope_positions[input * 3 + a]
                      : p.raw_rope[slot * p.rope_slot_stride + input * 3 + a];
  }
#pragma unroll
  for (int d = 0; d < 4; ++d)
    x[d] = round_bf(x[d] * 0.25f);
  encode_index(x, p.key_weight, rope, p, lane);
  int const group = int(pos / 4), page_size = p.page_size / 4,
            ph = physical(p, r, group, true);
#pragma unroll
  for (int d = 0; d < 4; ++d)
    static_cast<BFloat *>(p.compressed_cache)
        [(int64_t(ph) * page_size + group % page_size) * 128 + lane * 4 + d] =
            bf(x[d]);
}
__global__ void retain_ring(Qsa p) {
  int const r = blockIdx.x, lane = threadIdx.x;
  if (r >= *p.live || p.errors[r])
    return;
  int const start = p.starts[r], end = p.starts[r + 1];
  auto slot = p.raw_slots[int64_t(r) * p.raw_slot_stride];
  if (slot < 0 || start == end || p.request_ids[start] < 0)
    return;
  auto first = p.positions[start];
  int const count = min(end - start, int(p.lengths[r] - first));
  for (int t = max(0, count - p.ring_size); t < count; ++t) {
    int const row = start + t, ri = int((first + t) % p.ring_size);
#pragma unroll
    for (int d = 0; d < 4; ++d)
      static_cast<BFloat *>(
          p.raw_ring)[slot * p.ring_slot_stride + ri * 128 + lane * 4 + d] =
          static_cast<BFloat const *>(
              p.raw_key)[int64_t(row) * p.raw_stride + lane * 4 + d];
    if (!lane) {
      p.raw_tags[slot * p.tag_slot_stride + ri] = first + t;
      for (int a = 0; a < 3; ++a)
        p.raw_rope[slot * p.rope_slot_stride + ri * 3 + a] =
            p.rope_positions[int64_t(row) * 3 + a];
    }
  }
  if (!lane)
    p.anchors[slot * p.anchor_stride] =
        p.prefilling[r] ? first + count - 1 : first;
}

// Retain top 512 from a 512-entry carry + the next 512 groups. This is the
// existing exact packed-key bitonic order, applied in bounded merge windows.
// No query x context score allocation (shared storage is constant 8 KiB).
__global__ void select_groups(Qsa p) {
  __shared__ uint64_t order[1024];
  int const row = blockIdx.x, tid = threadIdx.x, lane = tid % 32,
            warp = tid / 32;
  bool const live = active(p, row);
  int const r = live ? int(p.request_ids[row]) : 0;
  int const position = live ? int(p.positions[row]) : -1;
  int const eligible = live ? min((position + 1) / 4, p.lengths[r] / 4) : 0;
  for (int i = tid; i < 1024; i += 256)
    order[i] = 0;
  __syncthreads();
  float q[4][4];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
#pragma unroll
    for (int d = 0; d < 4; ++d)
      q[h][d] = p.encoded_query[(int64_t(row) * 4 + h) * 128 + lane * 4 + d];
  }
  for (int tile = 0; tile < eligible; tile += 512) {
    for (int local = warp; local < 512; local += 8) {
      int const group = tile + local;
      float score = 0;
      if (group < eligible) {
        int const ph = physical(p, r, group, true), ps = p.page_size / 4;
        float key[4];
#pragma unroll
        for (int d = 0; d < 4; ++d)
          key[d] =
              read_bf(p.compressed_cache,
                      (int64_t(ph) * ps + group % ps) * 128 + lane * 4 + d);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          float dot = 0;
#pragma unroll
          for (int d = 0; d < 4; ++d)
            dot += q[h][d] * key[d];
          score += fmaxf(warp_sum(dot), 0.0f);
        }
        score *= 0.08838834764831845f;
      }
      if (!lane)
        order[512 + local] =
            group < eligible
                ? (uint64_t(__float_as_uint(score) | 0x80000000U) << 32) |
                      (0xffffffffU - uint32_t(group))
                : 0;
    }
    __syncthreads();
    for (int k = 2; k <= 1024; k *= 2)
      for (int j = k / 2; j; j /= 2) {
        for (int i = tid; i < 1024; i += 256) {
          int peer = i ^ j;
          if (peer > i) {
            auto a = order[i], b = order[peer];
            if (((i & k) == 0 && a < b) || ((i & k) != 0 && b < a)) {
              order[i] = b;
              order[peer] = a;
            }
          }
        }
        __syncthreads();
      }
  }
  int const count = min(eligible, 512), tail = (position + 1) % 4,
            total = count * 4 + tail;
  for (int i = tid; i < 512; i += 256) {
    p.selected_groups[int64_t(row) * 512 + i] =
        i < count ? int(0xffffffffU - uint32_t(order[i])) : -1;
    p.selected_scores[int64_t(row) * 512 + i] =
        i < count ? __uint_as_float(uint32_t(order[i] >> 32) & 0x7fffffffU)
                  : -INFINITY;
  }
  for (int i = tid; i < 2051; i += 256) {
    int value = -1;
    if (i < count * 4)
      value = int(0xffffffffU - uint32_t(order[i / 4])) * 4 + i % 4;
    else if (i < total)
      value = (position + 1) / 4 * 4 + i - count * 4;
    p.selected[int64_t(row) * 2051 + i] = value;
  }
  if (!tid)
    p.selected_count[row] = total;
}

// Preserve qsa_core's online-softmax math; only token -> physical page
// address generation changes. The TC path is not silently claimed here.
__global__ void paged_attention(Qsa p) {
  int const row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
  float q[8], out[8] = {}, maximum = -INFINITY, sum = 0;
  bool const live = active(p, row);
#pragma unroll
  for (int d = 0; d < 8; ++d)
    q[d] = live ? read_bf(p.query, int64_t(row) * p.query_stride + h * 256 +
                                       lane * 8 + d)
                : 0;
  if (live) {
    int const r = int(p.request_ids[row]),
              kh = h / (p.query_heads / p.kv_heads);
    for (int i = 0; i < p.selected_count[row]; ++i) {
      int const pos = p.selected[int64_t(row) * 2051 + i];
      if (pos < 0 || pos > p.positions[row] || pos >= p.lengths[r])
        continue;
      int const ph = physical(p, r, pos);
      auto const offset =
          ((int64_t(ph) * p.page_size + pos % p.page_size) * p.kv_heads + kh) *
              256 +
          lane * 8;
      float dot = 0;
#pragma unroll
      for (int d = 0; d < 8; ++d)
        dot += q[d] * read_bf(p.key_cache, offset + d);
      float score = warp_sum(dot) * 0.0625f, next = fmaxf(score, maximum);
      float prob = expf(score - next), correction = expf(maximum - next);
      sum = sum * correction + prob;
#pragma unroll
      for (int d = 0; d < 8; ++d)
        out[d] =
            out[d] * correction + prob * read_bf(p.value_cache, offset + d);
      maximum = next;
    }
  }
#pragma unroll
  for (int d = 0; d < 8; ++d)
    static_cast<BFloat *>(
        p.output)[(int64_t(row) * p.query_heads + h) * 256 + lane * 8 + d] =
        bf(sum > 0 ? out[d] / sum : 0.0f);
}
} // namespace actlize_la::serving

extern "C" int ala_qsa_struct_size() {
  return sizeof(actlize_la::serving::Qsa);
}
extern "C" int ala_qsa_run(actlize_la::serving::Qsa const *p, void *stream) {
  using namespace actlize_la::serving;
  if (!p || p->requests <= 0 || p->tokens <= 0 || p->pages <= 0 ||
      p->raw_slot_count <= 0 || p->table_width <= 0 ||
      (p->page_size != 64 && p->page_size != 128) || p->ring_size < 12 ||
      p->page_size % p->ring_size || p->rope_rows <= 0 || p->query_heads <= 0 ||
      p->query_heads > 65535 || p->kv_heads <= 0 ||
      p->query_heads % p->kv_heads || !(p->norm_epsilon > 0))
    return -1;
  if (!p->query || !p->new_key || !p->new_value || !p->index_query ||
      !p->raw_key || !p->query_weight || !p->key_weight || !p->cosine ||
      !p->sine || !p->request_ids || !p->positions || !p->rope_positions ||
      !p->raw_slots || !p->slot_mapping || !p->starts || !p->live ||
      !p->lengths || !p->accepted_previous || !p->prefilling ||
      !p->main_table || !p->compressed_table || !p->key_cache ||
      !p->value_cache || !p->compressed_cache || !p->raw_ring || !p->raw_tags ||
      !p->raw_rope || !p->anchors || !p->encoded_query || !p->selected ||
      !p->selected_count || !p->selected_groups || !p->selected_scores ||
      !p->output || !p->errors)
    return -1;
  // Dtype/stride/alias admission is performed at prepare by Python.
  auto s = static_cast<Stream>(stream);
  validate_qsa<<<(p->requests + 127) / 128, 128, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  append_kv<<<dim3(p->tokens, p->kv_heads), 128, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  encode_queries<<<dim3(p->tokens, 4), 32, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  compress_completed<<<p->tokens, 32, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  // Retain AFTER every compression read, so speculative ring writes cannot
  // erase accepted keys needed by another newly completed group.
  retain_ring<<<p->requests, 32, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  select_groups<<<p->tokens, 256, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  paged_attention<<<dim3(p->tokens, p->query_heads), 32, 0, s>>>(*p);
  return last_error();
}
