#pragma once
#include "abi.h"
#include "aux_v1.h"
#include "runtime.cuh"
#include <climits>
#include <initializer_list>

namespace actlize_la::serving::aux {
using Tensor = ala_tensor_v1;
inline int item_size(int t) {
  return t == ALA_BF16                    ? 2
         : t == ALA_I64                   ? 8
         : t == ALA_U8                    ? 1
         : (t == ALA_F32 || t == ALA_I32) ? 4
                                          : 0;
}
inline bool absent(Tensor const &t) { return !t.data && !t.nbytes && !t.rank; }
inline bool valid(Tensor const &t) {
  int const item = item_size(t.dtype);
  if (!t.data || !item || t.rank < 1 || t.rank > 3 || uintptr_t(t.data) % item)
    return false;
  int order[3] = {0, 1, 2};
  for (int i = 0; i < t.rank; ++i)
    for (int j = i + 1; j < t.rank; ++j)
      if (t.stride[order[i]] > t.stride[order[j]]) {
        int z = order[i];
        order[i] = order[j];
        order[j] = z;
      }
  uint64_t span = 1;
  for (int i = 0; i < t.rank; ++i) {
    int d = order[i];
    if (t.shape[d] <= 0 || t.stride[d] < 1 ||
        (t.shape[d] > 1 && uint64_t(t.stride[d]) < span))
      return false;
    uint64_t n = t.shape[d] - 1, s = t.stride[d];
    if (n && s > (UINT64_MAX - span) / n)
      return false;
    span += n * s;
  }
  return span <= t.nbytes / item && span <= INT64_MAX / item &&
         uintptr_t(t.data) <= UINTPTR_MAX - span * item;
}
inline bool shape(Tensor const &t, int dtype,
                  std::initializer_list<int64_t> dims) {
  if (!valid(t) || t.dtype != dtype || t.rank != int(dims.size()))
    return false;
  int d = 0;
  for (auto n : dims)
    if (t.shape[d++] != n)
      return false;
  return true;
}
// Exact disjointness for regular inner rows, including separated column views
// in a shared allocation. No rejection merely because storage owners match.
struct Intervals {
  Tensor t;
  int order[3] = {0, 1, 2}, outer = 0;
  uint64_t width = 0, count = 1;
  explicit Intervals(Tensor v) : t(v) {
    for (int i = 0; i < t.rank; ++i)
      for (int j = i + 1; j < t.rank; ++j)
        if (t.stride[order[i]] > t.stride[order[j]]) {
          int z = order[i];
          order[i] = order[j];
          order[j] = z;
        }
    uint64_t elems = 1;
    while (outer < t.rank && uint64_t(t.stride[order[outer]]) == elems)
      elems *= t.shape[order[outer++]];
    width = elems * item_size(t.dtype);
    for (int i = outer; i < t.rank; ++i)
      count *= t.shape[order[i]];
  }
  uintptr_t start(uint64_t i) const {
    uint64_t off = 0;
    for (int j = outer; j < t.rank; ++j) {
      int d = order[j];
      off += (i % t.shape[d]) * t.stride[d];
      i /= t.shape[d];
    }
    return uintptr_t(t.data) + off * item_size(t.dtype);
  }
};
inline bool overlap(Tensor const &a, Tensor const &b) {
  if (absent(a) || absent(b))
    return false;
  Intervals aa(a), bb(b);
  if (aa.start(0) >= bb.start(bb.count - 1) + bb.width ||
      bb.start(0) >= aa.start(aa.count - 1) + aa.width)
    return false;
  uint64_t ai = 0, bi = 0;
  while (ai < aa.count && bi < bb.count) {
    uintptr_t al = aa.start(ai), bl = bb.start(bi);
    if (al < bl + bb.width && bl < al + aa.width)
      return true;
    if (al + aa.width <= bl)
      ++ai;
    else
      ++bi;
  }
  return false;
}
inline bool disjoint(std::initializer_list<Tensor> reads,
                     std::initializer_list<Tensor> writes) {
  for (auto const &w : writes) {
    for (auto const &r : reads)
      if (overlap(r, w))
        return false;
  }
  for (auto a = writes.begin(); a != writes.end(); ++a)
    for (auto b = a + 1; b != writes.end(); ++b)
      if (overlap(*a, *b))
        return false;
  return true;
}
inline bool valid_rows(ala_rows_v1 const &p) {
  if (p.tokens < 1 || p.tokens > INT_MAX - 255 || p.requests < 0 ||
      p.requests > INT_MAX - 255 ||
      !(absent(p.num_tokens) || shape(p.num_tokens, ALA_I32, {1})))
    return false;
  if (!p.requests)
    return absent(p.starts) && absent(p.slots) && absent(p.num_seqs) &&
           absent(p.upstream_errors);
  return shape(p.starts, ALA_I32, {int64_t(p.requests) + 1}) &&
         shape(p.num_seqs, ALA_I32, {1}) && shape(p.num_tokens, ALA_I32, {1}) &&
         (absent(p.slots) || shape(p.slots, ALA_I64, {p.requests})) &&
         (absent(p.upstream_errors) ||
          shape(p.upstream_errors, ALA_I32, {p.requests}));
}
template <class T>
__device__ T &at(Tensor const &v, int64_t i, int64_t j = 0, int64_t k = 0) {
  return static_cast<T *>(
      v.data)[i * v.stride[0] + j * v.stride[1] + k * v.stride[2]];
}
__device__ inline float get(Tensor const &v, int64_t i, int64_t j = 0,
                            int64_t k = 0) {
  return v.dtype == ALA_F32 ? at<float>(v, i, j, k)
                            : bf(at<BFloat>(v, i, j, k));
}
__device__ inline void put(Tensor const &v, float x, int64_t i, int64_t j = 0,
                           int64_t k = 0) {
  at<BFloat>(v, i, j, k) = bf(x);
}
struct Row {
  int request;
  int error;
  bool live;
};
__device__ inline Row row(ala_rows_v1 const &p, int token) {
  int nt = p.num_tokens.data ? at<int32_t>(p.num_tokens, 0) : p.tokens;
  if (nt < 0 || nt > p.tokens)
    return {-1, Bounds, false};
  if (!p.requests)
    return {-1, Ok, token < nt};
  int nr = at<int32_t>(p.num_seqs, 0);
  if (nr < 0 || nr > p.requests)
    return {-1, Bounds, false};
  int found = -1;
  for (int r = 0; r < nr; ++r) {
    int a = at<int32_t>(p.starts, r), b = at<int32_t>(p.starts, r + 1);
    if (a < 0 || b < a || b > nt)
      return {-1, Bounds, false};
    if (token >= a && token < b)
      found = r;
  }
  if (token >= nt || found < 0)
    return {-1, Ok, false};
  int e = p.upstream_errors.data ? at<int32_t>(p.upstream_errors, found) : 0;
  if (e < 0 || e > Unsupported)
    e = Bounds;
  if (!e && p.slots.data && at<int64_t>(p.slots, found) < -1)
    e = Slot;
  bool live = !e && (!p.slots.data || at<int64_t>(p.slots, found) >= 0);
  return {found, e, live};
}
__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float silu(float x) { return x * sigmoid(x); }
__device__ inline float sum256(float x, float *scratch) {
  x = warp_sum(x);
  if (!(threadIdx.x % 32))
    scratch[threadIdx.x / 32] = x;
  __syncthreads();
  float y = threadIdx.x < 8 ? scratch[threadIdx.x] : 0;
  y = warp_sum(y);
  if (!threadIdx.x)
    scratch[0] = y;
  __syncthreads();
  return scratch[0];
}
} // namespace actlize_la::serving::aux
