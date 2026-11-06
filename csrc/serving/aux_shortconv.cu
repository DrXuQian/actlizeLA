// Packed depthwise convolution/history contract: Flare/b12x, Apache-2.0.
#include "aux_common.cuh"
namespace actlize_la::serving::aux {
__device__ int conv_error(ala_shortconv_v1 const &p, int r) {
  int nr = at<int32_t>(p.rows.num_seqs, 0),
      nt = at<int32_t>(p.rows.num_tokens, 0);
  if (nr < 0 || nr > p.rows.requests || nt < 0 || nt > p.rows.tokens)
    return Bounds;
  if (r >= nr)
    return Ok;
  int first = at<int32_t>(p.rows.starts, r),
      end = at<int32_t>(p.rows.starts, r + 1);
  if (first < 0 || end < first || end > nt)
    return Bounds;
  if (first == end)
    return Ok;
  // Locally valid intervals can still overlap around a different malformed
  // start pair. Reject both output owners, not just the reversed pair.
  for (int j = 0; j < nr; ++j) {
    int a = at<int32_t>(p.rows.starts, j),
        b = at<int32_t>(p.rows.starts, j + 1);
    if (j != r && a >= 0 && b > a && b <= nt && first < b && a < end)
      return Conflict;
  }
  int e =
      p.rows.upstream_errors.data ? at<int32_t>(p.rows.upstream_errors, r) : 0;
  if (e)
    return e > 0 && e <= Unsupported ? e : Bounds;
  int64_t slot = at<int64_t>(p.rows.slots, r);
  if (slot == -1)
    return Ok;
  if (slot < -1 || slot >= p.state.shape[0])
    return Slot;
  if (p.kind != ALA_CONV_SNAPSHOT)
    for (int j = 0; j < nr; ++j) {
      int a = at<int32_t>(p.rows.starts, j),
          b = at<int32_t>(p.rows.starts, j + 1);
      if (j != r && a >= 0 && b > a && b <= nt &&
          slot == at<int64_t>(p.rows.slots, j))
        return Conflict;
    }
  if (p.kind == ALA_CONV_VERIFY) {
    if (end - first > p.drafts + 1)
      return Bounds;
    int previous = at<int32_t>(p.previous_accepted, r);
    if (!at<uint8_t>(p.fresh, r) && (previous < 1 || previous > p.drafts + 1))
      return Prefix;
  }
  if (p.kind == ALA_CONV_RESTORE_ZERO) {
    int n = at<int32_t>(p.consumed_prefix, r);
    if (n < 0 || n > end - first || slot != at<int64_t>(p.stamps, r))
      return Prefix;
  }
  return Ok;
}
__global__ void validate_conv(ala_shortconv_v1 p) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < p.rows.requests) {
    at<int32_t>(p.errors, i) = conv_error(p, int(i));
    if (p.kind == ALA_CONV_SNAPSHOT)
      at<int64_t>(p.stamps, i) = -1;
  }
  if (p.output.data && i < int64_t(p.rows.tokens) * p.channels)
    put(p.output, 0, i / p.channels, i % p.channels);
}
__global__ void shortconv(ala_shortconv_v1 p) {
  int r = blockIdx.x, c = blockIdx.y * 128 + threadIdx.x;
  if (c >= p.channels || r >= at<int32_t>(p.rows.num_seqs, 0) ||
      at<int32_t>(p.errors, r))
    return;
  int first = at<int32_t>(p.rows.starts, r),
      end = at<int32_t>(p.rows.starts, r + 1), n = end - first;
  int64_t slot = at<int64_t>(p.rows.slots, r);
  if (!n || slot < 0)
    return;
  int H = 3 * p.dilation, L = H + p.drafts;
  if (p.kind == ALA_CONV_SNAPSHOT || p.kind == ALA_CONV_RESTORE_ZERO) {
    if (p.kind == ALA_CONV_RESTORE_ZERO &&
        at<int32_t>(p.consumed_prefix, r) != 0)
      return;
    for (int j = 0; j < L; ++j) {
      if (p.kind == ALA_CONV_SNAPSHOT)
        at<BFloat>(p.snapshot, r, c, j) = at<BFloat>(p.state, slot, c, j);
      else
        at<BFloat>(p.state, slot, c, j) = at<BFloat>(p.snapshot, r, c, j);
    }
    if (!c && p.kind == ALA_CONV_SNAPSHOT)
      at<int64_t>(p.stamps, r) = slot;
    return;
  }
  float history[9], weight[4];
  bool fresh = at<uint8_t>(p.fresh, r);
  int shift = p.kind == ALA_CONV_VERIFY && !fresh
                  ? at<int32_t>(p.previous_accepted, r) - 1
                  : 0;
  for (int j = 0; j < H; ++j)
    history[j] = fresh ? 0 : get(p.state, slot, c, j + shift);
  for (int j = 0; j < 4; ++j)
    weight[j] = get(p.weight, c, j);
  for (int t = 0; t < n; ++t) {
    float acc = 0;
    for (int tap = 0; tap < 4; ++tap) {
      int pos = t - (3 - tap) * p.dilation;
      float x = pos >= 0 ? get(p.x, first + pos, c) : history[H + pos];
      // Explicit fp32 product/add; every BF16 input product is exact fp32.
      acc = __fadd_rn(acc, __fmul_rn(x, weight[tap]));
    }
    put(p.output, silu(round_bf(acc)), first + t, c);
  }
  int committed = p.kind == ALA_CONV_PREFILL ? n : 1;
  for (int j = 0; j < H; ++j) {
    int pos = j + committed;
    put(p.state, pos < H ? history[pos] : get(p.x, first + pos - H, c), slot, c,
        j);
  }
  for (int j = 0; j < p.drafts; ++j)
    put(p.state,
        p.kind == ALA_CONV_VERIFY && j + 1 < n ? get(p.x, first + j + 1, c) : 0,
        slot, c, H + j);
}
bool valid_conv(ala_shortconv_v1 const &p) {
  int R = p.rows.requests, T = p.rows.tokens, C = p.channels,
      H = 3 * p.dilation, L = H + p.drafts;
  if (p.header.version != 1 || p.header.struct_bytes != sizeof(p) ||
      !valid_rows(p.rows) || R < 1 || p.kind < 0 || p.kind > 3 || C < 1 ||
      C > 65535 * 128 || (p.dilation != 1 && p.dilation != 3) || p.drafts < 0 ||
      p.drafts > 7 || !shape(p.rows.slots, ALA_I64, {R}) ||
      !shape(p.state, ALA_BF16, {p.state.shape[0], C, L}) ||
      !shape(p.errors, ALA_I32, {R}))
    return false;
  bool transport = p.kind >= ALA_CONV_SNAPSHOT;
  if (transport) {
    if (!shape(p.snapshot, ALA_BF16, {R, C, L}) ||
        !shape(p.stamps, ALA_I64, {R}) ||
        !shape(p.consumed_prefix, ALA_I32, {R}) || !absent(p.output) ||
        !absent(p.x) || !absent(p.weight) || !absent(p.fresh) ||
        !absent(p.previous_accepted))
      return false;
  } else {
    if (!shape(p.x, ALA_BF16, {T, C}) || !shape(p.weight, ALA_BF16, {C, 4}) ||
        !shape(p.output, ALA_BF16, {T, C}) || !shape(p.fresh, ALA_U8, {R}) ||
        !shape(p.previous_accepted, ALA_I32, {R}) || !absent(p.snapshot) ||
        !absent(p.stamps) || !absent(p.consumed_prefix))
      return false;
  }
  // State is a single read/write operand; snapshot/copy never aliases it.
  if (p.kind == ALA_CONV_SNAPSHOT)
    return disjoint({p.state, p.rows.starts, p.rows.slots, p.rows.num_seqs,
                     p.rows.num_tokens, p.rows.upstream_errors,
                     p.consumed_prefix},
                    {p.snapshot, p.stamps, p.errors});
  if (p.kind == ALA_CONV_RESTORE_ZERO)
    return disjoint({p.snapshot, p.stamps, p.consumed_prefix, p.rows.starts,
                     p.rows.slots, p.rows.num_seqs, p.rows.num_tokens,
                     p.rows.upstream_errors},
                    {p.state, p.errors});
  return disjoint({p.x, p.weight, p.fresh, p.previous_accepted, p.rows.starts,
                   p.rows.slots, p.rows.num_seqs, p.rows.num_tokens,
                   p.rows.upstream_errors},
                  {p.state, p.output, p.errors});
}
} // namespace actlize_la::serving::aux
extern "C" int ala_shortconv_v1_run(ala_shortconv_v1 const *p, void *stream) {
  using namespace actlize_la::serving;
  if (!p || !aux::valid_conv(*p))
    return -1;
  auto s = static_cast<Stream>(stream);
  int64_t work = p->output.data ? int64_t(p->rows.tokens) * p->channels : 0;
  if (work < p->rows.requests)
    work = p->rows.requests;
  if ((work + 255) / 256 > INT_MAX)
    return -1;
  aux::validate_conv<<<unsigned((work + 255) / 256), 256, 0, s>>>(*p);
  if (int e = last_error())
    return e;
  aux::shortconv<<<dim3(p->rows.requests, (p->channels + 127) / 128), 128, 0,
                   s>>>(*p);
  return last_error();
}
