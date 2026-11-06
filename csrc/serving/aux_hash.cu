// Integer n-gram semantics follow b12x PLE hash (Apache-2.0; see NOTICE).
#include "aux_common.cuh"
namespace actlize_la::serving::aux {
__global__ void hash_ids(ala_ple_hash_v1 p) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= p.rows.tokens)
    return;
  Row r = row(p.rows, t);
  int64_t tokens[3] = {p.eos, p.eos, p.eos};
  if (r.live) {
    int begin = at<int32_t>(p.rows.starts, r.request);
    for (int lag = 0; lag < 3; ++lag) {
      int pos = t - begin - lag;
      tokens[lag] = pos >= 0 ? at<int64_t>(p.tokens, t - lag)
                             : at<int64_t>(p.history, r.request, 2 + pos);
      if (tokens[lag] < 0 || tokens[lag] >= p.vocab_size)
        r.error = Bounds;
    }
    // Current EOS still uses its preceding window; prior EOS cuts older lag.
    if (tokens[1] == p.eos)
      tokens[2] = p.eos;
  }
  uint64_t products[3] = {};
  for (int lag = 0; lag < 3; ++lag) {
    int64_t m = at<int64_t>(p.multipliers, lag);
    if (m <= 0 || !(m & 1) || m > INT64_MAX / p.vocab_size)
      r.error = Bounds;
    products[lag] = uint64_t(tokens[lag]) * uint64_t(m);
  }
  for (int h = 0; h < 16; ++h) {
    int64_t prime = at<int64_t>(p.primes, h), off = at<int64_t>(p.offsets, h);
    if (prime <= 1 || off < 0 || off > p.padded_vocab_size ||
        prime > p.padded_vocab_size - off)
      r.error = Bounds;
  }
  at<int32_t>(p.errors, t) = r.error;
  at<int64_t>(p.request_ids, t) = r.live && !r.error ? r.request : -1;
  for (int h = 0; h < 16; ++h) {
    int64_t out = -1;
    if (r.live && !r.error) {
      uint64_t u = products[0] ^ products[1] ^ (h < 8 ? 0 : products[2]);
      // Reinterpret two's-complement bits without signed multiply overflow.
      int64_t mixed = u <= uint64_t(INT64_MAX) ? int64_t(u) : -1 - int64_t(~u);
      int64_t prime = at<int64_t>(p.primes, h), rem = mixed % prime;
      if (rem < 0)
        rem += prime;
      out = at<int64_t>(p.offsets, h) + rem;
    }
    at<int64_t>(p.hashes, t, h) = out;
  }
}
bool valid_hash(ala_ple_hash_v1 const *p) {
  if (!p || p->header.version != 1 || p->header.struct_bytes != sizeof(*p) ||
      !valid_rows(p->rows) || p->rows.requests < 1 || p->vocab_size < 1 ||
      p->eos < 0 || p->eos >= p->vocab_size || p->padded_vocab_size < 1)
    return false;
  int T = p->rows.tokens, R = p->rows.requests;
  if (!shape(p->tokens, ALA_I64, {T}) || !shape(p->history, ALA_I64, {R, 2}) ||
      !shape(p->multipliers, ALA_I64, {3}) ||
      !shape(p->primes, ALA_I64, {16}) || !shape(p->offsets, ALA_I64, {16}) ||
      !shape(p->hashes, ALA_I64, {T, 16}) ||
      !shape(p->request_ids, ALA_I64, {T}) || !shape(p->errors, ALA_I32, {T}) ||
      !disjoint({p->tokens, p->history, p->multipliers, p->primes, p->offsets,
                 p->rows.starts, p->rows.slots, p->rows.num_seqs,
                 p->rows.num_tokens, p->rows.upstream_errors},
                {p->hashes, p->request_ids, p->errors}))
    return false;
  return true;
}
} // namespace actlize_la::serving::aux
extern "C" int ala_ple_hash_v1_run(ala_ple_hash_v1 const *p, void *stream) {
  using namespace actlize_la::serving;
  if (!aux::valid_hash(p))
    return -1;
  aux::hash_ids<<<(p->rows.tokens + 127) / 128, 128, 0,
                  static_cast<Stream>(stream)>>>(*p);
  return last_error();
}
