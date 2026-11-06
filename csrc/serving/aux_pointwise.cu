// Arithmetic boundaries follow Flare and b12x (Apache-2.0; see NOTICE).
#include "aux_common.cuh"
namespace actlize_la::serving::aux {
__device__ float injection(ala_pointwise_v1 const &p, int t, int g, int d) {
  float scale = 2.0f * sigmoid(get(p.gate, t, g) / float(p.groups));
  return round_bf(
      __fadd_rn(get(p.x, t, g, d), __fmul_rn(scale, get(p.value, t, 0, d))));
}
__device__ void trace(ala_pointwise_v1 const &p, float x, int t, int g, int d) {
  if (p.trace.data)
    at<float>(p.trace, t, g, d) = x;
}
__global__ void pointwise(ala_pointwise_v1 p) {
  int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, D = p.width;
  Row r = row(p.rows, t);
  if (!g && !tid)
    at<int32_t>(p.errors, t) = r.error;
  if (!r.live) {
    for (int d = tid; d < D; d += 256) {
      put(p.output, 0, t, g, d);
      if (p.combined.data)
        put(p.combined, 0, t, g, d);
    }
    if (p.trace.data)
      for (int s = 0; s < (p.kind == ALA_HC_GATE_MEAN ? 4 : 1); ++s)
        for (int d = tid; d < p.trace.shape[2]; d += 256)
          trace(p, 0, t, g + s, d);
    return;
  }
  __shared__ float sums[8];
  if (p.kind == ALA_HC_SILU) {
    for (int d = tid; d < D; d += 256) {
      float scaled = round_bf(get(p.x, t, 0, d) / 4.0f);
      trace(p, scaled, t, 0, d);
      put(p.output, silu(scaled), t, 0, d);
    }
  } else if (p.kind == ALA_HC_GATE_MEAN) {
    for (int d = tid; d < D; d += 256) {
      float total = 0;
      for (int s = 0; s < 4; ++s) {
        float gate = round_bf(sigmoid(get(p.gate, t, s, d)));
        float product = round_bf(gate * get(p.x, t, s, d));
        trace(p, gate, t, s, d);
        trace(p, product, t, s, D + d);
        total += product;
      }
      put(p.output, total / 4.0f, t, 0, d);
    }
  } else if (p.kind == ALA_PLE_CONTRIBUTION) {
    for (int d = tid; d < D; d += 256)
      put(p.output, get(p.x, t, g, d) + get(p.value, t, g, d), t, g, d);
  } else if (p.kind == ALA_HC_COMBINE) {
    for (int d = tid; d < D; d += 256)
      put(p.output, injection(p, t, g, d), t, g, d);
  } else if (p.kind == ALA_PLE_GATE) {
    float dot = 0;
    for (int d = tid; d < D; d += 256)
      dot += round_bf(get(p.x, t, g, d) * get(p.gate, t, g, d));
    float stages[6];
    stages[0] = round_bf(sum256(dot, sums));
    stages[1] = round_bf(stages[0] / sqrtf(float(D)));
    stages[2] = round_bf(fmaxf(fabsf(stages[1]), round_bf(1e-6f)));
    stages[3] = round_bf(sqrtf(stages[2]));
    stages[4] = round_bf(stages[3] * (stages[1] > 0   ? 1.0f
                                      : stages[1] < 0 ? -1.0f
                                                      : 0.0f));
    stages[5] = round_bf(sigmoid(stages[4]));
    if (!tid)
      for (int i = 0; i < 6; ++i)
        trace(p, stages[i], t, g, i);
    for (int d = tid; d < D; d += 256)
      put(p.output, stages[5] * get(p.value, t, 0, d), t, g, d);
  } else {
    bool gated = p.kind == ALA_GATED_NORM;
    bool combine = p.kind == ALA_HC_COMBINE_NORM;
    float square = 0;
    for (int d = tid; d < D; d += 256) {
      float x = combine ? injection(p, t, g, d) : get(p.x, t, g, d);
      if (combine)
        put(p.combined, x, t, g, d);
      if (gated && !p.norm_before_gate) {
        float z = get(p.gate, t, g, d);
        x *= p.silu_gate ? silu(z) : sigmoid(z);
      }
      square = __fadd_rn(square, __fmul_rn(x, x));
    }
    float inv = rsqrtf(sum256(square, sums) / float(D) + p.epsilon);
    for (int d = tid; d < D; d += 256) {
      float x = combine ? get(p.combined, t, g, d) : get(p.x, t, g, d),
            activation = 1;
      if (gated) {
        float z = get(p.gate, t, g, d);
        activation = p.silu_gate ? silu(z) : sigmoid(z);
        if (!p.norm_before_gate)
          x *= activation;
      }
      float normalized = x * inv;
      if (gated && p.round_normalized)
        normalized = round_bf(normalized);
      float w = get(p.weight, p.weight_per_group ? g : 0, d);
      if (!gated && p.zero_centered)
        w += 1.0f;
      float weighted = normalized * w;
      if (gated && p.round_normalized && p.weight.dtype == ALA_BF16)
        weighted = round_bf(weighted);
      if (gated) {
        trace(p, normalized, t, g, d);
        trace(p, weighted, t, g, D + d);
      }
      put(p.output,
          weighted * (gated && p.norm_before_gate ? activation : 1.0f), t, g,
          d);
    }
  }
}
bool valid_pointwise(ala_pointwise_v1 const &p) {
  int T = p.rows.tokens, G = p.groups, D = p.width;
  if (p.header.version != 1 || p.header.struct_bytes != sizeof(p) ||
      !valid_rows(p.rows) || G < 1 || G > 65535 || D < 1 || D > 16384 ||
      p.kind < 0 || p.kind > 7 || !shape(p.x, ALA_BF16, {T, G, D}) ||
      !shape(p.errors, ALA_I32, {T}))
    return false;
  for (int f : {p.zero_centered, p.weight_per_group, p.round_normalized,
                p.norm_before_gate, p.silu_gate})
    if (f != 0 && f != 1)
      return false;
  int outg = p.kind == ALA_HC_GATE_MEAN ? 1 : G;
  if (!shape(p.output, ALA_BF16, {T, outg, D}))
    return false;
  bool norm = p.kind == ALA_GROUPED_NORM || p.kind == ALA_GATED_NORM ||
              p.kind == ALA_HC_COMBINE_NORM;
  if (p.kind == ALA_GATED_NORM && p.zero_centered)
    return false;
  if (norm &&
      (!(p.epsilon > 0) || !std::isfinite(p.epsilon) ||
       (p.weight.dtype != ALA_F32 && p.weight.dtype != ALA_BF16) ||
       !shape(p.weight, p.weight.dtype, {p.weight_per_group ? G : 1, D})))
    return false;
  if (!norm && !absent(p.weight))
    return false;
  if (p.kind == ALA_GROUPED_NORM || p.kind == ALA_HC_COMBINE_NORM) {
    if (!p.weight_per_group)
      return false; // no accidental stream-wise broadcast
  }
  if (p.kind == ALA_GATED_NORM || p.kind == ALA_HC_GATE_MEAN ||
      p.kind == ALA_PLE_GATE) {
    if (!shape(p.gate, ALA_BF16, {T, G, D}))
      return false;
  } else if (p.kind == ALA_HC_COMBINE || p.kind == ALA_HC_COMBINE_NORM) {
    if (!shape(p.gate, ALA_BF16, {T, G}))
      return false;
  } else if (!absent(p.gate))
    return false;
  bool value = p.kind == ALA_PLE_GATE || p.kind == ALA_HC_COMBINE ||
               p.kind == ALA_HC_COMBINE_NORM;
  if (value && !shape(p.value, ALA_BF16, {T, 1, D}))
    return false;
  if (p.kind == ALA_PLE_CONTRIBUTION && !shape(p.value, ALA_BF16, {T, G, D}))
    return false;
  if (!value && p.kind != ALA_PLE_CONTRIBUTION && !absent(p.value))
    return false;
  if (p.kind == ALA_HC_COMBINE_NORM) {
    if (!shape(p.combined, ALA_BF16, {T, G, D}))
      return false;
  } else if (!absent(p.combined))
    return false;
  if ((p.kind == ALA_HC_GATE_MEAN || p.kind == ALA_HC_COMBINE ||
       p.kind == ALA_HC_COMBINE_NORM || p.kind == ALA_PLE_GATE) &&
      G != 4)
    return false;
  if (p.kind == ALA_HC_SILU && G != 1)
    return false;
  int trace_width = p.kind == ALA_GATED_NORM || p.kind == ALA_HC_GATE_MEAN
                        ? 2 * D
                    : p.kind == ALA_PLE_GATE ? 6
                    : p.kind == ALA_HC_SILU  ? D
                                             : 0;
  if (!absent(p.trace) &&
      (!trace_width || !shape(p.trace, ALA_F32, {T, G, trace_width})))
    return false;
  return disjoint({p.x, p.weight, p.gate, p.value, p.rows.starts, p.rows.slots,
                   p.rows.num_seqs, p.rows.num_tokens, p.rows.upstream_errors},
                  {p.output, p.combined, p.trace, p.errors});
}
} // namespace actlize_la::serving::aux
extern "C" int ala_aux_v1_struct_size(int kind) {
  switch (kind) {
  case 0:
    return sizeof(ala_tensor_v1);
  case 1:
    return sizeof(ala_rows_v1);
  case 2:
    return sizeof(ala_pointwise_v1);
  case 3:
    return sizeof(ala_shortconv_v1);
  case 4:
    return sizeof(ala_ple_hash_v1);
  default:
    return -1;
  }
}
namespace actlize_la::serving::aux {
bool valid_conv(ala_shortconv_v1 const &);
bool valid_hash(ala_ple_hash_v1 const *);
} // namespace actlize_la::serving::aux
extern "C" int ala_aux_v1_validate(int kind, void const *p) {
  using namespace actlize_la::serving::aux;
  if (!p)
    return -1;
  bool ok = false;
  switch (kind) {
  case 0:
    ok = valid(*static_cast<ala_tensor_v1 const *>(p));
    break;
  case 1:
    ok = valid_rows(*static_cast<ala_rows_v1 const *>(p));
    break;
  case 2:
    ok = valid_pointwise(*static_cast<ala_pointwise_v1 const *>(p));
    break;
  case 3:
    ok = valid_conv(*static_cast<ala_shortconv_v1 const *>(p));
    break;
  case 4:
    ok = valid_hash(static_cast<ala_ple_hash_v1 const *>(p));
    break;
  }
  return ok ? 0 : -1;
}
extern "C" int ala_pointwise_v1_run(ala_pointwise_v1 const *p, void *stream) {
  using namespace actlize_la::serving;
  if (!p || !aux::valid_pointwise(*p))
    return -1;
  int groups = p->kind == ALA_HC_GATE_MEAN ? 1 : p->groups;
  aux::pointwise<<<dim3(p->rows.tokens, groups), 256, 0,
                   static_cast<Stream>(stream)>>>(*p);
  return last_error();
}
