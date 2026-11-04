#include "common.cuh"
#include "launch.h"
#include <climits>

extern "C" int actlize_la_ppu15_target_arch() { return 150; }

extern "C" int actlize_la_ppu15_configure() {
  using namespace actlize_la::ppu15;
  if (int rc=configure_prepare()) return rc;
  if (int rc=configure_state()) return rc;
  return configure_output();
}
static int forward_impl(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* result, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* opaque_stream,
    bool configure) {
  using namespace actlize_la::ppu15;
  if (!q || !k || !v || !g || !beta || !result || !inverse || !snapshots || !vnew || !gates ||
      inverse == snapshots || inverse == vnew || snapshots == vnew ||
      batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 || value_heads % q_heads ||
      value_heads > INT_MAX / Dim)
    return int(hggcErrorInvalidValue);
  Shape const shape{batch, sequence, q_heads, value_heads};
  if (int64_t(batch) * value_heads > INT_MAX / (Dim / ValueTile) ||
      int64_t(batch) * value_heads > INT_MAX / shape.chunks())
    return int(hggcErrorInvalidValue);
  if (configure) {
    if (int rc=actlize_la_ppu15_configure()) return rc;
  }
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, shape};
  Workspace ws{static_cast<BF16*>(inverse), static_cast<BF16*>(snapshots), static_cast<BF16*>(vnew), gates};
  auto const stream = static_cast<hggcStream_t>(opaque_stream);
  int rc = prepare(p, ws, stream);
  if (rc) return rc;
  rc = state(p, ws, final, stream);
  if (rc) return rc;
  return output(p, ws, static_cast<BF16*>(result), stream);
}

extern "C" int actlize_la_ppu15_forward_prepared(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* result, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* stream) {
  return forward_impl(q,k,v,g,beta,initial,result,final,inverse,snapshots,vnew,gates,
                      batch,sequence,q_heads,value_heads,gate_fp32,stream,false);
}

// Preserve the existing allocating/public entry's initialization behavior.
// Serving callers configure once, before graph capture, and use _prepared.
extern "C" int actlize_la_ppu15_forward(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* result, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* stream) {
  return forward_impl(q,k,v,g,beta,initial,result,final,inverse,snapshots,vnew,gates,
                      batch,sequence,q_heads,value_heads,gate_fp32,stream,true);
}
