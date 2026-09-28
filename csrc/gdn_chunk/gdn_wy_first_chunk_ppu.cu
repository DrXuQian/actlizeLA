// Explicit experimental composition; no change to the original/default path.
#include "gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"
#include "gdn_qsa/ppu/wy_first_chunk.hpp"

namespace gdn_qsa::wy::first_chunk {
int configure_state();
int launch_state(Inputs, Workspace, float*, gdn_arch::Stream);
int configure_output();
int launch_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}
namespace gdn_qsa::wy::solve_static {
int configure();
int launch_inverse(Inputs, Workspace, gdn_arch::Stream);
}
extern "C" int gdn_wy_forward_residual_full_chunk(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, gdn_arch::Stream);

extern "C" int gdn_wy_forward_residual_first_chunk(
    void const* q, void const* k, void const* v, void const* g, void const* beta, float const* initial,
    void* output, float* final, void* inverse, void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) {
  using namespace gdn_qsa::wy;
  using residual_warps8_hvlayout::Key;
  if (!first_chunk::eligible(sequence, initial != nullptr)) {
    return gdn_wy_forward_residual_full_chunk(q, k, v, g, beta, initial,
        output, final, inverse, snapshots, vnew, gates, batch, sequence,
        q_heads, value_heads, gate_fp32, stream);
  }
  if (batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 || value_heads % q_heads ||
      !Key::admitted_stride(int64_t(q_heads) * Dim) ||
      !Key::admitted_stride(int64_t(value_heads) * Dim) || inverse == snapshots)
    return int(hggcErrorInvalidValue);
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, {batch, sequence, q_heads, value_heads}};
  Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots),
               static_cast<BF16*>(vnew), gates};
  int rc = solve_static::configure();
  if (rc) return rc;
  rc = first_chunk::configure_state();
  if (rc) return rc;
  rc = first_chunk::configure_output();
  if (rc) return rc;
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;  // Never alias the live state snapshots.
  rc = solve_static::launch_inverse(p, inverse_ws, stream);
  if (rc) return rc;
  rc = first_chunk::launch_state(p, ws, final, stream);
  if (rc) return rc;
  return first_chunk::launch_output(p, ws, static_cast<BF16*>(output), stream);
}
