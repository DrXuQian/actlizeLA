#include "launch.h"
#include "../../gdn_chunk/gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_full_chunk.hpp"
#include "gdn_qsa/ppu/wy_dispatch.hpp"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"
#include <climits>

// Bind the external workspace contract to the actual retained writer pitch.
static_assert(gdn_qsa::wy::split_prepare::Plan::inverse_base(1) == 128 * 128);
static_assert(gdn_qsa::wy::split_prepare::Plan::InverseElements == 64 * 64);

namespace gdn_qsa::wy::solve_static {
int configure();
int launch_inverse(Inputs, Workspace, gdn_arch::Stream);
int configure_native_bf16();
int launch_native_bf16(Inputs, Workspace, gdn_arch::Stream);
}
namespace gdn_qsa::wy::full_chunk {
int configure_state();
int launch_state(Inputs, Workspace, float*, gdn_arch::Stream);
int configure_native_bf16_state();
int launch_native_bf16_state(Inputs, Workspace, float*, gdn_arch::Stream);
}
namespace gdn_qsa::wy::gate_cache {
int configure_state();
int launch_state(Inputs, Workspace, float*, gdn_arch::Stream);
}
namespace gdn_qsa::wy::residual_warps8_hvlayout {
int configure_hvlayout_output();
int launch_hvlayout_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
int configure_native_bf16_output();
int launch_native_bf16_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}

extern "C" int actlize_la_ppu10_core_route_v2(
    int batch, int sequence, int key_heads, int value_heads, bool gate_fp32) {
  return int(gdn_qsa::wy::core_route({batch, sequence, key_heads, value_heads}, gate_fp32));
}

extern "C" int actlize_la_ppu10_configure() {
  using namespace gdn_qsa::wy;
  if (int rc = solve_static::configure()) return rc;
  if (int rc = full_chunk::configure_state()) return rc;
  if (int rc = gate_cache::configure_state()) return rc;
  if (int rc = residual_warps8_hvlayout::configure_hvlayout_output()) return rc;
  if (int rc = solve_static::configure_native_bf16()) return rc;
  if (int rc = full_chunk::configure_native_bf16_state()) return rc;
  return residual_warps8_hvlayout::configure_native_bf16_output();
}

extern "C" int actlize_la_ppu10_forward_prepared(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* opaque_stream) {
  using namespace gdn_qsa::wy;
  if (!q || !k || !v || !g || !beta || !output || !inverse || !snapshots || !vnew || !gates ||
      inverse == snapshots || inverse == vnew || snapshots == vnew ||
      batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 ||
      value_heads % q_heads || q_heads > INT_MAX / Dim || value_heads > INT_MAX / Dim)
    return int(hggcErrorInvalidValue);
  Shape const shape{batch, sequence, q_heads, value_heads};
  if (int64_t(batch) * value_heads > INT_MAX / 4 ||
      int64_t(batch) * value_heads > INT_MAX / shape.chunks())
    return int(hggcErrorInvalidValue);
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, shape};
  Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots),
               static_cast<BF16*>(vnew), gates};
  auto stream = static_cast<gdn_arch::Stream>(opaque_stream);
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;  // Padded inverse pitch is 128x128, NOT 64x64.
  if (core_route(shape, gate_fp32) == CoreRoute::NativeBf16) {
    if (int rc = solve_static::launch_native_bf16(p, inverse_ws, stream)) return rc;
    if (int rc = full_chunk::launch_native_bf16_state(p, ws, final, stream)) return rc;
    return residual_warps8_hvlayout::launch_native_bf16_output(p, ws, static_cast<BF16*>(output), stream);
  }
  if (int rc = solve_static::launch_inverse(p, inverse_ws, stream)) return rc;
  int const rc = gate_cache::full_chunks(sequence)
      ? full_chunk::launch_state(p, ws, final, stream)
      : gate_cache::launch_state(p, ws, final, stream);
  if (rc) return rc;
  return residual_warps8_hvlayout::launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);
}
