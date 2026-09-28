// Host composition only. Three opt-in arms reuse the immutable full-C64 state.
#include "gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"
#include "gdn_qsa/ppu/wy_full_chunk.hpp"

namespace gdn_qsa::wy {
namespace solve_static {
int configure();
int launch_inverse(Inputs, Workspace, gdn_arch::Stream);
}
namespace full_chunk_solve {
int configure();
int launch_inverse(Inputs, Workspace, gdn_arch::Stream);
}
namespace full_chunk_output {
int configure();
int launch_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}
namespace full_chunk {
int configure_state();
int launch_state(Inputs, Workspace, float*, gdn_arch::Stream);
}
namespace residual_warps8_hvlayout {
int configure_hvlayout_output();
int launch_hvlayout_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}
}  // namespace gdn_qsa::wy

extern "C" int gdn_wy_forward_residual_full_chunk(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, gdn_arch::Stream);

namespace gdn_qsa::wy {
template <bool FullSolve, bool FullOutput>
int forward_full_chunk_stages(
    void const* q, void const* k, void const* v, void const* g, void const* beta, float const* initial,
    void* output, float* final, void* inverse, void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) {
  using namespace residual_warps8_hvlayout;
  static_assert(FullSolve || FullOutput, "use the retained full-state control for no changed stage");
  if (!gate_cache::full_chunks(sequence)) {
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
  int rc;
  if constexpr (FullSolve) rc = full_chunk_solve::configure();
  else rc = solve_static::configure();
  if (rc) return rc;
  rc = full_chunk::configure_state();
  if (rc) return rc;
  if constexpr (FullOutput) rc = full_chunk_output::configure();
  else rc = configure_hvlayout_output();
  if (rc) return rc;
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;  // inverse and live H remain separate allocations.
  if constexpr (FullSolve) rc = full_chunk_solve::launch_inverse(p, inverse_ws, stream);
  else rc = solve_static::launch_inverse(p, inverse_ws, stream);
  if (rc) return rc;
  rc = full_chunk::launch_state(p, ws, final, stream);
  if (rc) return rc;
  if constexpr (FullOutput) return full_chunk_output::launch_output(p, ws, static_cast<BF16*>(output), stream);
  else return launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);
}
}  // namespace gdn_qsa::wy

// Keep external C ABI unchanged; the stage choice is compiled on the host.
#define GDN_FULL_CHUNK_ENTRY(Name, Solve, Output) \
extern "C" int Name( \
    void const* q, void const* k, void const* v, void const* g, void const* beta, float const* initial, \
    void* output, float* final, void* inverse, void* snapshots, void* vnew, float* gates, \
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) { \
  return gdn_qsa::wy::forward_full_chunk_stages<Solve, Output>(q, k, v, g, beta, initial, \
      output, final, inverse, snapshots, vnew, gates, batch, sequence, q_heads, value_heads, gate_fp32, stream); \
}
GDN_FULL_CHUNK_ENTRY(gdn_wy_forward_residual_full_chunk_solve, true, false)
GDN_FULL_CHUNK_ENTRY(gdn_wy_forward_residual_full_chunk_output, false, true)
GDN_FULL_CHUNK_ENTRY(gdn_wy_forward_residual_full_chunk_both, true, true)
#undef GDN_FULL_CHUNK_ENTRY
