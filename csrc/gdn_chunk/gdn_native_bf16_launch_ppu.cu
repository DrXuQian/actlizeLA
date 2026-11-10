// Opt-in host ABI: same residual/TF32x3 algorithm, native BF16 conversions.
#include "gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_full_chunk.hpp"
#include "gdn_qsa/ppu/native_bf16.cuh"
#include <climits>

namespace gdn_qsa::wy::native_bf16_solve {
int configure();
int launch_inverse_default(Inputs, Workspace, gdn_arch::Stream);
}
namespace gdn_qsa::wy::native_bf16_state {
int configure_state();
int launch_state(Inputs, Workspace, float*, gdn_arch::Stream);
}
namespace gdn_qsa::wy::native_bf16_output {
int configure_hvlayout_output();
int launch_hvlayout_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}

extern "C" int actlize_la_ppu10_native_bf16_configure() {
  using namespace gdn_qsa::wy;
  if (int rc = native_bf16_solve::configure()) return rc;
  if (int rc = native_bf16_state::configure_state()) return rc;
  return native_bf16_output::configure_hvlayout_output();
}

extern "C" int actlize_la_ppu10_native_bf16_forward_prepared(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* opaque_stream) {
  using namespace gdn_qsa::wy;
  if (!q || !k || !v || !g || !beta || !output || !inverse || !snapshots || !vnew || !gates ||
      inverse == snapshots || inverse == vnew || snapshots == vnew ||
      batch <= 0 || sequence <= 0 || sequence % Chunk || q_heads <= 0 || value_heads <= 0 ||
      value_heads % q_heads || q_heads > INT_MAX / Dim || value_heads > INT_MAX / Dim)
    return int(hggcErrorInvalidValue);
  Shape const shape{batch, sequence, q_heads, value_heads};
  if (int64_t(batch) * value_heads > INT_MAX / 4 ||
      int64_t(batch) * value_heads > INT_MAX / shape.chunks() ||
      default_core_route(shape) == CoreRoute::Solve8Value64)
    return int(hggcErrorInvalidValue);  // This experiment retains the V32 geometry.
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, shape};
  Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots),
               static_cast<BF16*>(vnew), gates};
  auto stream = static_cast<gdn_arch::Stream>(opaque_stream);
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;
  if (int rc = native_bf16_solve::launch_inverse_default(p, inverse_ws, stream)) return rc;
  if (int rc = native_bf16_state::launch_state(p, ws, final, stream)) return rc;
  return native_bf16_output::launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);
}

// Admission only, never part of an ACU subject. Raw FP32 input bits include
// every BF16 exponent/sign pattern, tie boundaries and special values.
__global__ void gdn_native_bf16_probe(uint32_t const* input, uint16_t* old_bits,
                                     uint16_t* new_bits, int count) {
  int const i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (i < count) {
    float x;
    __builtin_memcpy(&x, input + i, sizeof(x));
    old_bits[i] = cutlass::bfloat16_t(x).raw();
    new_bits[i] = gdn_qsa::wy::native_bf16::convert(x).raw();
  }
}
extern "C" int actlize_la_ppu10_native_bf16_probe(
    uint32_t const* input, uint16_t* old_bits, uint16_t* new_bits, int count, void* stream) {
  if (!input || !old_bits || !new_bits || old_bits == new_bits || count <= 0 || count > INT_MAX - 255)
    return int(hggcErrorInvalidValue);
  gdn_native_bf16_probe<<<(count + 255) / 256, 256, 0, static_cast<hggcStream_t>(stream)>>>(input, old_bits, new_bits, count);
  return int(hggcGetLastError());
}
