#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "gdn_qsa/wy_contract.hpp"

extern "C" int gdn_wy_forward(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_delivery(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t, unsigned);
extern "C" int gdn_wy_forward_residual(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_prefetch(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_operands(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_v16(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_blayout(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);

extern "C" int gdn_wy_forward_residual_warps8(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_warps8_blayout(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_warps8_operands(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_warps8_hlayout(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_warps8_hvlayout(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_warps8_metadata(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_residual_solve_static(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);

namespace {
extern "C" int gdn_wy_forward_residual_gate_cache(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
template <bool Residual = false, unsigned Variant = 0>
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> initial,
    bool output_final_state, unsigned delivery) {
  using namespace gdn_qsa::wy;
  static_assert(Variant <= 12 && (!Variant || Residual), "invalid residual-only delivery");
  TORCH_CHECK(valid_delivery(delivery), "invalid or conflicting WY delivery mask");
  TORCH_CHECK(!Residual || delivery == 0, "residual is an algorithm, not a WY delivery mask");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == Dim && k.sizes() == q.sizes(),
              "WY q/k must have identical [B,S,Hk,128] shapes");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(3) == Dim,
              "WY v must be [B,S,Hv,128]");
  int64_t const B = q.size(0), S = q.size(1), Hk = q.size(2), Hv = v.size(2);
  TORCH_CHECK(B > 0 && S > 0 && Hk > 0 && Hv > 0 && Hv % Hk == 0, "invalid WY/GVA extents");
  TORCH_CHECK(g.sizes() == torch::IntArrayRef({B, S, Hv}) && beta.sizes() == g.sizes(),
              "WY g/beta must be [B,S,Hv]");
  for (auto const& t : {q, k, v, g, beta}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous(),
                "WY inputs must be contiguous on one PPU");
  }
  for (auto const& t : {q, k, v, beta})
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, "WY q/k/v/beta must be BF16");
  TORCH_CHECK(g.scalar_type() == torch::kBFloat16 || g.scalar_type() == torch::kFloat32,
              "WY natural-log gate must be BF16 or FP32");
  TORCH_CHECK(B <= INT_MAX && S <= INT_MAX && Hv <= INT_MAX,
              "WY dimensions exceed 32-bit kernel extents");
  TORCH_CHECK(!(Residual || (delivery & AiuOptions)) || Hv <= INT_MAX / Dim,
              "WY AIU row pitch exceeds 32-bit element descriptor");
  int64_t const nt = (S - 1) / Chunk + 1;
  constexpr int slices = Variant == 3 ? 8 : 4;
  TORCH_CHECK(B <= INT_MAX / Hv && B * Hv <= INT_MAX / nt && B * Hv <= INT_MAX / slices,
              "WY launch grid overflow");
  if (initial.has_value()) {
    auto const& h = *initial;
    TORCH_CHECK(h.device() == q.device() && h.is_contiguous() && h.scalar_type() == torch::kFloat32 &&
                h.sizes() == torch::IntArrayRef({B, Hv, Dim, Dim}),
                "WY initial state must be contiguous FP32 [B,Hv,128,128] on the same PPU");
  }
  c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(v);
  auto final = output_final_state ? torch::empty({B, Hv, Dim, Dim}, q.options().dtype(torch::kFloat32))
                                  : torch::empty({0}, q.options().dtype(torch::kFloat32));
  // Residual uses w as a separate padded inverse plane, never H snapshots.
  // U is absent; do not allocate/materialize either old W or old U.
  auto w = torch::empty({B * Hv * nt, Residual ? Dim : Chunk, Dim}, q.options());
  auto u = Residual ? torch::empty({0}, q.options()) : torch::empty_like(w);
  auto vn = torch::empty({B * Hv * nt, Chunk, Dim}, q.options());
  auto snapshots = torch::empty({B * Hv * nt, Dim, Dim}, q.options());
  auto gates = torch::empty({B * Hv * nt, Chunk}, q.options().dtype(torch::kFloat32));
  int rc;
  if constexpr (Residual) {
    if constexpr (Variant != 0) {
      constexpr auto launch = Variant == 1 ? gdn_wy_forward_residual_prefetch :
                              Variant == 2 ? gdn_wy_forward_residual_operands :
                              Variant == 3 ? gdn_wy_forward_residual_v16 :
                              Variant == 4 ? gdn_wy_forward_residual_blayout :
                              Variant == 5 ? gdn_wy_forward_residual_warps8 :
                              Variant == 6 ? gdn_wy_forward_residual_warps8_blayout :
                              Variant == 7 ? gdn_wy_forward_residual_warps8_operands :
                              Variant == 8 ? gdn_wy_forward_residual_warps8_hlayout :
                              Variant == 9 ? gdn_wy_forward_residual_warps8_hvlayout :
                              Variant == 10 ? gdn_wy_forward_residual_warps8_metadata :
                              Variant == 11 ? gdn_wy_forward_residual_solve_static :
                                              gdn_wy_forward_residual_gate_cache;
      rc = launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        initial.has_value() ? initial->data_ptr<float>() : nullptr, out.data_ptr(),
        output_final_state ? final.data_ptr<float>() : nullptr, w.data_ptr(),
        snapshots.data_ptr(), vn.data_ptr(), gates.data_ptr<float>(), int(B), int(S), int(Hk), int(Hv),
        g.scalar_type() == torch::kFloat32, at::cuda::getCurrentCUDAStream().stream());
    } else {
      rc = gdn_wy_forward_residual(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        initial.has_value() ? initial->data_ptr<float>() : nullptr, out.data_ptr(),
        output_final_state ? final.data_ptr<float>() : nullptr, w.data_ptr(),
        snapshots.data_ptr(), vn.data_ptr(), gates.data_ptr<float>(), int(B), int(S), int(Hk), int(Hv),
        g.scalar_type() == torch::kFloat32, at::cuda::getCurrentCUDAStream().stream());
    }
  } else {
    rc = gdn_wy_forward_delivery(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      initial.has_value() ? initial->data_ptr<float>() : nullptr, out.data_ptr(),
      output_final_state ? final.data_ptr<float>() : nullptr, w.data_ptr(), u.data_ptr(),
      snapshots.data_ptr(), vn.data_ptr(), gates.data_ptr<float>(), int(B), int(S), int(Hk), int(Hv),
      g.scalar_type() == torch::kFloat32, at::cuda::getCurrentCUDAStream().stream(), delivery);
  }
  TORCH_CHECK(rc == 0, "PPU WY kernel launch failed: status=", rc);
  return {out, final};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward<false>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual", &forward<true>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_prefetch", &forward<true, 1>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_operands", &forward<true, 2>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_v16", &forward<true, 3>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_warps8", &forward<true, 5>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_warps8_blayout", &forward<true, 6>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_blayout", &forward<true, 4>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_warps8_operands", &forward<true, 7>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_warps8_hlayout", &forward<true, 8>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_warps8_hvlayout", &forward<true, 9>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_warps8_metadata", &forward<true, 10>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_solve_static", &forward<true, 11>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
  m.def("residual_gate_cache", &forward<true, 12>, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
}
