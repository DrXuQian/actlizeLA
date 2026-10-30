#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "launch.h"

namespace {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> initial, bool output_final_state) {
  constexpr int Dim = 128, Chunk = 64;
  TORCH_CHECK(q.dim() == 4 && q.size(3) == Dim && k.sizes() == q.sizes(),
              "ppu15 q/k must have identical [B,T,Hk,128] shapes");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(3) == Dim,
              "ppu15 v must be [B,T,Hv,128]");
  int64_t const B = q.size(0), T = q.size(1), Hk = q.size(2), Hv = v.size(2);
  TORCH_CHECK(B > 0 && T > 0 && Hk > 0 && Hv > 0 && Hv % Hk == 0, "invalid ppu15/GVA extents");
  TORCH_CHECK(g.sizes() == torch::IntArrayRef({B,T,Hv}) && beta.sizes() == g.sizes(),
              "ppu15 g/beta must be [B,T,Hv]");
  for (auto const& x : {q,k,v,g,beta})
    TORCH_CHECK(x.is_cuda() && x.device() == q.device() && x.is_contiguous(),
                "ppu15 inputs must be contiguous on one device");
  for (auto const& x : {q,k,v,beta})
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "ppu15 q/k/v/beta must be BF16");
  for (auto const& x : {q,k,v})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "ppu15 AIU inputs require 16-byte alignment");
  TORCH_CHECK(g.scalar_type() == torch::kBFloat16 || g.scalar_type() == torch::kFloat32,
              "ppu15 natural-log gate must be BF16 or FP32");
  TORCH_CHECK(B <= INT_MAX && T <= INT_MAX && Hv <= INT_MAX / Dim, "ppu15 descriptor extent overflow");
  int64_t const chunks = (T - 1) / Chunk + 1;
  TORCH_CHECK(B <= INT_MAX / Hv && B * Hv <= INT_MAX / chunks && B * Hv <= INT_MAX / 4,
              "ppu15 grid overflow");
  if (initial.has_value()) {
    auto const& h = *initial;
    TORCH_CHECK(h.device() == q.device() && h.is_contiguous() && h.scalar_type() == torch::kFloat32 &&
                h.sizes() == torch::IntArrayRef({B,Hv,Dim,Dim}),
                "ppu15 initial state must be FP32 [B,Hv,128,128] on the same device");
  }
  c10::cuda::CUDAGuard guard(q.device());
  auto result = torch::empty_like(v);
  auto final = torch::empty(output_final_state ? std::vector<int64_t>{B,Hv,Dim,Dim} : std::vector<int64_t>{0},
                            q.options().dtype(torch::kFloat32));
  int64_t const groups = B * Hv * chunks;
  auto inverse = torch::empty({groups,Chunk,Chunk}, q.options());
  auto snapshots = torch::empty({groups,Dim,Dim}, q.options());
  auto vnew = torch::empty({groups,Dim,Chunk}, q.options());
  auto gates = torch::empty({groups,Chunk}, q.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int const rc = actlize_la_ppu15_forward(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      initial.has_value() ? initial->data_ptr<float>() : nullptr, result.data_ptr(),
      output_final_state ? final.data_ptr<float>() : nullptr, inverse.data_ptr(), snapshots.data_ptr(),
      vnew.data_ptr(), gates.data_ptr<float>(), int(B), int(T), int(Hk), int(Hv),
      g.scalar_type() == torch::kFloat32, static_cast<void*>(stream));
  TORCH_CHECK(rc == 0, "PPU1.5 GDN launch failed: status=", rc);
  return {result, final};
}
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("target_arch", &actlize_la_ppu15_target_arch);
  m.def("forward", &forward, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true);
}
