#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "launch.h"

#ifndef GDN_SM90_TARGET_NAME
#error "the build must bind an explicit target identity"
#endif

namespace {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> initial,
    bool output_final_state) {
    TORCH_CHECK(q.dim()==4 && q.size(3)==128 && k.sizes()==q.sizes(),
                "fused_sm90 q/k must be [B,T,Hk,128]");
    TORCH_CHECK(v.dim()==4 && v.size(0)==q.size(0) && v.size(1)==q.size(1) && v.size(3)==128,
                "fused_sm90 v must be [B,T,Hv,128]");
    auto B=q.size(0), T=q.size(1), H=q.size(2), HV=v.size(2);
    TORCH_CHECK(B>0 && T>0 && H>0 && HV>0 && HV%H==0 &&
                B<=INT32_MAX/T && B<=INT32_MAX/HV && HV<=INT32_MAX/128,
                "invalid fused_sm90 GVA/extents/grid");
    TORCH_CHECK(g.sizes()==torch::IntArrayRef({B,T,HV}) && beta.sizes()==g.sizes(),
                "fused_sm90 g/beta must be [B,T,Hv]");
    for (auto const& t : {q,k,v,g,beta})
        TORCH_CHECK(t.is_cuda() && t.device()==q.device() && t.is_contiguous(),
                    "fused_sm90 inputs must be contiguous on one device");
    for (auto const& t : {q,k,v,beta})
        TORCH_CHECK(t.scalar_type()==torch::kBFloat16, "fused_sm90 q/k/v/beta must be BF16");
    TORCH_CHECK(g.scalar_type()==torch::kFloat32 || g.scalar_type()==torch::kBFloat16,
                "fused_sm90 natural-log gate must be BF16/FP32");
    for (auto const& t : {q,k,v})
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0, "unaligned TMA input");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad() &&
                !g.requires_grad() && !beta.requires_grad(), "fused_sm90 is forward-only");
    if (initial) {
        TORCH_CHECK(initial->is_cuda() && initial->device()==q.device() && initial->is_contiguous() &&
                    initial->scalar_type()==torch::kFloat32 &&
                    initial->sizes()==torch::IntArrayRef({B,HV,128,128}) && !initial->requires_grad(),
                    "initial state must be contiguous FP32 [B,Hv,K,V], forward-only");
    }
    c10::cuda::CUDAGuard guard(q.device());
    auto properties = at::cuda::getDeviceProperties(q.get_device());
#if defined(GDN_SM90_PPU17) && !defined(GDN_SM90_SOURCE_CHECK)
    TORCH_CHECK(std::string(properties->name).find("PPU")!=std::string::npos,
                "native PPU1.7 binary cannot run on a CUDA GPU");
#else
    TORCH_CHECK(properties->major==9 && properties->minor==0,
                "SM90a binary requires SM90a; no SM80/SM120 fallback");
#endif
    auto out=torch::empty_like(v);
    auto final=output_final_state ? torch::empty({B,HV,128,128},v.options().dtype(torch::kFloat32)) : torch::Tensor();
    gdn::sm90::Arguments args{q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr(),beta.data_ptr(),
        initial ? initial->data_ptr<float>() : nullptr, out.data_ptr(),
        output_final_state ? final.data_ptr<float>() : nullptr,
        int(B),int(T),int(H),int(HV),g.scalar_type()==torch::kFloat32};
    gdn::sm90::launch(args,at::cuda::getCurrentCUDAStream(q.get_device()));
    return {out,final};
}
} // namespace

#include "configuration.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward",&forward);
    m.attr("target")=GDN_SM90_TARGET_NAME;
    m.attr("math_contract")="cula-scalar-gdn-fused-bf16-v1";
    m.attr("numeric_schedule")=gdn::sm90::ConfigurationTraits::Tuned
        ? "scalar-GDN-BF16-WGMMA; FP16 inverse precomputed by auxiliary warpgroup"
        : "original control; TF32 conditioning and FP16 inverse";
    m.attr("device_admission")="UNVERIFIED";
    m.attr("configuration")=gdn::sm90::ConfigurationTraits::Name;
}
