#include "scalar_gdn_state.cuh"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using namespace kda::sm90::kernel;
using BF16=cutlass::bfloat16_t;
using GdnStride=cute::tuple<int64_t,_1,int32_t>;
using Builder=FlatBuilderKdaFwd<BF16,float,float,Shape<_64,_64,_128>,
    GdnStride,GdnStride,GdnStride,GdnStride,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    std::tuple<Option<Tag::kElementGateGmem,BF16>,Option<Tag::kElementBetaGmem,BF16>>>;
using Collective=gdn::sm90::ScalarGdnState<typename Builder::CollectiveMainloop,true>;
void need(bool p) { if(!p) throw std::runtime_error("O2/KV delta ownership"); }
void map(bool wrong,bool missing) {
    auto o2=typename Collective::TiledMmaO2{};
    auto kv=typename Collective::TiledMmaKV{};
    static_assert(std::is_same_v<typename Collective::TiledMmaO2::LayoutA_TV,
                                 typename Collective::TiledMmaKV::LayoutA_TV>);
    std::array<int,8192> owners{};int checked=0;
    for(int tid=0;tid<256-(missing?1:0);++tid) {
        auto a=o2.get_slice(tid).partition_A(make_identity_tensor(Shape<_128,_64>{}));
        auto b=kv.get_slice(tid).partition_A(make_identity_tensor(Shape<_128,_64>{}));
        need(size(a)==size(b));
        for(int i=0;i<size(a);++i) {
            auto [v,t]=a(i);auto [w,u]=b(wrong?(i+1)%size(b):i);
            need(v==w && t==u);need(v>=0 && v<128 && t>=0 && t<64);
            ++owners[int(v)*64+int(t)];++checked;
        }
    }
    need(checked==8192);for(int x:owners)need(x==1);
}
template<class F> void red(F f) { try{f();}catch(std::runtime_error const&){return;}
    throw std::runtime_error("negative escaped"); }
int main(){map(false,false);red([]{map(true,false);});red([]{map(false,true);});
    std::cout<<"[SM90 actual delta operands] 8192/8192 exact-once, O2/KV identical coordinates;2negativesPASS\n";}
