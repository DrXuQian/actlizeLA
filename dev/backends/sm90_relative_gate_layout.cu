#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include "relative_gate_layout.cuh"
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
using namespace cute;
using S=tuple<int64_t,_1,int32_t>;
using B=kda::sm90::kernel::FlatBuilderKdaFwd<cutlass::bfloat16_t,float,float,
    Shape<_64,_64,_128>,S,S,S,S,cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
void require(bool b) { if(!b) throw std::runtime_error("relative gate coverage"); }
void check_owners(int threads) {
    std::array<int,8192> owners{};
    auto mma=typename B::CollectiveMainloop::TiledMmaKV{};
    int distinct_thread_tokens=0;
    for(int tid=0;tid<threads;++tid) {
        auto coordinates=mma.get_slice(tid).partition_A(make_identity_tensor(Shape<_128,_64>{}));
        std::set<int> tokens;
        for(int i=0;i<size(coordinates);++i) {
            auto [v,t]=coordinates(i);
            require(v>=0 && v<128 && t>=0 && t<64);
            ++owners[int(v)*64+int(t)]; tokens.insert(int(t));
        }
        distinct_thread_tokens+=int(tokens.size());
    }
    for(int count:owners) require(count==1);
    require(distinct_thread_tokens==4096);
}
void check_producer(bool wrong_last) {
    using Mainloop=typename B::CollectiveMainloop;
    constexpr int stages=Mainloop::StagesAlpha::value;
    static_assert(stages==2);
    for(int valid=1;valid<=64;++valid) {
        int last=gdn::sm90::relative_gate_last_lane(valid)+
                 (gdn::sm90::relative_gate_last_is_hi(valid)?32:0);
        if(wrong_last) last=(last+1)%64;
        require(last==valid-1);
        std::array<int,64*stages> writers{};
        for(int stage=0;stage<stages;++stage) for(int lane=0;lane<32;++lane)
            for(int half=0;half<2;++half) {
                int token=lane+half*32;
                int index=gdn::sm90::relative_gate_index(stage,token);
                require(index>=0 && index<128); ++writers[index];
                require(index==stage*64+token);
            }
        for(int count:writers) require(count==1);
    }
}
template<class F> void expect_red(F check) {
    try { check(); }
    catch(std::runtime_error const&) { return; }
    throw std::runtime_error("planted map defect escaped");
}
int main() {
    check_owners(256);
    check_producer(false);
    expect_red([] { check_owners(255); });
    expect_red([] { check_producer(true); });
    std::cout<<"relative-gate: actual8192/8192 owners;4096 per-thread token evaluations ->64 producer tokens;64 tails/2stages;wrong-last+missing-owner EXPECTED_RED/PASS\n";
}
