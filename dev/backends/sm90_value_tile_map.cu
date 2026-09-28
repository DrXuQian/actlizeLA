#include "value_types.cuh"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace cute;
using BF16=cutlass::bfloat16_t;
using Parent=gdn::sm90::ValueKernelTypes<BF16,false>;
using Split=gdn::sm90::ValueKernelTypes<BF16,false,64,104>;
void need(bool condition) { if(!condition) throw std::runtime_error("value-tile ownership/protocol"); }

void owners(int tiles,bool overlap) {
    auto parent_mma=typename Parent::Collective::TiledMmaKV{};
    auto child_mma=typename Split::Collective::TiledMmaKV{};
    auto parent_o=typename Parent::Collective::TiledMmaO1{};
    auto child_o=typename Split::Collective::TiledMmaO1{};
    std::array<int,128*128> state{};
    std::array<int,128*64> output{};
    for(int tile=0;tile<tiles;++tile) for(int tid=0;tid<128;++tid) {
        int start=gdn::sm90::ValueTileScheduler<64>::value_begin(overlap?0:tile);
        auto h=child_mma.get_slice(tid).partition_C(make_identity_tensor(Shape<_64,_128>{}));
        auto ph=parent_mma.get_slice(tid+tile*128).partition_C(make_identity_tensor(Shape<_128,_128>{}));
        need(size(h)==size(ph));
        for(int i=0;i<size(h);++i) {
            auto [v,k]=h(i); auto [pv,pk]=ph(i);
            need(v>=0 && v<64 && k>=0 && k<128);
            need(pv==v+start && pk==k);
            ++state[int(k)*128+int(v)+start];
        }
        auto o=child_o.get_slice(tid).partition_C(make_identity_tensor(Shape<_64,_64>{}));
        auto po=parent_o.get_slice(tid+tile*128).partition_C(make_identity_tensor(Shape<_128,_64>{}));
        need(size(o)==size(po));
        for(int i=0;i<size(o);++i) {
            auto [v,t]=o(i); auto [pv,pt]=po(i);
            need(v>=0 && v<64 && t>=0 && t<64);
            need(pv==v+start && pt==t);
            ++output[int(t)*128+int(v)+start];
        }
    }
    for(int count:state) need(count==1);
    for(int count:output) need(count==1);
}

void global_maps(bool wrong_stride) {
    for(int B:{1,2,3}) for(int T:{1,33,64,65,127,128,129,2048})
    for(int H:{1,2,16,32}) {
        auto abi=gdn::sm90::state_layout<128,128>(H,B);
        for(int b=0;b<B;++b) for(int h=0;h<H;++h) for(int slice=0;slice<2;++slice) {
            int begin=gdn::sm90::ValueTileScheduler<64>::value_begin(slice);
            auto tensor=make_identity_tensor(shape(abi))(_,_,h,b);
            auto half=local_tile(domain_offset(make_coord(_0{},begin),tensor),
                                 Shape<_128,_64>{},make_coord(_0{},_0{}));
            for(int k=0;k<128;++k) for(int v=0;v<64;++v) {
                // Follow the exact domain_offset/local_tile used by state;
                // independently anchor [batch,head,K,V] linear storage.
                auto c=half(k,v);
                need(abi(c)==((int64_t(b)*H+h)*128+k)*128+begin+v);
            }
            for(int t=0;t<T;++t) for(int v=0;v<64;++v) {
                int64_t got=gdn::sm90::value_output_index(int64_t(b)*T+t,H,h,begin,v);
                if(wrong_stride) got=((int64_t(b)*T+t)*H+h)*64+begin+v;
                need(got==((int64_t(b)*T+t)*H+h)*128+begin+v);
            }
        }
    }
}
template<class Gate,bool Initial,int Aux>
void actual_type() {
    using Types=gdn::sm90::ValueKernelTypes<Gate,Initial,64,Aux>;
    using K=typename Types::Kernel;
    using C=typename Types::Collective;
    static_assert(C::NumStateMmaWarpGroups==1 && K::MaxThreadsPerBlock==384);
    static_assert(size(typename C::TiledMmaKV{})==128 && size(typename C::TiledMmaO1{})==128);
    static_assert(K::LdStRegisterRequirement==24 && K::StateMmaRegisterRequirement==192 &&
                  K::AuxMmaRegisterRequirement==Aux);
    typename Types::Scheduler::Params p{dim3(32,1,1),1,32,2};
    auto grid=Types::Scheduler::get_grid_shape(p);
    need(grid.x==32 && grid.y==2 && grid.z==1);
    std::cout<<"value64 actual gate_fp32="<<std::is_same_v<Gate,float><<" initial="<<Initial
             <<" aux="<<Aux<<" threads="<<K::MaxThreadsPerBlock<<" shared="<<K::SharedStorageSize<<"\n";
}
template<class F> void red(F f) {
    try { f(); } catch(std::runtime_error const&) { return; }
    throw std::runtime_error("planted defect escaped");
}
void arrivals(bool stale) {
    using K=typename Split::Kernel;
    // Enumerate the actual384-thread roles: loader32 alpha readers, one
    // 128-thread state group, one128-thread auxiliary group, no phantom WG.
    int data=0,alpha=0;
    for(int tid=0;tid<K::MaxThreadsPerBlock;++tid) {
        if(tid>=128) { ++data;++alpha; }
        else if(tid/32==3) ++alpha;
    }
    need(K::QKInputConsumers==data+(stale?128:0));
    need(K::BetaConsumers==data && K::AlphaConsumers==alpha);
    need(K::StateThreads==128 && K::AuxThreads==128);
}
int main() {
    actual_type<BF16,false,104>();actual_type<BF16,true,104>();
    actual_type<float,false,104>();actual_type<float,true,104>();
    actual_type<BF16,false,232>();actual_type<BF16,true,232>();
    actual_type<float,false,232>();actual_type<float,true,232>();
    owners(2,false);global_maps(false);arrivals(false);
    red([]{owners(1,false);});red([]{owners(2,true);});red([]{global_maps(true);});
    red([]{arrivals(true);});
    std::cout<<"value64 actual fragments: state16384/16384 output8192/8192 EXACT-ONCE; "
               "parent per-lane coordinates match;96 shape maps;4 negatives PASS\n";
}
