// Run on CPU. Types/layouts are those instantiated by the new shipping TU.
#include <cute/tensor.hpp>
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
using namespace cute;
using S=cute::tuple<int64_t,_1,int32_t>;
using Builder=kda::sm90::kernel::FlatBuilderKdaFwd<cutlass::bfloat16_t,float,float,
    Shape<_64,_64,_128>,S,S,S,S,cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
using M=Builder::CollectiveMainloop;
void require(bool v){if(!v)throw std::runtime_error("SM90 layout mismatch");}
int main(int argc,char** argv){
    bool bad=argc>1;
    // Swizzled pointers use low address bits: use the production storage's
    // alignment, not an arbitrarily aligned std::vector allocation.
    typename M::SharedStorage storage{};
    auto s=make_tensor(make_smem_ptr(storage.smem_alpha.data()),typename M::QKQSmemLayoutAlpha{});
    for(int stage=0;stage<2;++stage)for(int lane=0;lane<32;++lane)for(int d=0;d<128;++d){
        s(lane,d,stage)=stage*64+lane;
        s(lane+32,d,stage)=stage*64+lane+(bad?31:32);
    }
    for(int stage=0;stage<2;++stage)for(int row=0;row<64;++row)for(int d=0;d<128;++d)
        require(s(row,d,stage)==stage*64+row);
    static_assert(cosize(typename M::QKQSmemLayoutAlpha{})==128);
    // Execute the actual auto-vectorized A copy on CPU, including stride0
    // broadcast. Shape equality alone is not consumer-layout evidence.
    using Copy=Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,float>;
    auto logical=make_identity_tensor(Shape<_64,_128>{});
    for(int tid=0;tid<128;++tid)for(int stage=0;stage<2;++stage){
        auto thr=typename M::TiledMmaQK_RS_Quar{}.get_slice(tid);
        auto src=thr.partition_A(s(_,_,stage));
        auto coord=thr.partition_A(logical);
        auto dst=make_fragment_like<float>(src);
        copy(Copy{},src,dst);
        for(int i=0;i<size(dst);++i)require(dst(i)==stage*64+int(get<0>(coord(i))));
    }
    using Aux=decltype(make_tiled_mma(SM80_16x8x8_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_2,_1>>{},Shape<_16,_16,_32>{}));
    auto acopy=make_tiled_copy_A(Copy{},Aux{});
    auto bcopy=make_tiled_copy_B(Copy{},Aux{});
    for(int stage=0;stage<2;++stage)for(int row_tile=0;row_tile<4;++row_tile)
    for(int col_tile=0;col_tile<4;++col_tile)for(int tid=0;tid<64;++tid){
        auto tile=local_tile(s(_,_,stage),Shape<_16,_32>{},make_coord(row_tile,col_tile));
        auto id=make_identity_tensor(Shape<_16,_32>{});
        auto a=acopy.get_slice(tid);
        auto b=bcopy.get_slice(tid);
        auto ma=Aux{}.get_slice(tid);
        auto ar=make_fragment_like<float>(ma.partition_A(tile));
        auto br=make_fragment_like<float>(ma.partition_B(tile));
        auto ar_view=a.retile_D(ar);auto br_view=b.retile_D(br);
        copy(acopy,a.partition_S(tile),ar_view);
        copy(bcopy,b.partition_S(tile),br_view);
        auto ai=ma.partition_A(id);auto bi=ma.partition_B(id);
        for(int i=0;i<size(ar);++i)require(ar(i)==stage*64+row_tile*16+int(get<0>(ai(i))));
        for(int i=0;i<size(br);++i)require(br(i)==stage*64+row_tile*16+int(get<0>(bi(i))));
    }
    std::vector<int> owners(128*128,0),outputs(64*128,0);
    auto state=make_identity_tensor(Shape<_128,_128>{}); // mathematical (V,K)
    auto output=make_identity_tensor(Shape<_128,_64>{}); // mathematical (V,T)
    for(int tid=0;tid<256;++tid){
        auto st=typename M::TiledMmaKV{}.get_slice(tid).partition_C(state);
        for(int i=0;i<size(st);++i){auto c=st(i); ++owners[int(get<1>(c))*128+int(get<0>(c))];}
        auto ot=typename M::TiledMmaO1{}.get_slice(tid).partition_C(output);
        for(int i=0;i<size(ot);++i){auto c=ot(i); ++outputs[int(get<1>(c))*128+int(get<0>(c))];}
    }
    for(int n:owners)require(n==1);
    for(int n:outputs)require(n==1);
    auto state_layout=gdn::sm90::state_layout<128,128>(3,2);
    for(int b=0;b<2;++b)for(int h=0;h<3;++h)for(int k=0;k<128;++k)for(int v=0;v<128;++v)
        require(state_layout(k,v,h,b)==((b*3+h)*128+k)*128+v);
    for(int tail=1;tail<=64;++tail){
        std::vector<int> written(2*64*128,0);
        for(int lane=0;lane<32;++lane)for(int i=lane;i<tail*128;i+=32)++written[i];
        for(int i=0;i<int(written.size());++i)require(written[i]==int(i<tail*128));
    }
    std::cout<<"[SM90 layouts] gate=16384 state=16384 output=8192 tails=64 exact-once PASS"
             <<" threads="<<Builder::Kernel::MaxThreadsPerBlock
             <<" shared="<<Builder::Kernel::SharedStorageSize<<"\n";
}
