#include "scalar_gdn_state.cuh"
#include "kda/sm90/kernel/builder_kda_fwd.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace cute;
using BF16=cutlass::bfloat16_t;
using Mma=SM80_16x8x16_F32F16F16F32_TN;
void need(bool p){if(!p)throw std::runtime_error("inverse row/K-half ownership");}
void check(int warps,bool wrong_half){
    auto first=make_tiled_mma(Mma{},Layout<Shape<_1,_1>>{},Shape<_16,_32,_32>{});
    auto second=make_tiled_mma(Mma{},Layout<Shape<_1,_1>>{},Shape<_16,_32,_16>{});
    std::array<int,1024> dc_owners{},output_owners{};
    auto whole=make_identity_tensor(Shape<_64,_64>{});
    for(int y=0;y<warps;++y){
        auto tiles=gdn::sm90::inverse64_row_tiles(whole,y);
        auto d=get<0>(tiles);
        auto c=get<1>(tiles);
        auto a=get<2>(tiles);
        auto o=get<3>(tiles);
        for(int m=0;m<16;++m)for(int k=0;k<32;++k){
            auto [r,col]=d(m,k);need(r==32+16*y+m && col==32+k);
        }
        for(int n=0;n<32;++n)for(int k=0;k<32;++k){
            auto [cr,cc]=c(n,k);auto [ar,ac]=a(n,k);
            need(cr==32+k && cc==n && ar==k && ac==n);
        }
        for(int lane=0;lane<32;++lane){
            auto dc=first.get_slice(lane).partition_C(make_identity_tensor(Shape<_16,_32>{}));
            auto out=second.get_slice(lane).partition_C(o);
            for(int i=0;i<size(dc);++i){auto [m,n]=dc(i);++dc_owners[(16*y+int(m))*32+int(n)];}
            for(int i=0;i<size(out);++i){auto [m,n]=out(i);need(m>=32 && m<64 && n>=0 && n<32);
                ++output_owners[(int(m)-32)*32+int(n)];}
            auto left=second.get_slice(lane).partition_A(make_identity_tensor(Shape<_16,_32>{}));
            auto right=second.get_slice(lane).partition_B(a);
            need(size<2>(left)==2 && size<2>(right)==2);
            for(int x=0;x<2;++x){
                auto lp=left(_,_,wrong_half?1-x:x);auto rp=right(_,_,x);
                for(int i=0;i<size(lp);++i){auto [m,k]=lp(i);need(k>=16*x && k<16*(x+1));}
                for(int i=0;i<size(rp);++i){auto [k,n]=rp(i);need(k>=16*x && k<16*(x+1));}
            }
        }
    }
    for(int n:dc_owners)need(n==1);for(int n:output_owners)need(n==1);
}
template<class F>void red(F f){try{f();}catch(std::runtime_error const&){return;}
    throw std::runtime_error("inverse negative escaped");}
int main(){
    check(2,false);red([]{check(1,false);});red([]{check(2,true);});
    // Constructible FP16-input dot: 1*1 + 2^-12*1, and -1*1.
    float lo=1.0f+1.0f/4096.0f,hi=-1.0f;
    cutlass::half_t separate=cutlass::half_t(hi)+cutlass::half_t(lo);
    cutlass::half_t contracted=cutlass::half_t(hi+lo);
    need(float(separate)==0.0f && float(contracted)!=0.0f);
    std::cout<<"[SM90 actual map] DC1024/output1024 exact-once; K16 partials; omittedwarp/wronghalf/contracted-rounding negatives PASS\n";
}
