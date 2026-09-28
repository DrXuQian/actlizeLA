#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cutlass/bfloat16.h>
#include "gdn_qsa/ppu/wy_prepare_rows.cuh"

using namespace gdn_qsa::wy;
using BF16 = cutlass::bfloat16_t;
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
float product(float a, float b) { volatile float x = a * b; return x; }

struct Counts { uint64_t chunks=0, shared=0, warp=0, consumers=0; };

uint64_t reuse(Counts& count, unsigned plant=0) {
  uint64_t bad=0;
  for (unsigned fixture=0; fixture<8; ++fixture) for (unsigned valid=1; valid<=64; ++valid) {
    // Already-produced FP32 prefix, not a replacement model of the scan.
    float prefix[64], sum=0, shared[64], warp[4][2][32];
    std::array<unsigned,64> shared_owners{};
    std::array<unsigned,256> warp_owners{};
    for (unsigned r=0; r<64; ++r) {
      if (r<valid) sum -= float((r*17+fixture*11)%31+1)/64.0f;
      prefix[r]=sum;
    }
    for (unsigned tid=0; tid<128; ++tid) {
      if (PrepareRowsPlan::shared_writer(tid)) {
        unsigned const row=plant==1 ? (tid+1)%64 : tid;
        shared[tid]=std::exp(prefix[row]);
        ++shared_owners[tid]; ++count.shared;
      }
      for (unsigned slot=0; slot<2; ++slot) {
        unsigned const row=PrepareRowsPlan::producer_row(tid%32,slot);
        bad += row != tid%32+slot*32;
        warp[tid/32][slot][tid%32]=std::exp(prefix[row]);
        ++warp_owners.at(tid/32*64+row); ++count.warp;
      }
    }
    for (unsigned n:shared_owners) bad += n!=1;
    for (unsigned n:warp_owners) bad += n!=1;
    // A full-mask shuffle requires every lane to execute every iteration.
    unsigned const participants=plant==3 ? 0x7fffffffu : PrepareRowsPlan::FullMask;
    bool const cache_ready=plant!=4;  // publication must precede consumption
    for (unsigned tid=0; tid<128; ++tid) for (unsigned row=0; row<64; ++row) {
      float const control=std::exp(prefix[row]);
      unsigned const lane=PrepareRowsPlan::lane(row);
      unsigned const slot=PrepareRowsPlan::slot(row) ^ unsigned(plant==2);
      bad += participants!=0xffffffffu || !(participants & (1u << (tid%32)));
      bad += !cache_ready;
      float const k=float(BF16(float(int((tid*37+row*13+fixture*71)%2047)-1023)/128.0f));
      float const beta=float(BF16(float((row*43+fixture*17)%127+1)/128.0f));
      float const expected=product(product(k,beta),control);
      for (float factor : {shared[row], warp[tid/32][slot][lane]}) {
        if (plant==9) factor=float(BF16(factor));
        bad += bits(control)!=bits(factor);
        float const actual=plant==5 ? product(k,product(beta,factor)) : product(product(k,beta),factor);
        bad += bits(expected)!=bits(actual) || BF16(expected).raw()!=BF16(actual).raw();
        ++count.consumers;
      }
    }
    ++count.chunks;
  }
  return bad;
}

uint64_t selectors(unsigned plant=0) {
  std::array<bool,32768> expected{};
  for (unsigned p : {0u,1u,8u,256u,1280u,2304u})
    for (unsigned s : {0u,2u,16u,80u,144u,208u})
      for (unsigned o : {0u,4u,32u,544u}) expected[p|s|o]=true;
  for (unsigned mask : {5616u,9712u,13808u,30192u}) expected[mask]=true;
  unsigned bad=0,valid=0,cases=0,dispatches=0;
  for (unsigned mask=0; mask<32768; ++mask) {
    if (plant==8 && mask==2048) continue;
    bool const actual=plant==7 && mask==3328 ? true : valid_delivery(mask);
    bad += actual!=expected[mask]; valid+=actual; ++cases;
    unsigned const selected=plant==6 ? mask & ~PrepareRowsOptions : mask;
    int calls=0;
    int const mode=visit_prepare_rows(selected,[&](auto tag) { ++calls; return decltype(tag)::value; },-1);
    int const want=mask==1024 ? 1 : mask==2048 ? 2 : -1;
    bad += mode!=want || calls!=int(want!=-1); dispatches+=calls;
  }
  return bad+(cases!=32768)+(valid!=148)+(dispatches!=2);
}

int main() {
  Counts count;
  if (reuse(count) || selectors()) return 1;
  if (count.chunks!=512 || count.shared!=32768 || count.warp!=131072 || count.consumers!=8388608)
    throw std::runtime_error("prepare row-cache denominator differs");
  for (unsigned plant=1; plant<=9; ++plant) {
    Counts scratch;
    auto const bad=plant>=6 && plant<=8 ? selectors(plant) : reuse(scratch,plant);
    if (!bad) throw std::runtime_error("prepare row-cache negative escaped");
    std::printf("[WY prepare-rows negative] plant=%u bad=%llu EXPECTED-RED/PASS\n",plant,
                static_cast<unsigned long long>(bad));
  }
  std::printf("[WY prepare-rows] chunks=%llu shared_producers=%llu warp_producers=%llu consumers=%llu selectors=32768/148 typed-dispatch=32768/2 FP32-factor+product/BF16-bits=PASS device_execution=NOT_RUN\n",
      static_cast<unsigned long long>(count.chunks),static_cast<unsigned long long>(count.shared),
      static_cast<unsigned long long>(count.warp),static_cast<unsigned long long>(count.consumers));
}
