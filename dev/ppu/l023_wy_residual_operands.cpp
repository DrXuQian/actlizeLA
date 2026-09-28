#include <array>
#include <vector>
#include <stdexcept>
#include <cstdio>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_operands.cuh"
using namespace gdn_qsa::wy;

template <int Steps> int check(bool wrong_slot=false) {
  std::array<int,2> slots{{-1,-1}};
  std::array<bool,2> live{{false,false}};
  std::vector<int> order, loads;
  int bad=0;
  residual_operands::prefetch_atoms<Steps>([&](auto atom,auto slot) {
    bad += live[int(slot)];
    slots[int(slot)]=int(atom); live[int(slot)]=true; loads.push_back(int(atom));
  },[&](auto atom,auto slot) {
    int const s=wrong_slot ? 1-int(slot):int(slot);
    bad += !live[s] || slots[s]!=int(atom);
    order.push_back(slots[s]); live[s]=false;
  });
  for (int k=0;k<Steps;++k) bad += order.at(k)!=k || loads.at(k)!=k;
  bad += live[0] || live[1] || order.size()!=Steps || loads.size()!=Steps;
  return bad;
}
int main() {
  if(check<4>() || check<8>()) throw std::runtime_error("operand register-slot lifetime/K-order mismatch");
  if(!check<4>(true) || !check<8>(true)) throw std::runtime_error("wrong register slot escaped");
  std::puts("[residual operands] actual-production-template steps4/8, K-order+slot-lifetimes PASS; wrong-slot EXPECTED-RED/PASS");
}
