// Hardware admission of the actual producer/reader pair, before full GDN.
#include "csrc/backends/ppu15/primitives.cuh"
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <cmath>
using namespace actlize_la::ppu15;
static void check(hggcError_t rc) {
  if (rc != hggcSuccess) throw std::runtime_error(hggcGetErrorString(rc));
}
template <class T> struct DeviceBuffer {
  T* ptr = nullptr;
  explicit DeviceBuffer(size_t n) { check(hggcMalloc(&ptr,n*sizeof(T))); }
  ~DeviceBuffer() { if(ptr) hggcFree(ptr); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
};

template <int Rows, int Cols, bool Manual>
__global__ void read_tiles(uint16_t const* input, uint16_t* physical, uint16_t* fragments, int valid) {
  using L = Tile<Rows,Cols>;
  __shared__ __align__(128) BF16 sm[Rows*Cols];
  int const lane = int(threadIdx.x);
  if constexpr (Manual) {
    for (int i = lane; i < Rows*Cols; i += 32)
      reinterpret_cast<uint16_t*>(sm)[L::offset(i/Cols,i%Cols)] = i/Cols < valid ? input[(i/Cols)*(Cols+16)+i%Cols] : 0;
    __syncthreads();
  } else {
    L::stage(sm,reinterpret_cast<BF16 const*>(input),Cols+16,valid);
    commit_wait();
  }
  for (int i = lane; i < Rows*Cols; i += 32)
    physical[i] = reinterpret_cast<uint16_t const*>(sm)[L::offset(i/Cols,i%Cols)];
  for (int tile = 0; tile < Rows*Cols/256; ++tile) {
    int const r = tile/(Cols/16)*16, c = tile%(Cols/16)*16;
    uint32_t regs[4][4];
    L::template load<true>(sm,r,c,regs[0]);
    L::template load<false>(sm,r,c,regs[1]);
    L::template load<true,true>(sm,r,c,regs[2]);
    L::template load<false,true>(sm,r,c,regs[3]);
    for (int role=0; role<4; ++role) for(int s=0; s<8; ++s)
      fragments[((tile*4+role)*32+lane)*8+s] = uint16_t(regs[role][s/2] >> (16*(s%2)));
  }
}
template <int Rows, int Cols> void admit_tiles() {
  std::vector<uint16_t> input(Rows*(Cols+16),0x7fc1), physical(Rows*Cols), fragments(Rows*Cols*4);
  for(int r=0;r<Rows;++r) for(int c=0;c<Cols;++c) input[r*(Cols+16)+c]=uint16_t(0x1000+r*Cols+c);
  DeviceBuffer<uint16_t> src(input.size()), dst(physical.size()), regs(fragments.size());
  check(hggcMemcpy(src.ptr,input.data(),input.size()*2,hggcMemcpyHostToDevice));
  for(int valid : {Rows,Rows-1,1}) for(int manual : {0,1}) {
    if(manual) read_tiles<Rows,Cols,true><<<1,32>>>(src.ptr,dst.ptr,regs.ptr,valid);
    else read_tiles<Rows,Cols,false><<<1,32>>>(src.ptr,dst.ptr,regs.ptr,valid);
    check(hggcGetLastError());
    check(hggcMemcpy(physical.data(),dst.ptr,physical.size()*2,hggcMemcpyDeviceToHost));
    check(hggcMemcpy(fragments.data(),regs.ptr,fragments.size()*2,hggcMemcpyDeviceToHost));
    for(int i=0;i<Rows*Cols;++i) {
      uint16_t const want=i/Cols<valid?input[(i/Cols)*(Cols+16)+i%Cols]:0;
      if(physical[i]!=want) throw std::runtime_error("AIU physical layout/stride/padz mismatch at "+std::to_string(i));
    }
    for(int tile=0;tile<Rows*Cols/256;++tile) for(int role=0;role<4;++role)
      for(int lane=0;lane<32;++lane) for(int s=0;s<8;++s) {
        // Independent native ABI formula: no call to the kernel's CuTe map.
        bool const a=role%2==0, trans=role>=2;
        int const rr=lane/4+(a?((s/2)%2):s/4)*8;
        int const cc=(lane%4)*2+s%2+(a?s/4:(s/2)%2)*8;
        int const r=tile/(Cols/16)*16+(trans?cc:rr), c=tile%(Cols/16)*16+(trans?rr:cc);
        uint16_t const want=r<valid?input[r*(Cols+16)+c]:0;
        size_t const at=((tile*4+role)*32+lane)*8+s;
        if(fragments[at]!=want) throw std::runtime_error("native A/B/trans fragment mismatch at "+std::to_string(at));
      }
  }
  std::printf("[ppu15 primitive] tile=%dx%d stride=%d AIU/manual x A/B/AT/BT x full/tail RAW-BIT/PASS\n",Rows,Cols,Cols+16);
}

__global__ void products(BF16 const* bf, float const* fp, float* result) {
  __shared__ __align__(128) BF16 sm[32*32];
  Tile<32,32>::stage(sm,bf,32,32);
  commit_wait();
  uint32_t a[4],b[4];
  Tile<32,32>::load<true>(sm,0,0,a);
  Tile<32,32>::load<false>(sm,16,0,b);
  float c[8]={};
  bf16_mma(c,a,b);
  CUTE_UNROLL
  for(int s=0;s<8;++s) {
    auto rc=result_coord(threadIdx.x,s);
    result[rc.row*16+rc.col]=c[s];
    c[s]=0;
  }
  tf32_product(c,fp,16,fp+256,16);
  CUTE_UNROLL
  for(int s=0;s<8;++s) {
    auto rc=result_coord(threadIdx.x,s);
    result[256+rc.row*16+rc.col]=c[s];
  }
}
void admit_products() {
  std::vector<BF16> bf(1024); std::vector<float> fp(512), got(512);
  for(int i=0;i<1024;++i) bf[i]=BF16(float((i*17+i/32*3)%13-6));
  for(int i=0;i<512;++i) fp[i]=float((i*7+i/16*3)%11-5);
  DeviceBuffer<BF16> b(bf.size()); DeviceBuffer<float> f(fp.size()), out(got.size());
  check(hggcMemcpy(b.ptr,bf.data(),bf.size()*2,hggcMemcpyHostToDevice));
  check(hggcMemcpy(f.ptr,fp.data(),fp.size()*4,hggcMemcpyHostToDevice));
  products<<<1,32>>>(b.ptr,f.ptr,out.ptr); check(hggcGetLastError());
  check(hggcMemcpy(got.data(),out.ptr,got.size()*4,hggcMemcpyDeviceToHost));
  for(int r=0;r<16;++r) for(int c=0;c<16;++c) {
    float a=0,bv=0;
    for(int k=0;k<16;++k) {a+=float(bf[r*32+k])*float(bf[(16+c)*32+k]); bv+=fp[r*16+k]*fp[256+k*16+c];}
    if(got[r*16+c]!=a || got[256+r*16+c]!=bv) throw std::runtime_error("BF16/TF32 exact product mismatch");
  }
  std::puts("[ppu15 primitive] BF16/TF32 integer-exact products RAW-BIT/PASS");
}
int main() {
  try {
    hggcDeviceProp prop{}; check(hggcGetDeviceProperties(&prop,0));
    if(std::string(prop.name).find("890")==std::string::npos)
      throw std::runtime_error("PPU1.5 admission requires ZW890, refusing another device");
    int driver=0,runtime=0;
    check(hggcDriverGetVersion(&driver)); check(hggcRuntimeGetVersion(&runtime));
    std::printf("[ppu15 device] name=%s cu=%d driver=%d runtime=%d\n",prop.name,prop.multiProcessorCount,driver,runtime);
    admit_tiles<64,128>(); admit_tiles<64,64>(); admit_tiles<64,32>();
    admit_tiles<32,128>(); admit_tiles<32,64>(); admit_products();
    std::puts("[ppu15 primitive] PASS"); return 0;
  } catch(std::exception const& e) {std::fprintf(stderr,"[ppu15 primitive] FAIL: %s\n",e.what()); return 1;}
}
