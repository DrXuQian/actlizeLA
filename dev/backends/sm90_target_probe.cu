#include "target.cuh"
#include <cuda_runtime.h>
__global__ void gdn_sm90_compiler_receipt(int* value) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x==0) *value=__CUDA_ARCH__;
#endif
}
