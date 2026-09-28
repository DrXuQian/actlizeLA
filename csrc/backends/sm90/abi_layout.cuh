#pragma once
#include <cute/tensor.hpp>

namespace gdn::sm90 {
// Public contiguous [B,Hv,K,V]. cuLA's register accumulator is (V,K),
// which is a view permutation, not a transpose of the public allocation.
template <int K,int V>
CUTE_HOST_DEVICE constexpr auto state_layout(int heads,int batch) {
    using namespace cute;
    return make_layout(make_shape(Int<K>{},Int<V>{},heads,batch),
        make_stride(Int<V>{},_1{},K*V,int64_t(K)*V*heads));
}
} // namespace gdn::sm90
