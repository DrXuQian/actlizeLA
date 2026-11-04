// State transport only. ShortConv arithmetic stays in Flare/b12x.
#include "abi.h"
#include "runtime.cuh"
namespace actlize_la::serving {
template <bool Restore> __global__ void validate_conv(ConvHistory p) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= p.requests)
    return;
  if constexpr (!Restore)
    p.recorded_slots[r] = 0;
  int error = Ok;
  if (*p.live < 0 || *p.live > p.requests)
    error = Bounds;
  else if (r < *p.live) {
    auto slot = p.slots[int64_t(r) * p.slot_stride];
    if (slot >= p.state_slots)
      error = Slot;
    if constexpr (Restore) {
      if (slot > 0 && (p.counts[r] < 0 || p.counts[r] > p.columns - 2 ||
                       slot != p.recorded_slots[r]))
        error = Prefix;
      for (int j = 0; j < *p.live; ++j)
        if (j != r && slot > 0 && slot == p.slots[int64_t(j) * p.slot_stride])
          error = Conflict;
    } else {
      p.recorded_slots[r] = error == Ok && slot > 0 ? slot : 0;
    }
  } else if constexpr (!Restore)
    p.recorded_slots[r] = 0;
  p.errors[r] = error;
}
template <bool Restore> __global__ void transfer_conv(ConvHistory p) {
  int const r = blockIdx.x;
  if (r >= *p.live || p.errors[r])
    return;
  auto const slot = p.slots[int64_t(r) * p.slot_stride];
  if (slot <= 0)
    return;
  if constexpr (Restore)
    if (p.counts[r] != 0)
      return;
  auto *state = static_cast<BFloat *>(p.state) + slot * p.state_stride;
  auto *copy =
      static_cast<BFloat *>(p.snapshot) + int64_t(r) * p.channels * p.columns;
  for (int64_t i = blockIdx.y * blockDim.x + threadIdx.x;
       i < int64_t(p.channels) * p.columns; i += gridDim.y * blockDim.x) {
    if constexpr (Restore)
      state[i] = copy[i];
    else
      copy[i] = state[i];
  }
}
template <bool Restore> int conv_launch(ConvHistory const *p, void *stream) {
  if (!p || !p->state || !p->snapshot || !p->slots || !p->recorded_slots ||
      !p->live || !p->counts || !p->errors || p->requests < 1 ||
      p->state_slots < 2 || p->channels < 1 || p->columns < 3 ||
      p->columns > 10 || p->state_stride < int64_t(p->channels) * p->columns ||
      p->slot_stride < 1)
    return -1;
  auto s = static_cast<Stream>(stream);
  validate_conv<Restore><<<(p->requests + 127) / 128, 128, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
  transfer_conv<Restore><<<dim3(p->requests, 16), 128, 0, s>>>(*p);
  return last_error();
}
} // namespace actlize_la::serving
extern "C" int ala_conv_capture(actlize_la::serving::ConvHistory const *p,
                                void *stream) {
  return actlize_la::serving::conv_launch<false>(p, stream);
}
extern "C" int ala_conv_rollback_zero(actlize_la::serving::ConvHistory const *p,
                                      void *stream) {
  return actlize_la::serving::conv_launch<true>(p, stream);
}
