#pragma once

// PPU1.0 prefill transport policy. Other targets and incremental modes retain
// their original schedule. Neither policy reads device-side request metadata.

namespace actlize_la::serving::prefill_schedule {
constexpr int TokensPerBlock = 32;
constexpr int PrepareThreads = 128;
constexpr int PrepareWarps = PrepareThreads / 32;

constexpr bool direct_pack(int requests, int width, int hk, int hv) {
  return requests == 1 && hk == 16 && (hv == 32 || hv == 48 || hv == 64) &&
         (width == 2048 || width == 8192);
}

#if defined(ALA_PPU_TARGET) || defined(__CUDACC__)
#define ALA_PREFILL_HD __host__ __device__
#else
#define ALA_PREFILL_HD
#endif
ALA_PREFILL_HD constexpr int first(int partition, int lanes, int lane) {
  return partition * lanes + lane;
}
ALA_PREFILL_HD constexpr int stride(int partitions, int lanes) {
  return partitions * lanes;
}
#undef ALA_PREFILL_HD

// grid.z is limited to 65535. For unusually large capacities, each block
// strides over further token tiles; launch geometry never reads device data.
constexpr int blocks(int width) {
  int const count = 1 + (width - 1) / TokensPerBlock;
  return count < 65535 ? count : 65535;
}
} // namespace actlize_la::serving::prefill_schedule
