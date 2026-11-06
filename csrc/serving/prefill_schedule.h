#pragma once

// Scheduling only: no arithmetic or public buffer-layout changes. Keep the
// admitted serial path until the opt-in candidate passes PPU device gates.
#ifndef ALA_PREFILL_TOKEN_PARALLEL
#define ALA_PREFILL_TOKEN_PARALLEL 0
#endif
#if ALA_PREFILL_TOKEN_PARALLEL != 0 && ALA_PREFILL_TOKEN_PARALLEL != 1
#error "ALA_PREFILL_TOKEN_PARALLEL must be 0 or 1"
#endif

namespace actlize_la::serving::prefill_schedule {
constexpr int TokensPerBlock = 32;
constexpr int PrepareThreads = 128;
constexpr int PrepareWarps = PrepareThreads / 32;

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
