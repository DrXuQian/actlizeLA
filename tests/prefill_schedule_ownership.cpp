// Execute the production index helpers; independent denominator is tensor size.
#include "prefill_schedule.h"
#include <cstdint>
#include <vector>
using namespace actlize_la::serving::prefill_schedule;

bool exact_once(int extent, int partitions, int lanes) {
  std::vector<int> seen(extent, 0);
  for (int z = 0; z < partitions; ++z)
    for (int lane = 0; lane < lanes; ++lane)
      for (int64_t i = first(z, lanes, lane); i < extent;
           i += stride(partitions, lanes)) {
        if (i < 0 || ++seen[i] != 1)
          return false;
      }
  for (int count : seen)
    if (count != 1)
      return false;
  return true;
}

int main() {
  static_assert(PrepareThreads == 128 && PrepareWarps == 4);
  static_assert(blocks(1) == 1 && blocks(32) == 1 && blocks(33) == 2);
  static_assert(blocks(2048) == 64 && blocks(8192) == 256);
  static_assert(blocks(2147483647 / 128) == 65535);
  for (int width : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 8192}) {
    int z = blocks(width);
    if (!exact_once(width, z, PrepareWarps) ||
        !exact_once(width * 128, z, 128) || !exact_once(16384, z, 256))
      return 1;
    for (int live : {0, 1, width / 2, width - 1, width})
      if (!exact_once(live * 128, z, 256))
        return 2;
  }
  return 0;
}
