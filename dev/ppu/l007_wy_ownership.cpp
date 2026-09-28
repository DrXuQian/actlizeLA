#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/wy_contract.hpp"
#include "gdn_qsa/ppu/wy_mma.cuh"

using namespace cute;
using namespace gdn_qsa::wy;

template <int R, int C> int shared_map() {
  int bad = 0;
  std::vector<int> memory(R * C, -1);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) memory[swizzle<R, C>(r, c)] = r * C + c;
  // Independent hardware cube formula, not a second call to RowLayout.
  auto hw = [](int r, int c) {
    int const line = r / 4, vec = ((r % 4) * 2 + c / 8) ^ (line % 2);
    return (line * 32 + vec * 4 + (c % 8) / 2) * 2 + c % 2;
  };
  for (int br = 0; br < R; br += 16)
    for (int bc = 0; bc < C; bc += 16)
      for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
          int const base = swizzle<R, C>(br, bc);
          bad += memory[base + hw(r, c)] != (br + r) * C + bc + c;
          bad += memory[base + hw(c, r)] != (br + c) * C + bc + r;
        }
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; c += 8)
      for (int v = 0; v < 8; ++v) bad += swizzle<R, C>(r, c + v) != swizzle<R, C>(r, c) + v;
  return bad;
}

int check(bool wrong_slice, bool wrong_tf32, bool missing_chunk, bool wrong_register = false) {
  int bad = 0;
  using T = MMA_Traits<PPU0010_16x16x8_F32TF32TF32F32_TN>;
  for (int lane = 0; lane < 32; ++lane)
    for (int s = 0; s < 4; ++s) {
      auto const rc = tf32_coord(lane, s);
      int const r = rc.row, c = rc.col ^ (wrong_tf32 ? 1 : 0);
      int const actual = T::ALayout{}(lane, s);
      bad += actual != r + c * 16;
    }
  using Bf = decltype(make_tiled_mma(MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{}));
  for (unsigned lane = 0; lane < 32; ++lane)
    for (int slot = 0; slot < 8; ++slot) {
      auto owner = state_b_owner(lane, slot);
      if (wrong_register) owner.slot ^= 1;
      auto from = Bf{}.get_slice(owner.lane).partition_C(make_identity_tensor(cute::Shape<_16, _16>{}));
      auto to = Bf{}.get_slice(lane).partition_B(make_identity_tensor(cute::Shape<_16, _16>{}));
      auto rc = from(owner.slot);
      auto expected = to(slot);
      bad += get<0>(rc) != get<1>(expected) || get<1>(rc) != get<0>(expected);
    }
  std::vector<int> owners(Dim * Dim, 0);
  for (int slice = 0; slice < 4; ++slice)
    for (int warp = 0; warp < StateThreads / 32; ++warp)
      for (int lane = 0; lane < 32; ++lane) {
        auto coords = Bf{}.get_slice(lane).partition_C(make_identity_tensor(cute::Shape<_16, _16>{}));
        for (int tile = 0; tile < 8; ++tile)
          for (int s = 0; s < 8; ++s) {
            auto rc = coords(s);
            int const row = state_k_start(tile) + int(get<0>(rc));
            int const col = state_v_start(wrong_slice ? 0 : slice, warp) + int(get<1>(rc));
            ++owners[row * Dim + col];
          }
      }
  for (int x : owners) bad += x != 1;
  int64_t cells = 0;
  for (int length : {1, 16, 63, 64, 65, 129, 2048})
    for (int b : {1, 2})
      for (int hv : {2, 32, 64}) {
        gdn_qsa::wy::Shape shape{b, length, hv / 2, hv};
        std::vector<int> written(b * hv * length, 0);
        std::vector<int> groups(shape.groups(), 0);
        for (int batch = 0; batch < b; ++batch)
          for (int h = 0; h < hv; ++h)
            for (int ct = 0; ct < shape.chunks() - int(missing_chunk); ++ct) {
              ++groups[shape.group(batch, h, ct)];
              for (int r = 0; r < Chunk && ct * Chunk + r < length; ++r) {
                ++written[(batch * hv + h) * length + ct * Chunk + r];
                int const token = ct * Chunk + r;
                int64_t const value_offset = ((int64_t(batch) * length + token) * hv + h) * 128;
                int64_t const key_offset = ((int64_t(batch) * length + token) * (hv / 2) + h / 2) * 128;
                bad += shape.input(batch, token, h, hv) != value_offset;
                bad += shape.input(batch, token, shape.q_head(h), hv / 2) != key_offset;
                ++cells;
              }
            }
        for (int x : groups) bad += x != 1;
        for (int x : written) bad += x != 1;
      }
  if (!wrong_slice && !wrong_tf32 && !missing_chunk && !wrong_register)
    std::printf("[WY ownership] state=%zu cells=%lld independent=actlize-traits bad=%d\n",
                owners.size(), (long long)cells, bad);
  return bad;
}
int main() {
  if (shared_map<64, 128>() + shared_map<64, 64>() +
      shared_map<128, 128>() + shared_map<64, 32>()) return 1;
  if (check(false, false, false)) return 1;
  for (int n = 0; n < 4; ++n) {
    int bad = check(n == 0, n == 1, n == 2, n == 3);
    if (!bad) throw std::runtime_error("WY negative escaped");
    std::printf("[WY ownership negative] plant=%d bad=%d EXPECTED-RED/PASS\n", n, bad);
  }
  std::puts("[WY ownership] PASS device_execution=NOT_RUN");
}
