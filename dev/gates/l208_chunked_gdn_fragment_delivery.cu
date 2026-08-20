// L208 -- exhaust the logical-coordinate -> compact physical-register map
// used by the shipping BF16 resident MMA and both TF32 inverse MMAs.
//
// Actual comes from the production helper plus its real fragment layout.
// Expected is derived independently from the public PPU0010 atom/register ABI
// and the frozen warp topology.  This distinction matters: coordinate tensors
// carry basis strides while register fragments deliberately use a compact
// make_fragment_like layout, so their layout types must not be equal.

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh"

#if !defined(__HGGCCC__)
#error "L208 must instantiate the production __HGGCCC__ MMA helpers"
#endif

#if !defined(__CUDA_ARCH__)
namespace {

using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Args = cutlass::linear_attention::PpuChunkedGdnArguments<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float>;
using Collective =
    cutlass::linear_attention::PpuChunkedGdnCollectiveBf16C64D128BV64<
        Traits, Args>;

constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

struct Coordinate {
  int row = -1;
  int column = -1;
};

bool same(Coordinate a, Coordinate b) {
  return a.row == b.row && a.column == b.column;
}

void hash_u32(std::uint64_t& hash, std::uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) {
    hash ^= (value >> (8 * byte)) & 0xffu;
    hash *= kFnvPrime;
  }
}

void hash_coordinate(
    std::uint64_t& hash, int thread, int storage, Coordinate coordinate) {
  hash_u32(hash, std::uint32_t(thread));
  hash_u32(hash, std::uint32_t(storage));
  hash_u32(hash, std::uint32_t(coordinate.row));
  hash_u32(hash, std::uint32_t(coordinate.column));
}

enum class MapKind { kBf16C64, kTf32M16, kTf32M32 };

template <MapKind Kind>
int canonical_slot_from_storage(int storage) {
  if constexpr (Kind == MapKind::kBf16C64) {
    int const value = storage % 8;
    int const rest_m = storage / 32;
    int const rest_k = (storage % 32) / 8;
    return value + 8 * rest_m + 16 * rest_k;
  } else {
    return storage;
  }
}

template <MapKind Kind>
Coordinate expected_operand(bool is_b, int thread, int canonical_slot) {
  int const lane = thread % 32;
  int const warp = thread / 32;
  if constexpr (Kind == MapKind::kBf16C64) {
    int const warp_axis = is_b ? warp / 2 : warp % 2;
    int const value = canonical_slot % 16;
    return {
        lane / 4 + 16 * warp_axis + 8 * ((value / 4) % 2) +
            32 * (value / 8),
        2 * (lane % 4) + (value % 2) + 8 * ((value / 2) % 2) +
            16 * (canonical_slot / 16)};
  } else {
    // Explicit TF32 topology is Layout<Shape<2,2>,Stride<2,1>>:
    // physical warp id is therefore 2*warp_m + warp_n.
    int const warp_m = Kind == MapKind::kTf32M16 ? 0 : warp / 2;
    int const warp_n = Kind == MapKind::kTf32M16 ? 0 : warp % 2;
    int const warp_axis = is_b ? warp_n : warp_m;
    int const value = canonical_slot % 4;
    return {
        lane / 4 + 16 * warp_axis + 8 * (value / 2),
        lane % 4 + 4 * (value % 2) + 8 * (canonical_slot / 4)};
  }
}

template <MapKind Kind>
Coordinate expected_output(int thread, int storage) {
  int const lane = thread % 32;
  int const warp = thread / 32;
  if constexpr (Kind == MapKind::kBf16C64) {
    int const warp_m = warp % 2;
    int const warp_n = warp / 2;
    int const value = storage % 16;
    return {
        lane / 4 + 16 * warp_m + 8 * ((value / 4) % 2) +
            32 * (value / 8),
        lane % 4 + 16 * warp_n + 4 * (value % 4) +
            32 * (storage / 16)};
  }
  int warp_m = 0;
  int warp_n = 0;
  if constexpr (Kind == MapKind::kTf32M32) {
    warp_m = warp / 2;
    warp_n = warp % 2;
  }
  int const value = storage % 8;
  return {
      lane / 4 + 16 * warp_m + 8 * (value / 4),
      lane % 4 + 16 * warp_n + 4 * (value % 4)};
}

struct MapResult {
  int visits = 0;
  int holes = 0;
  int aliases = 0;
  int mismatches = 0;
  std::uint64_t actual_hash = kFnvOffset;
  std::uint64_t expected_hash = kFnvOffset;
};

template <class Helper, MapKind Kind, int Threads, int Slots, bool IsB>
MapResult enumerate_operand() {
  using Fragment = std::conditional_t<IsB, typename Helper::FragmentB,
                                      typename Helper::FragmentA>;
  typename Fragment::layout_type physical_layout{};
  MapResult result{};
  for (int thread = 0; thread < Threads; ++thread) {
    std::array<Coordinate, Slots> by_storage{};
    std::array<int, Slots> owners{};
    auto visit = [&](int canonical_slot, int row, int reduction) {
      int storage = int(physical_layout(cute::idx2crd(
          canonical_slot, cute::shape(physical_layout))));
#if defined(L208_PLANT_PHYSICAL_SLOT_ROTATE)
      storage = (storage + 1) % Slots;
#endif
      Coordinate actual{row, reduction};
#if defined(L208_PLANT_B_TRANSPOSE)
      if constexpr (IsB) actual = {reduction, row};
#endif
      if (storage >= 0 && storage < Slots) {
        by_storage[std::size_t(storage)] = actual;
        ++owners[std::size_t(storage)];
      }
      ++result.visits;
    };
    if constexpr (IsB) {
      Helper::for_each_b_coordinate(thread, visit);
    } else {
      Helper::for_each_a_coordinate(thread, visit);
    }

    for (int storage = 0; storage < Slots; ++storage) {
      result.holes += owners[std::size_t(storage)] == 0;
      result.aliases += owners[std::size_t(storage)] > 1
                            ? owners[std::size_t(storage)] - 1
                            : 0;
      int const canonical = canonical_slot_from_storage<Kind>(storage);
      Coordinate const expected =
          expected_operand<Kind>(IsB, thread, canonical);
      Coordinate const actual = by_storage[std::size_t(storage)];
      result.mismatches += !same(actual, expected);
      hash_coordinate(result.actual_hash, thread, storage, actual);
      hash_coordinate(result.expected_hash, thread, storage, expected);
    }
  }
  return result;
}

template <class Helper, MapKind Kind, int Threads, int Slots>
MapResult enumerate_output() {
  typename Helper::Accumulator::layout_type physical_layout{};
  MapResult result{};
  for (int thread = 0; thread < Threads; ++thread) {
    std::array<Coordinate, Slots> by_storage{};
    std::array<int, Slots> owners{};
    Helper::for_each_c_coordinate(
        thread, [&](int canonical_slot, int row, int column) {
          int storage = int(physical_layout(cute::idx2crd(
              canonical_slot, cute::shape(physical_layout))));
          if (storage >= 0 && storage < Slots) {
            by_storage[std::size_t(storage)] = {row, column};
            ++owners[std::size_t(storage)];
          }
          ++result.visits;
        });
    for (int storage = 0; storage < Slots; ++storage) {
      result.holes += owners[std::size_t(storage)] == 0;
      result.aliases += owners[std::size_t(storage)] > 1
                            ? owners[std::size_t(storage)] - 1
                            : 0;
      Coordinate const expected = expected_output<Kind>(thread, storage);
      Coordinate const actual = by_storage[std::size_t(storage)];
      result.mismatches += !same(actual, expected);
      hash_coordinate(result.actual_hash, thread, storage, actual);
      hash_coordinate(result.expected_hash, thread, storage, expected);
    }
  }
  return result;
}

bool pass(MapResult const& x, int visits, std::uint64_t frozen_hash) {
  return x.visits == visits && x.holes == 0 && x.aliases == 0 &&
         x.mismatches == 0 && x.actual_hash == x.expected_hash &&
         x.actual_hash == frozen_hash;
}

void print(char const* name, MapResult const& x) {
  std::printf("%s=(visits:%d,holes:%d,aliases:%d,bad:%d,map:%016llx,anchor:%016llx)",
              name, x.visits, x.holes, x.aliases, x.mismatches,
              static_cast<unsigned long long>(x.actual_hash),
              static_cast<unsigned long long>(x.expected_hash));
}

template <class Helper, MapKind Kind, int Threads, int OperandSlots,
          int AccumulatorSlots>
bool run(char const* name, int operand_visits, int output_visits,
         std::uint64_t a_hash, std::uint64_t b_hash, std::uint64_t c_hash) {
  MapResult const a =
      enumerate_operand<Helper, Kind, Threads, OperandSlots, false>();
  MapResult const b =
      enumerate_operand<Helper, Kind, Threads, OperandSlots, true>();
  MapResult const c =
      enumerate_output<Helper, Kind, Threads, AccumulatorSlots>();
  bool const ok = pass(a, operand_visits, a_hash) &&
                  pass(b, operand_visits, b_hash) &&
                  pass(c, output_visits, c_hash);
  std::printf(" %s[%s] ", name, ok ? "PASS" : "FAIL");
  print("A", a);
  std::printf(" ");
  print("B", b);
  std::printf(" ");
  print("C", c);
  return ok;
}

}  // namespace

int main() {
  bool const bf16 = run<typename Collective::ResidentMma, MapKind::kBf16C64,
                        128, 64, 32>(
      "bf16-64", 8192, 4096, UINT64_C(0x6926c37c01088f25),
      UINT64_C(0x1d4e7cc9f73f9b25), UINT64_C(0xee9011938bb9c325));
  bool const tf16 = run<typename Collective::InverseMma16, MapKind::kTf32M16,
                        32, 8, 8>(
      "tf32-16", 256, 256, UINT64_C(0x0da6524437ae9b25),
      UINT64_C(0x0da6524437ae9b25), UINT64_C(0xf17712ae1e769b25));
  bool const tf32 = run<typename Collective::InverseMma32, MapKind::kTf32M32,
                        128, 16, 8>(
      "tf32-32", 2048, 1024, UINT64_C(0x1b925a1ff30f7f25),
      UINT64_C(0xbdd31935d0318f25), UINT64_C(0x7a553edd9c2a1525));
  bool const ok = bf16 && tf16 && tf32;
  std::printf("\n[l208] %s: actual=shipping-helper+compact-fragment expected=public-PPU0010-atom+warp-topology\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
#endif
