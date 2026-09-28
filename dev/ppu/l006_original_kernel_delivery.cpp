#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

// actlize's unused device simulator names threadIdx while being parsed by a
// host compiler. No device instruction is executed by this coordinate test.
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/shared_copy.cuh"

namespace {
using namespace cute;
using namespace gdn_qsa::ppu;
using Mma = decltype(make_tiled_mma(
    MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>{},
    Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{}));

// Independent physical address from actlize's hardware-facing
// ppu_tsm_ld_swzl_sim(SWAP=true), CUBE_H=CUBE_W=16, coord=(0,0).
int hardware_half_address(int row, int col, bool omit_swizzle = false) {
    int const line = row / 4;
    int const vector = ((row % 4) * 2 + col / 8) ^
                       (omit_swizzle ? 0 : line % 2);
    return (line * 32 + vector * 4 + (col % 8) / 2) * 2 + col % 2;
}

template <int R, int C, bool Trans>
int shared_tiles(bool plant_swizzle, bool plant_transpose) {
    using Physical = RowLayout<R, C>;
    std::vector<int> memory(R * C, -1);
    auto row = make_tensor(memory.data(), Physical{});
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) row(r, c) = r * C + c;
    auto logical = [&]() {
        if constexpr (Trans) return make_tensor(memory.data(), ColumnLayout<C, R>{});
        else return row;
    }();
    int bad = 0;
    for (int br = 0; br < size<0>(logical) / 16; ++br) {
        for (int bc = 0; bc < size<1>(logical) / 16; ++bc) {
            auto tile = local_tile(logical, Shape<_16, _16>{}, make_coord(br, bc));
            int const base = int(&tile(0, 0) - memory.data());
            for (int lane = 0; lane < 32; ++lane) {
                auto identity = make_identity_tensor(Shape<_16, _16>{});
                auto coordinates = Mma{}.get_slice(lane).partition_A(identity);
                for (int slot = 0; slot < 8; ++slot) {
                    auto rc = coordinates(slot);
                    int const r = get<0>(rc), c = get<1>(rc);
                    auto map = operand_coord(lane, slot);
                    bad += map.row != r || map.col != c;
                    int read_r = r, read_c = c;
                    if constexpr (Trans) {
                        if (!plant_transpose) { read_r = c; read_c = r; }
                    }
                    int const address = base + hardware_half_address(read_r, read_c, plant_swizzle);
                    int const want = Trans ? (bc * 16 + c) * C + br * 16 + r
                                           : (br * 16 + r) * C + bc * 16 + c;
                    bad += memory[std::size_t(address)] != want;
                }
            }
        }
    }
    // The original vector loads/stores and cp.async require 8 contiguous
    // elements, including inside the tile-to-shape extension.
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; c += 8) {
            for (int v = 0; v < 8; ++v)
                bad += Physical{}(r, c + v) != Physical{}(r, c) + v;
        }
    }
    return bad;
}

template <FragmentMap From, FragmentMap To, bool Trans>
int remap_check(bool select_before_shuffle) {
    auto coords = [](FragmentMap map, int lane, int slot) {
        auto identity = make_identity_tensor(Shape<_16, _16>{});
        if (map == FragmentMap::Operand) {
            auto c = Mma{}.get_slice(lane).partition_A(identity)(slot);
            return Coordinate{int(get<0>(c)), int(get<1>(c))};
        }
        auto c = Mma{}.get_slice(lane).partition_C(identity)(slot);
        return Coordinate{int(get<0>(c)), int(get<1>(c))};
    };
    int bad = 0;
    for (int lane = 0; lane < 32; ++lane) {
        for (int slot = 0; slot < 8; ++slot) {
            auto owner = remap_owner<From, To, Trans>(lane, slot);
            if (select_before_shuffle)
                owner.slot = remap_owner<From, To, Trans>(owner.lane, slot).slot;
            auto got = coords(From, owner.lane, owner.slot);
            auto want = coords(To, lane, slot);
            if constexpr (Trans) { int tmp = want.row; want.row = want.col; want.col = tmp; }
            bad += got.row != want.row || got.col != want.col;
        }
    }
    return bad;
}

// CuTe retile returns a non-owning register VIEW. Production callers bind it
// with `auto`, not `auto&`; returning the owning Tensor by reference copies
// its ArrayEngine and disconnects every later load from the MMA operand.
struct RetileResult { int cells = 0, detached_d = 0, stale_s = 0; };

template <CopyRole Role, class Fragment>
RetileResult retile_alias(Fragment fragment, int lane, bool legacy) {
    using Element = typename Fragment::value_type;
    auto thread = SharedCopy<Role>{}.get_slice(lane);
    clear(fragment);
    auto legacy_d = [](auto& t) -> auto& { return t; };
    auto legacy_s = [](auto const& t) -> auto const& { return t; };
    RetileResult result;
    auto exercise = [&](auto dst_view, auto src_view) {
        // Both arguments are intentionally copied, just like the call sites.
        for (int s = 0; s < size(fragment); ++s) {
            ++result.cells;
            auto const value = Element(lane * 8 + s + 1);
            dst_view(s) = value;
            result.detached_d += &dst_view(s) != &fragment(s) || fragment(s) != value;
            fragment(s) = Element(s + 16);
            result.stale_s += &src_view(s) != &fragment(s) || src_view(s) != fragment(s);
        }
    };
    if (legacy) exercise(legacy_d(fragment), legacy_s(fragment));
    else exercise(thread.retile_D(fragment), thread.retile_S(fragment));
    return result;
}

template <class Element>
RetileResult all_retiles(bool legacy) {
    RetileResult total;
    auto add = [&](RetileResult x) {
        total.cells += x.cells;
        total.detached_d += x.detached_d;
        total.stale_s += x.stale_s;
    };
    Element memory[256]{};
    auto tile = make_tensor(memory + 0, RowLayout<16, 16>{});
    for (int lane = 0; lane < 32; ++lane) {
        auto mma = Mma{}.get_slice(lane);
        auto a = make_fragment_like<Element>(mma.partition_fragment_A(tile));
        auto b = make_fragment_like<Element>(mma.partition_fragment_B(tile));
        auto c = make_fragment_like<Element>(mma.make_fragment_C(mma.partition_C(tile)));
        add(retile_alias<CopyRole::A>(a, lane, legacy));
        add(retile_alias<CopyRole::AT>(a, lane, legacy));
        add(retile_alias<CopyRole::B>(b, lane, legacy));
        add(retile_alias<CopyRole::C>(c, lane, legacy));
        add(retile_alias<CopyRole::CT>(c, lane, legacy));
        add(retile_alias<CopyRole::StoreC>(c, lane, legacy));
    }
    return total;
}

using Matrix = std::array<float, 256>;
using Warp = std::array<std::array<float, 8>, 32>;

// Exercise the complete six-product Neumann chain through actual native
// fragment ownership. Independent oracle: forward substitution, not Neumann.
template <FragmentMap Map>
Warp distribute(Matrix const& matrix) {
    Warp result{};
    auto identity = make_identity_tensor(Shape<_16, _16>{});
    for (int lane = 0; lane < 32; ++lane) {
        auto coords = [&]() {
            if constexpr (Map == FragmentMap::Operand)
                return Mma{}.get_slice(lane).partition_A(identity);
            else return Mma{}.get_slice(lane).partition_C(identity);
        }();
        for (int s = 0; s < 8; ++s) {
            auto const rc = coords(s);
            result[lane][s] = matrix[int(get<0>(rc)) * 16 + int(get<1>(rc))];
        }
    }
    return result;
}

template <FragmentMap From, FragmentMap To, bool Trans>
Warp exchange(Warp const& input) {
    Warp output{};
    for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 8; ++s) {
            auto owner = remap_owner<From, To, Trans>(lane, s);
            output[lane][s] = input[owner.lane][owner.slot];
        }
    return output;
}

Warp native_product(Warp const& a, Warp const& b, bool omit_c_remap) {
    Matrix result{};
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            for (int k = 0; k < 16; ++k) {
                auto x = operand_owner(r, k), y = operand_owner(c, k);
                result[r * 16 + c] += a[x.lane][x.slot] * b[y.lane][y.slot];
            }
            // All selected dyadic fixture products are exactly representable.
            float rounded = float(cutlass::half_t(result[r * 16 + c]));
            if (rounded != result[r * 16 + c]) std::abort();
        }
    auto c = distribute<FragmentMap::Result>(result);
    return omit_c_remap ? c : exchange<FragmentMap::Result, FragmentMap::Operand, false>(c);
}

int neumann_chain(bool omit_c_remap) {
    Matrix lower{}, initial{}, want{};
    for (int r = 0; r < 16; ++r) {
        initial[r * 16 + r] = 1;
        if (r) {
            lower[r * 16 + r - 1] = 0.5f;
            initial[r * 16 + r - 1] = -0.5f;
        }
    }
    for (int c = 0; c < 16; ++c)
        for (int r = 0; r < 16; ++r) {
            float x = float(r == c);
            for (int k = 0; k < r; ++k) x -= lower[r * 16 + k] * want[k * 16 + c];
            want[r * 16 + c] = x;
        }
    auto power = distribute<FragmentMap::Operand>(lower);
    auto inv = distribute<FragmentMap::Operand>(initial);
    for (int step = 0; step < 3; ++step) {
        power = native_product(power,
            exchange<FragmentMap::Operand, FragmentMap::Operand, true>(power), omit_c_remap);
        auto term = native_product(inv,
            exchange<FragmentMap::Operand, FragmentMap::Operand, true>(power), omit_c_remap);
        for (int lane = 0; lane < 32; ++lane)
            for (int s = 0; s < 8; ++s) inv[lane][s] += term[lane][s];
    }
    auto result = exchange<FragmentMap::Operand, FragmentMap::Result, false>(inv);
    auto golden = distribute<FragmentMap::Result>(want);
    int bad = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 8; ++s) bad += result[lane][s] != golden[lane][s];
    return bad;
}
}  // namespace

int main() {
    int bad = shared_tiles<16, 16, false>(false, false) +
              shared_tiles<16, 128, false>(false, false) +
              shared_tiles<16, 128, true>(false, false) +
              shared_tiles<32, 32, false>(false, false) +
              shared_tiles<32, 128, true>(false, false) +
              shared_tiles<128, 128, false>(false, false) +
              shared_tiles<128, 128, true>(false, false) +
              shared_tiles<128, 256, false>(false, false);
    int const regs = remap_check<FragmentMap::Result, FragmentMap::Operand, true>(false) +
        remap_check<FragmentMap::Result, FragmentMap::Operand, false>(false) +
        remap_check<FragmentMap::Operand, FragmentMap::Result, false>(false) +
        remap_check<FragmentMap::Operand, FragmentMap::Operand, true>(false);
    int const swizzle = shared_tiles<128, 128, false>(true, false);
    int const transpose = shared_tiles<16, 128, true>(false, true);
    int const wrong_slot = remap_check<FragmentMap::Result, FragmentMap::Operand, true>(true);
    int const inverse = neumann_chain(false), wrong_inverse = neumann_chain(true);
    auto alias_bf16 = all_retiles<cutlass::bfloat16_t>(false);
    auto alias_f16 = all_retiles<cutlass::half_t>(false);
    auto legacy_alias = all_retiles<cutlass::bfloat16_t>(true);
    int const alias_bad = alias_bf16.detached_d + alias_bf16.stale_s +
                          alias_f16.detached_d + alias_f16.stale_s;
    bool const alias_ok = alias_bad == 0 && alias_bf16.cells == 1536 &&
                          alias_f16.cells == 1536 &&
                          legacy_alias.detached_d == 1536 && legacy_alias.stale_s == 1536;
    bool const ok = bad == 0 && regs == 0 && inverse == 0 &&
                    swizzle > 0 && transpose > 0 && wrong_slot > 0 && wrong_inverse > 0 &&
                    alias_ok;
    std::printf("[PPU original delivery] %s shared_bad=%d register_bad=%d "
                "negatives=swizzle:%d/transpose:%d/source-slot:%d "
                "vector16B=CONTIGUOUS native_mma=16x16x16\n",
                ok ? "PASS" : "FAIL", bad, regs, swizzle, transpose, wrong_slot);
    std::printf("[PPU original inverse] native-products=6 all-256-elements "
                "substitution_bad=%d wrong-C-order=%d %s\n",
                inverse, wrong_inverse, ok ? "PASS" : "FAIL");
    std::printf("[PPU retile lifetime] cells=%d BF16+FP16 "
                "detached-D=%d stale-S=%d legacy-detached-D=%d legacy-stale-S=%d %s\n",
                alias_bf16.cells + alias_f16.cells,
                alias_bf16.detached_d + alias_f16.detached_d,
                alias_bf16.stale_s + alias_f16.stale_s,
                legacy_alias.detached_d, legacy_alias.stale_s,
                alias_ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
