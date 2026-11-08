#include "abi.h"
#include "gdn_transaction_v2.h"
#include "runtime.cuh"
#include "prefill_schedule.h"
#include <type_traits>
#if ALA_PPU_TARGET == 150
#include "../backends/ppu15/launch.h"
#elif ALA_PPU_TARGET == 100
#include "../backends/ppu10/launch.h"
#endif

namespace actlize_la::serving {
bool valid(Gdn const &);
int prepare(Gdn const &, Stream);
int prepare_checked(Gdn const &, Stream, ala_gdn_transaction_v2 const &);
bool valid_dependency(ala_gdn_transaction_v2 const *, int);

#if ALA_PPU_TARGET == 100 || ALA_PPU_TARGET == 150
constexpr bool TokenParallel = ALA_PREFILL_TOKEN_PARALLEL && ALA_PPU_TARGET == 100;
#if ALA_PREFILL_DIRECT_PACK && ALA_PPU_TARGET == 100
int validate_prefill(Gdn const &, Stream, ala_gdn_transaction_v2 const *);

// 只在已完成配对ACU和RAW-BIT的容量几何上默认选H1；其余保留token-parallel。
// 选择只看调用方descriptor，不读取device starts/live/slot或tensor内容。
inline bool direct_pack_shape(int requests, int width, int hk, int hv) {
  return requests == 1 && hk == 16 && (hv == 32 || hv == 48 || hv == 64) &&
         (width == 2048 || width == 8192);
}

// H1：沿用一整warp处理一个head/token及原XOR归一化树，直接发布最终BF16。
// 旧FP32 queries/keys 和临时decays/betas不再写；packed与输出/状态保持RAW-BIT。
__global__ void prepare_pack_prefill(Prefill p) {
  auto const &g = p.gdn;
  int const r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x % 32;
  bool const active = r < *g.live && !g.errors[r] &&
                      g.final_slots[int64_t(r) * g.final_stride] > 0;
  int const first = active ? g.starts[r] : 0;
  int const length = active ? g.starts[r + 1] - first : 0;
  int const begin = prefill_schedule::first(blockIdx.z, prefill_schedule::PrepareWarps,
                                          int(threadIdx.x) / 32);
  int const step = prefill_schedule::stride(gridDim.z, prefill_schedule::PrepareWarps);
  for (int t = begin; t < g.width; t += step) {
    bool const real = t < length;
    int64_t const token = int64_t(first) + t;
    BFloat result[4] = {};
    BFloat *dst;
    if (h < 2 * g.hk) {
      int const kh = h % g.hk;
      dst = static_cast<BFloat *>(h < g.hk ? p.q : p.k) +
            ((int64_t(r) * g.width + t) * g.hk + kh) * 128 + lane * 4;
      if (real) {
        float x[4], square = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          x[d] = read_bf(g.mixed, token * g.mixed_stride + h * 128 + lane * 4 + d);
          square += x[d] * x[d];
        }
        float const factor = rsqrtf(warp_sum(square) + 1e-6f);
#pragma unroll
        for (int d = 0; d < 4; ++d) result[d] = bf(x[d] * factor);
      }
    } else {
      int const vh = h - 2 * g.hk;
      dst = static_cast<BFloat *>(p.v) +
            ((int64_t(r) * g.width + t) * g.hv + vh) * 128 + lane * 4;
      if (real) {
#pragma unroll
        for (int d = 0; d < 4; ++d)
          result[d] = static_cast<BFloat const *>(g.mixed)[
              token * g.mixed_stride + h * 128 + lane * 4 + d];
      }
      if (lane == 0) {
        int64_t const at = (int64_t(r) * g.width + t) * g.hv + vh;
        p.log_g[at] = real ? log_decay(read_bf(g.a, token * g.a_stride + vh),
                                      g.a_log[vh], g.dt_bias[vh]) : 0.0f;
        static_cast<BFloat *>(p.beta)[at] = real
            ? bf(1.0f / (1.0f + expf(-read_bf(g.b, token * g.b_stride + vh)))) : 0;
      }
    }
    // caller-owned workspace可能为偏移view；不增加原有入口的对齐要求。
    if ((reinterpret_cast<uintptr_t>(dst) & 7u) == 0) {
      uint2 packed{uint32_t(result[0]) | (uint32_t(result[1]) << 16),
                   uint32_t(result[2]) | (uint32_t(result[3]) << 16)};
      *reinterpret_cast<uint2 *>(dst) = packed;
    } else {
#pragma unroll
      for (int d = 0; d < 4; ++d) dst[d] = result[d];
    }
  }
}
#endif

template <bool Parallel>
__global__ void pack_prefill(Prefill p) {
  auto const &g = p.gdn;
  int const r = blockIdx.x, h = blockIdx.y;
  bool const active = r < *g.live && !g.errors[r] &&
                      g.final_slots[int64_t(r) * g.final_stride] > 0;
  int const first = active ? g.starts[r] : 0;
  int const length = active ? g.starts[r + 1] - first : 0;
  // Compile out the new address arithmetic as well as the grid change for
  // the admitted baseline / other targets. Pad every neutral capacity row.
  using Index = std::conditional_t<Parallel, int64_t, int>;
  int const begin = Parallel ? prefill_schedule::first(blockIdx.z, blockDim.x, threadIdx.x)
                             : int(threadIdx.x);
  int const step = Parallel ? prefill_schedule::stride(gridDim.z, blockDim.x) : int(blockDim.x);
  for (Index i = begin; i < Index(g.width) * 128; i += step) {
    int const t = i / 128, d = i % 128;
    bool const real = t < length;
    if (h < 2 * g.hk) {
      int const kh = h % g.hk;
      auto src = h < g.hk ? g.queries : g.keys;
      auto dst = static_cast<BFloat *>(h < g.hk ? p.q : p.k);
      dst[((int64_t(r) * g.width + t) * g.hk + kh) * 128 + d] =
          real ? bf(src[((int64_t(r) * g.hk + kh) * g.width + t) * 128 + d])
               : 0;
    } else {
      int const vh = h - 2 * g.hk;
      static_cast<BFloat *>(
          p.v)[((int64_t(r) * g.width + t) * g.hv + vh) * 128 + d] =
          real ? static_cast<BFloat const *>(
                     g.mixed)[int64_t(first + t) * g.mixed_stride + h * 128 + d]
               : 0;
      if (!d) {
        int64_t const dst = (int64_t(r) * g.width + t) * g.hv + vh,
                      src = (int64_t(r) * g.hv + vh) * g.width + t;
        p.log_g[dst] = real ? g.decays[src] : 0;
        static_cast<BFloat *>(p.beta)[dst] =
            real ? static_cast<BFloat const *>(g.betas)[src] : 0;
      }
    }
  }
}
__global__ void gather_state(Prefill p) {
  auto const &g = p.gdn;
  int const r = blockIdx.x, h = blockIdx.y;
  auto const src = r < *g.live && !g.errors[r]
                       ? g.initial_slots[int64_t(r) * g.initial_stride]
                       : 0;
  for (int i = threadIdx.x; i < 16384; i += blockDim.x) {
    int const k = i / 128, v = i % 128;
    p.initial_kv[(int64_t(r) * g.hv + h) * 16384 + i] =
        src ? g.state[src * g.state_stride + (int64_t(h) * 128 + v) * 128 + k]
            : 0;
  }
}
template <bool Parallel>
__global__ void scatter_prefill(Prefill p) {
  auto const &g = p.gdn;
  int const r = blockIdx.x, h = blockIdx.y;
  if (r >= *g.live || g.errors[r])
    return;
  int const first = g.starts[r], length = g.starts[r + 1] - first;
  auto dst = g.final_slots[int64_t(r) * g.final_stride];
  using Index = std::conditional_t<Parallel, int64_t, int>;
  int const begin = Parallel ? prefill_schedule::first(blockIdx.z, blockDim.x, threadIdx.x)
                             : int(threadIdx.x);
  int const step = Parallel ? prefill_schedule::stride(gridDim.z, blockDim.x) : int(blockDim.x);
  for (Index i = begin; i < Index(length) * 128; i += step) {
    int const t = i / 128, d = i % 128;
    static_cast<BFloat *>(g.output)[(int64_t(first + t) * g.hv + h) * 128 + d] =
        dst ? static_cast<BFloat const *>(
                  p.output)[((int64_t(r) * g.width + t) * g.hv + h) * 128 + d]
            : 0;
  }
  if (dst && length)
    // Each state element has one owner too: do not repeat the entire state
    // write in every token block. Empty rows never modify their final slot.
    for (int i = begin; i < 16384; i += step) {
      int const k = i / 128, v = i % 128;
      g.state[dst * g.state_stride + (int64_t(h) * 128 + v) * 128 + k] =
          p.final_kv[(int64_t(r) * g.hv + h) * 16384 + i];
    }
}
#endif
} // namespace actlize_la::serving
// Preparation-time experiment identity, not an execution/device query.
extern "C" int ala_prefill_token_parallel() {
#if ALA_PREFILL_TOKEN_PARALLEL && ALA_PPU_TARGET == 100
  return 1;
#else
  return 0;
#endif
}
// 准备期只读route断言，绑定真正默认raw入口采用的transport。
extern "C" int ala_prefill_direct_pack_route(int requests, int width, int hk, int hv) {
#if ALA_PREFILL_DIRECT_PACK && ALA_PPU_TARGET == 100
  return actlize_la::serving::direct_pack_shape(requests, width, hk, hv) ? 1 : 0;
#else
  return 0;
#endif
}
extern "C" int ala_prefill_configure() {
#if ALA_PPU_TARGET == 150
  return actlize_la_ppu15_configure();
#elif ALA_PPU_TARGET == 100
  return actlize_la_ppu10_configure();
#else
  return 0;
#endif
}
static int prefill_impl(actlize_la::serving::Prefill const *p,
                        ala_gdn_transaction_v2 const *guard, void *stream) {
  using namespace actlize_la::serving;
  if (!p || !valid(p->gdn) || p->gdn.mode != ChunkPrefill)
    return -1;
  if (guard && !valid_dependency(guard, p->gdn.requests))
    return -1;
#if ALA_PPU_TARGET == 100 || ALA_PPU_TARGET == 150
  if (!p->q || !p->k || !p->v || !p->beta || !p->output || !p->log_g ||
      !p->initial_kv || !p->final_kv || !p->inverse || !p->snapshots ||
      !p->vnew || !p->gates)
    return -1;
  auto s = static_cast<Stream>(stream);
  auto const &g = p->gdn;
  bool const direct = ala_prefill_direct_pack_route(g.requests, g.width, g.hk, g.hv) != 0;
#if ALA_PREFILL_DIRECT_PACK && ALA_PPU_TARGET == 100
  if (direct) {
    if (int rc = validate_prefill(g, s, guard)) return rc;
  } else
#endif
  {
    if (int rc = guard ? prepare_checked(g, s, *guard) : prepare(g, s)) return rc;
  }
  int token_blocks = 1;
#if ALA_PREFILL_TOKEN_PARALLEL && ALA_PPU_TARGET == 100
  token_blocks = prefill_schedule::blocks(g.width);
#endif
#if ALA_PREFILL_DIRECT_PACK && ALA_PPU_TARGET == 100
  if (direct) {
    prepare_pack_prefill<<<dim3(g.requests, 2 * g.hk + g.hv, token_blocks),
                           prefill_schedule::PrepareThreads, 0, s>>>(*p);
  } else
#endif
  {
    pack_prefill<TokenParallel><<<dim3(g.requests, 2 * g.hk + g.hv, token_blocks), 128, 0, s>>>(*p);
  }
  if (int rc = last_error())
    return rc;
  gather_state<<<dim3(g.requests, g.hv), 256, 0, s>>>(*p);
  if (int rc = last_error())
    return rc;
#if ALA_PPU_TARGET == 100
  auto forward = actlize_la_ppu10_forward_prepared;
#else
  auto forward = actlize_la_ppu15_forward_prepared;
#endif
  if (int rc = forward(
          p->q, p->k, p->v, p->log_g, p->beta, p->initial_kv, p->output,
          p->final_kv, p->inverse, p->snapshots, p->vnew, p->gates, g.requests,
          g.width, g.hk, g.hv, true, stream))
    return rc;
  scatter_prefill<TokenParallel><<<dim3(g.requests, g.hv, token_blocks), 256, 0, s>>>(*p);
  return last_error();
#else
  // Explicit SM120 semantic-validation path, not a tensor-core performance
  // claim or a fallback for either native PPU generation.
  return guard ? ala_gdn_checked_v2(&p->gdn, guard, stream)
               : ala_gdn_run(&p->gdn, stream);
#endif
}
extern "C" int ala_prefill_run(actlize_la::serving::Prefill const *p,
                               void *stream) {
  return prefill_impl(p, nullptr, stream);
}
extern "C" int ala_prefill_checked_v2(actlize_la::serving::Prefill const *p,
                                      ala_gdn_transaction_v2 const *guard,
                                      void *stream) {
  if (!guard)
    return -1;
  return prefill_impl(p, guard, stream);
}
