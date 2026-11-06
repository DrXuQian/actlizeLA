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
  if (int rc = guard ? prepare_checked(g, s, *guard) : prepare(g, s))
    return rc;
  int token_blocks = 1;
#if ALA_PREFILL_TOKEN_PARALLEL && ALA_PPU_TARGET == 100
  token_blocks = prefill_schedule::blocks(g.width);
#endif
  pack_prefill<TokenParallel><<<dim3(g.requests, 2 * g.hk + g.hv, token_blocks), 128, 0, s>>>(*p);
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
