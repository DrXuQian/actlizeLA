#pragma once
#include <cstdint>

// Versioned, torch-independent launch ABI. All strides are ELEMENTS. Dynamic
// metadata lives on device; the host only knows capacities. Slot 0 is null.
namespace actlize_la::serving {
constexpr int AbiVersion = 1;
enum Error : int {
  Ok = 0,
  Bounds = 1,
  Slot = 2,
  Conflict = 3,
  Prefix = 4,
  History = 5,
  Page = 6,
  Unsupported = 7
};
enum GdnMode : int { Decode = 0, Verify = 1, ChunkPrefill = 2 };
struct Gdn {
  void const *mixed, *a, *b;
  float const *a_log, *dt_bias;
  float *state; // [slot,Hv,V,K], not the legacy KV ABI
  int32_t const *starts, *live;
  int64_t const *initial_slots, *final_slots;
  void *output;
  float *queries, *keys, *decays; // [R,Hk,W,K], [R,Hk,W,K], [R,Hv,W]
  void *betas;                    // [R,Hv,W], BF16
  float *delta;                   // [R,Hv,W,V], verify only
  int32_t *recorded_lengths;
  int64_t *recorded_slots;
  int32_t *errors; // [R], device-side fail-closed status
  int64_t const
      *checkpoint_slots; // optional, unsupported nonzero exports fail on device
  int32_t const *checkpoint_offsets;
  int64_t mixed_stride, a_stride, b_stride, state_stride, initial_stride,
      final_stride;
  int32_t requests, tokens, slots, hk, hv, width, mode;
};
struct Commit {
  float *state;
  float const *delta, *keys, *decays;
  int64_t const *slots, *recorded_slots;
  int32_t const *counts, *live, *recorded_lengths;
  int32_t *errors;
  int64_t state_stride, slot_stride;
  int32_t requests, state_slots, hk, hv, width;
};
// Padded per-request buffers bridge packed/slot metadata to the existing
// fixed-shape residual ABI. Zero beta + zero log gate makes padding identity.
struct Prefill {
  Gdn gdn;
  void *q, *k, *v, *beta, *output;
  float *log_g, *initial_kv, *final_kv;
  void *inverse, *snapshots, *vnew;
  float *gates;
};
struct ConvHistory {
  void *state, *snapshot;
  int64_t const *slots;
  int64_t *recorded_slots;
  int32_t const *live, *counts;
  int32_t *errors;
  int64_t state_stride, slot_stride;
  int32_t requests, state_slots, channels, columns;
};
} // namespace actlize_la::serving
extern "C" {
int ala_serving_version();
int ala_serving_target();
int ala_serving_struct_size(int which);
int ala_gdn_run(actlize_la::serving::Gdn const *, void *stream);
int ala_gdn_commit(actlize_la::serving::Commit const *, void *stream);
int ala_prefill_configure();
// 准备阶段查询immutable descriptor对应的默认transport，0=保留路径，1=H1。
int ala_prefill_direct_pack_route(int requests, int width, int hk, int hv);
int ala_prefill_run(actlize_la::serving::Prefill const *, void *stream);
int ala_conv_capture(actlize_la::serving::ConvHistory const *, void *stream);
int ala_conv_rollback_zero(actlize_la::serving::ConvHistory const *,
                           void *stream);
}
