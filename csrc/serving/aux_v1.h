#pragma once
#include <stdint.h>

// Torch-independent C ABI. All strides are in elements. nbytes is the span
// accessible from data (not the entire backing allocation before that pointer).
// No pointers in these descriptors refer to host-side dynamic metadata.
enum ala_dtype_v1 {
  ALA_BF16 = 1,
  ALA_F32 = 2,
  ALA_I32 = 3,
  ALA_I64 = 4,
  ALA_U8 = 5
};
typedef struct {
  void *data;
  uint64_t nbytes;
  int64_t shape[3], stride[3];
  int32_t dtype, rank;
} ala_tensor_v1;
typedef struct {
  uint32_t version, struct_bytes;
} ala_header_v1;
// Flat rows: starts/slots/num_seqs empty, requests=0. Packed rows: slots -1
// inactive, slot 0 valid; upstream_errors is read-only [requests].
typedef struct {
  ala_tensor_v1 starts, slots, num_seqs, num_tokens, upstream_errors;
  int32_t requests, tokens;
} ala_rows_v1;
enum ala_pointwise_kind_v1 {
  ALA_GROUPED_NORM = 0,
  ALA_GATED_NORM = 1,
  ALA_HC_SILU = 2,
  ALA_HC_GATE_MEAN = 3,
  ALA_HC_COMBINE = 4,
  ALA_HC_COMBINE_NORM = 5,
  ALA_PLE_GATE = 6,
  ALA_PLE_CONTRIBUTION = 7
};
typedef struct {
  ala_header_v1 header;
  ala_rows_v1 rows;
  ala_tensor_v1 x, weight, gate, value, output, combined, trace, errors;
  int32_t kind, groups, width, zero_centered;
  int32_t weight_per_group, round_normalized, norm_before_gate, silu_gate;
  float epsilon;
} ala_pointwise_v1;
enum ala_conv_kind_v1 {
  ALA_CONV_PREFILL = 0,
  ALA_CONV_VERIFY = 1,
  ALA_CONV_SNAPSHOT = 2,
  ALA_CONV_RESTORE_ZERO = 3
};
typedef struct {
  ala_header_v1 header;
  ala_rows_v1 rows;
  ala_tensor_v1 x, weight, state, fresh, previous_accepted, output;
  // Snapshot is [requests,channels,history+drafts], stamps is [requests].
  // consumed_prefix is the framework's target-input count, NOT draft count.
  ala_tensor_v1 snapshot, stamps, consumed_prefix, errors;
  int32_t kind, channels, dilation, drafts;
} ala_shortconv_v1;
typedef struct {
  ala_header_v1 header;
  ala_rows_v1 rows;
  ala_tensor_v1 tokens, history, multipliers, primes, offsets;
  ala_tensor_v1 hashes, request_ids, errors;
  int64_t eos, vocab_size, padded_vocab_size;
} ala_ple_hash_v1;

#ifdef __cplusplus
extern "C" {
#endif
int ala_aux_v1_struct_size(
    int kind); // 0 tensor,1 rows,2 pointwise,3 conv,4 hash
int ala_aux_v1_validate(int kind,
                        void const *descriptor); // host-only; no driver calls
int ala_pointwise_v1_run(ala_pointwise_v1 const *, void *stream);
int ala_shortconv_v1_run(ala_shortconv_v1 const *, void *stream);
int ala_ple_hash_v1_run(ala_ple_hash_v1 const *, void *stream);
#ifdef __cplusplus
}
#endif
