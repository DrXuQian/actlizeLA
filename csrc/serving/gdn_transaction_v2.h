#pragma once
#include "abi.h"
#include "aux_v1.h"
// Additive ABI: legacy Gdn/Commit layouts and symbols are unchanged.
// Epochs are positive monotone per request row. Every nonempty verify is
// followed by exactly one commit, including count0, before record reuse.
struct ala_gdn_transaction_v2 {
  ala_header_v1 header;
  ala_tensor_v1 epochs, ready_epochs, consumed_epochs, upstream_errors;
  int32_t requests;
};
extern "C" {
int ala_gdn_transaction_v2_size();
int ala_gdn_verify_v2(actlize_la::serving::Gdn const *,
                      ala_gdn_transaction_v2 const *, void *stream);
int ala_gdn_commit_v2(actlize_la::serving::Commit const *,
                      ala_gdn_transaction_v2 const *, void *stream);
// Dependency-only guard: epochs/ready/consumed are absent. For decode/prefill.
int ala_gdn_checked_v2(actlize_la::serving::Gdn const *,
                       ala_gdn_transaction_v2 const *, void *stream);
int ala_prefill_checked_v2(actlize_la::serving::Prefill const *,
                           ala_gdn_transaction_v2 const *, void *stream);
}
