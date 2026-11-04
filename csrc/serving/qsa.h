#pragma once
#include <cstdint>
namespace actlize_la::serving {
struct Qsa {
  void const *query, *new_key, *new_value, *index_query, *raw_key;
  float const *query_weight, *key_weight;
  void const *cosine, *sine;
  int64_t const *request_ids, *positions, *rope_positions, *raw_slots,
      *slot_mapping;
  int32_t const *starts, *live, *lengths, *accepted_previous;
  bool const *prefilling;
  int32_t const *main_table, *compressed_table;
  void *key_cache, *value_cache, *compressed_cache, *raw_ring;
  int64_t *raw_tags, *raw_rope, *anchors;
  float *encoded_query;
  int32_t *selected, *selected_count, *selected_groups;
  float *selected_scores;
  void *output;
  int32_t *errors;
  int64_t query_stride, key_stride, value_stride, index_stride, raw_stride;
  int64_t ring_slot_stride, tag_slot_stride, rope_slot_stride, anchor_stride,
      raw_slot_stride;
  int32_t requests, tokens, pages, raw_slot_count, table_width, page_size,
      ring_size, rope_rows;
  int32_t query_heads, kv_heads, rope_bf16;
  float norm_epsilon;
};
} // namespace actlize_la::serving
extern "C" int ala_qsa_struct_size();
extern "C" int ala_qsa_run(actlize_la::serving::Qsa const *, void *stream);
