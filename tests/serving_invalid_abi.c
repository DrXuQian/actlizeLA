/* Host-only admission negative. Never computes or launches any device work. */
#ifndef TEST_ABI_VERSION
#define TEST_ABI_VERSION 0
#endif
int ala_serving_version(void) { return TEST_ABI_VERSION; }
int ala_serving_target(void) { return 100; }
int ala_serving_struct_size(int which) { (void)which; return 1; }
#define INVALID(name) int name(void) { return -1; }
INVALID(ala_gdn_run)
INVALID(ala_gdn_commit)
INVALID(ala_prefill_configure)
INVALID(ala_prefill_run)
INVALID(ala_conv_capture)
INVALID(ala_conv_rollback_zero)
INVALID(ala_qsa_struct_size)
INVALID(ala_qsa_run)
INVALID(ala_gdn_transaction_v2_size)
INVALID(ala_gdn_verify_v2)
INVALID(ala_gdn_commit_v2)
INVALID(ala_gdn_checked_v2)
INVALID(ala_prefill_checked_v2)
INVALID(ala_aux_v1_struct_size)
INVALID(ala_aux_v1_validate)
INVALID(ala_pointwise_v1_run)
INVALID(ala_shortconv_v1_run)
#ifndef TEST_MISSING_EXPORT
INVALID(ala_ple_hash_v1_run)
#endif
