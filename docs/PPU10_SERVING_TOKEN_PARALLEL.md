# PPU1.0 serving prefill: token-parallel transport experiment

Status: experimental build only, **not device/performance admitted**. Main,
v0.2.2, installed wheels and the GDN core selector are unchanged.

Local checkpoint:265 host tests +38 subtests PASS;2 native ABI tests SKIP
(no runtime library loaded in that host test process, not counted as PASS).
Both OFF/ON transport device bodies compile with SDK2.2.1 for PPU1.0. Candidate
DSO links against byte-unchanged released core objects with `--no-undefined`;
this is not a fresh full CMake rebuild or device execution. No-device runner
returns3/SKIP. Box script rebuilds the full exact candidate and requires all
device gates. Local linked DSO SHA256:
`39357f8d76583ed2f5bdef15da5bfaf06965ebb3e5748096be3ad5b241544eab`.

## Observed problem

The 2026-11-06 upload has one complete ACU serving capture for B1/T2048,
Hk16/Hv32,K=V128,BF16 operands, FP32 asymmetric state, g approximately -0.1.
It contains nine kernels, not just GDN math:

| Kernel | Duration (us) |
|---|---:|
| validate_gdn | 5.58882 |
| prepare_gdn | 1633.47118 |
| pack_prefill | 1061.14529 |
| gather_state | 14.08529 |
| GDN prefix + solve + state + output | 169.76353 |
| scatter_prefill | 419.04412 |
| Complete serving sum | 3303.09823 |

Source mechanism: prepare launches only B*(2Hk+Hv) **one-warp** CTAs and
serially traverses T tokens. Pack and scatter also expose only B*heads CTAs.
These are independent-token operations; they are not the recurrent state scan.
The core already uses the retained full-chunk residual implementation.

Evidence: uploaded results.tar.gz SHA256
`f31c5788f57057c246a030e0d11e666551e21c2930f94987a5c47e2061c06b7a`;
binary SHA256 `ef2dbfd30af2f095e0b01c749560cfabf9fdddab2814eeb6e181d1d793a2db83`.
The upload is incomplete only because the collector expected a wide CSV rather
than the site's long metric rows. FLA/default capture **did not run**. This is
one unpaired observation, not a resolved speedup or a whole-model measurement.
The reduced real Duration/Active Cycles rows are a parser regression fixture in
`tests/data/serving_acu_long_duration.csv`. Original reports stay on the box.

## One change: expose token parallelism

Build `-DALA_PREFILL_TOKEN_PARALLEL=ON` (default OFF, PPU10 only).
Launch z=`min(ceil(width/32),65535)`, using fixed capacity, not CPU-read lengths.
Prepare uses128 threads: one complete warp per head/token, four FP32 values
per lane and the **same XOR sum tree/epsilon**. Independent tokens stride over
the added warps. Pack/scatter distribute their original element intervals over
z as well. Final-state writes are partitioned too, never duplicated per block.
Grid z is capped and loops cover additional work without truncation. Scalar
address intervals use64-bit arithmetic where a grid stride could overflow int.

No new buffers/allocations, no kernel-count change, no state-layout change,
no changed BF16 boundary, no extra query scale, no gate approximation or fusion.
Gather, GDN core and decode/verify/commit scheduling remain unchanged. The
preparation-time `ala_prefill_token_parallel()` query binds the compiled arm;
it does not inspect a device or execute a kernel. Both old/new launch ABI and
workspace extents are unchanged.

## Admission and execution

1. Host coverage proof plus missing/duplicate ownership negatives. SDK2.2.1
   PPU10 compilation of both changed device translation units.
2. Existing full serving + auxiliary + native-graph device gates unchanged.
3. Same-input old/new parity for meaningful scratch, output and state, Hk16,
   Hv32/48, T31/32/33/63/64/65/127/128/129/2048/8192. Eight graph replays change
   live counts, lengths and initial/final slots; include null final and empty
   requests. Exact bit comparison and planted one-bit failure, no tolerance
   relaxation. Independent recurrence remains in the full device gates.
4. ACU capture of candidate serving, public default core and FLA, repeated
   serially with identical processed inputs and unchanged numerical admission.
   Count every helper. Kernel sums are not uninstrumented API latency. Final
   production routing waits for this evidence; no speedup has been measured
   locally. An improvement over the uploaded n=1 baseline is descriptive,
   not a same-session paired confidence claim.

On an idle PPU1.0, in the diagnostic branch checkout:

```bash
BASELINE_SERVING_RUN=/workspace/actlize-la-flare-ppu10-226355a5-20261106T031921Z-3421332 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 JOBS=16 bash tools/run_ppu_prefill_parallel_box.sh
```

To profile **only the existing default core** at Flash-Next Hk16/Hv48, without
building or waiting for this experiment (numerical preflight is outside ACU):

```bash
SERVING_RUN=/workspace/actlize-la-flare-ppu10-226355a5-20261106T031921Z-3421332 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 bash tools/run_ppu_flashnext_core_acu_box.sh
```

Its default is B1/T2048, BF16 QKV/beta, FP32 asymmetric state and synthetic
g~-0.1. Use `TOKENS=8192` for the longer cell. It prints all four core durations
and their sum; this one capture does not establish a repeated speed verdict.

This builds into a fresh `/workspace` directory, validates before profiling,
uses `/sim/eec/shared/junfu.qx/asight/bin/acu` with its own sections, and prints
small numerical and ACU `results.tar.gz` files. It never installs or overwrites
a binary. Defaults profile T2048/Hv32 and **Flash-Next Hv48**; add
`--lengths 2048 8192` for the extension. Missing device, incorrect arm, numerical
failure or profiler/parser failure cannot be reported as a performance PASS.
