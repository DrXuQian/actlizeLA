# ZW810: actlizeLA / PLA / FLA core comparison

This experiment adds no production kernel or default-route changes. It measures
the public actlizeLA default, the unmodified `t-head/pla` prefill API, and FLA's
Triton implementation with backend dispatch disabled. No performance verdict is
claimed until the box's numerical gates and full ACU inventories pass.

## Why chunk 64?

Our PPU1.0 WY solve/state/output implementation was built and validated around
C64. That is **not evidence that 64 wins a chunk-size sweep**. For T2048, C64
requires 32 recurrent chunk steps, while C32 requires 64. Smaller chunks reduce
intra-chunk triangular products and scratch, but increase recurrent transitions
and associated work. PLA also fuses state/output and retains intermediates in
registers, whereas our current implementation publishes state/Vnew for a
chunk-parallel output kernel. These are different decompositions, not a fair
one-line `64 -> 32` ablation.

Keep native validated geometry for this first comparison: **default/FLA C64,
PLA C32**. The result compares the implementations as a whole; it cannot by
itself attribute a win/loss to chunk size. A later C32/C64 experiment in one
implementation needs its own numerical/rounding and resource validation.

## Fixed scope and arithmetic

- Eight cells: B1, Hk16, Hv32/48, T2048/8192, nominal natural-log g=-0.1/-1.
  K=V=128. Reuse the previous ZW810 run's actual processed operand files;
  do not regenerate normalized Q/K or gate values.
- Q/K/V and output BF16; asymmetric nonzero initial and final state FP32,
  `[B,Hv,K,V]`. Initial state must not change. Query scale `1/sqrt(128)` once;
  inputs are already QK-normalized, g is already natural-log decay, beta is
  already sigmoid-rounded BF16.
- PLA accepts FP32 beta. Promote these exact BF16 values losslessly on CPU
  before the call. Do not recompute sigmoid or silently change their values.
  Hash the physical FP32 beta too: rounding it back before hashing could hide
  a mutation. The API's indexed VK state mode is **not** used.
- Independent saved sequential recurrence oracle, unchanged historical 2%
  normalized-max-error gate for **both output and state**, eight stable repeats.
  Different chunk rounding is not required to be cross-library bit-exact.
- PLA is pinned in `tools/gdn_threeway_contract.py`, with its pinned actlize
  submodule. Build only `pla.prefill.flashqla` using upstream default flags in
  a private runtime; never replace installed PLA, FLA, Torch or actlizeLA.
  `auto_cp` remains upstream default. These geometries/legacy nonzero state
  do not enter the current tc01 CP admission; unexpected work stays visible.

## Measurement and verdict

All three arms are measured again on the same physical ZW810. Historical
latencies are not mixed with the new samples. The old numerical receipts,
library hash, processed-input hashes and device/runtime must still match.

Use only `/sim/eec/shared/junfu.qx/asight/bin/acu` and its companion sections,
checked before building. Each profile process calls exactly one complete
public core API, no device warmup/verification kernels. Sum **every captured
kernel**, including FLA zero fills. Expected PLA inventory is cumsum + tc01
KKT solve + fused state/output; not just the last kernel. Missing, duplicate
or unrecognized work makes the verdict UNADJUDICATED, never a filtered win.

Three captures per role with rotating order; noisy cells and cells near the
existing default/FLA 0.80 target get three additional captures using the existing
collector's rule. Keep all samples. Report all three pairwise observed-envelope
verdicts. Overlap means UNRESOLVED. The existing 20%-lower-time default/FLA target
is retained separately, not silently applied as a different PLA target.

This is **instrumented ACU full-core kernel time**, not uninstrumented API time,
Flare serving time, host overhead, or total model latency. Capture one workload
at a time on an otherwise idle device. Driver/power/clock dumps and raw ACU
counters remain available to identify environment drift.

## Box command

From this branch's checkout, using the same box SDK/Torch/FLA as the passed run:

```bash
INPUT_RUN=/workspace/actlize-la-zw810-recheck-b810b6c7-20261109T083458Z-2075017 \
DEVICE=0 JOBS=4 \
bash tools/run_ppu_gdn_threeway_box.sh
```

Set `PPU_SDK=/actual/matching/PPU_SDK` only if the working SDK is not
`/usr/local/PPU_SDK`. `DEVICE` is the physical ordinal. `PYTHON` can select the
working PPU Torch Python. Unset inherited `FF_*`, `PLA_*` or
`GDN_QSA_WY_EXTENSION` experiment overrides; the script will not quietly alter
an overridden reference. Network access is needed to fetch the pinned PLA and
actlize sources. No actlizeLA rebuild is performed.

The runner creates `/workspace/actlize-la-gdn-threeway-<UTC>-<pid>/` with:

- `evidence/comparison.txt`: eight-cell, three-arm medians/ranges/verdicts;
- `evidence/summary.json`: all samples, registration, source/binary hashes,
  precision/chunk/route metadata, errors and collection status;
- `results.tar.gz`: small upload with logs, numerical receipts and exported
  ACU counters. Send this file, not the whole build tree;
- PLA source/build/runtime and original `.acurep` remain on the box. Original
  operand files and actlizeLA library remain untouched in `INPUT_RUN`.

`COLLECTED` means measurement completed, **not** that our kernel won.
`UNADJUDICATED`/`INCOMPLETE` are not PASS. No missing cell disappears from the
denominator. No automatic production promotion follows this experiment.

## Local qualification (2026-11-09)

Host comparison/adapter/inventory/receipt regressions: 262 passed, 26 subtests
passed; 2 native-library checks explicitly skipped without their compiled
libraries. Shell syntax and missing-INPUT_RUN failure also checked. These are
not PLA PPU compile/device/performance results; those three remain NOT_RUN
until this runner is executed on the box.
