# PPU1.0 matched geometry: uploaded ACU results

Measured source `3a8226d5b04956c0178692302467cd2cf47616c4`.
The [original geometry registration](PPU10_GEOMETRY.md) is unchanged in its
workloads and decision rules. This analysis performs no device execution.

**Geometry is shape-dependent. V16/4 is a finalist only for the two smaller
full-C64 workloads; V32/4 is not a finalist. Keep the V32/8 control for the
larger workloads and tails. No default selector is promoted by this sweep.**
Every number below is one ACU capture, not a repeated distribution. The
single-capture rule requires repeated paired confirmation before promotion.

## All registered cells, including losses

BF16 inputs, K=V128, C64, initial state=None, final state required. Times are
microseconds, summing **all kernels**: four for each WY arm, seven for FLA
(including both fill kernels). They are not host/API event spans. In V/W,
V means state value columns per CTA, W means warps per CTA.

| B / S / Hk / Hv | Gate | Retained V32/8 control | Matched V32/8 calibration | V32/4 | V16/4 | FLA |
|---|---:|---:|---:|---:|---:|---:|
|1 /2048 /16 /32|-1.0|169.503|168.852|184.136|164.750|227.142|
|1 /2048 /16 /32|-0.1|169.718|168.074|183.251|165.751|226.986|
|1 /2048 /8 /16|-1.0|125.842|126.073|144.085|122.684|158.988|
|1 /2048 /8 /16|-0.1|126.210|125.531|142.508|122.479|158.605|
|2 /2048 /16 /32|-1.0|284.578|285.025|288.224|325.201|402.198|
|2 /2048 /16 /32|-0.1|285.609|282.453|289.205|324.878|401.809|
|1 /2048 /16 /64|-1.0|278.399|280.204|281.295|320.848|389.914|
|1 /2048 /16 /64|-0.1|278.976|279.351|281.854|323.238|389.721|
|1 /2049 /16 /32|-1.0|183.733|180.560|210.979|184.462|229.083|
|1 /2049 /16 /32|-0.1|180.926|182.907|210.473|181.568|228.194|
|1 /2111 /16 /32|-1.0|180.784|181.125|210.891|184.953|231.054|
|1 /2111 /16 /32|-0.1|182.582|179.742|212.450|183.605|230.829|

Relative to the retained control:

- V16/4 has four lower observations, all in B1/S2048/Hv16 or Hv32:
  **2.34–2.96%** lower complete-call latency. Its state is 2.74–4.11us
  lower; unchanged-stage contributions range from -0.65 to +0.21us.
- V16/4 at B2/Hv32 or B1/Hv64 is **13.75–15.87% slower**. The changed
  state contributes +40.40–44.06us, versus -1.66 to +0.45us from the
  unchanged stages. This is not a regression manufactured by output timing.
- V16/4 tails have no observed complete-call gain: +0.36–2.31%. Calibration
  noise overlaps this scale, so these are not established regression sizes.
- V32/4 is higher in **all12 cells**, by1.03–16.65%. Do not promote it.
- The retained control itself is1.25–1.41x faster than same-cell FLA in
  these captures. This is a descriptive ratio, not a new cross-device claim
  or the suspended1.5x target.

The V32/8 calibration has the same native instructions/operands as the
retained control (both full and generic). Its complete-call differences
range from **-1.73% to +1.09%**, or -3.17 to +1.98us. They are a noise
check, not a new tolerance band to retroactively classify other arms. All
276 kernels report the same1.700GHz CE frequency. Do not blame these
differences on a measured frequency change that did not occur.

## Why the geometry rankings differ

At B1/S2048/Hk16/Hv32/g=-1, using the matched V32/8 arm as geometry control:

| State fact | V32/8 | V32/4 | V16/4 |
|---|---:|---:|---:|
|State time, us|79.539|94.765|75.386|
|CTA grid / threads|128 /256|128 /128|256 /128|
|Registers/thread|120|162|96|
|Shared bytes/CTA|46,080|46,080|35,840|
|Register/shared/warp block ceilings|4 /5 /8|6 /5 /16|10 /7 /16|
|Combined resident CTA ceiling/CU|4|5|7|
|Achieved active warps/CU|14.12|7.07|14.08|
|Executed instructions, `pu__inst_executed.sum`|12,317,824|10,438,272|12,108,032|
|Shared load transactions|11,239,424|9,109,504|11,239,424|
|Shared bank conflicts|4,440,064|4,440,064|4,374,528|
|KVD global-load bytes, MiB|18|18|36|
|KVD global-store bytes, MiB|50|50|52|
|Global-to-shared staging bytes, MiB|112|112|224|

**V32/4 does less delivery/control work but has half the runnable warps in
this finite grid.** Dynamic instructions fall15.26%, shared reads fall18.95%,
and BC is unchanged, yet state time rises about19.1%. The CTA count has not
increased, so its theoretical5-block capacity cannot create more work.
The lower per-issue memory-dependency and sync ratios (2.04->1.43 and
1.22->0.51) are not lower elapsed latency; their denominators and active
warp population changed. This is a counterexample to selecting from either
instruction count or one stall ratio alone.

**V16/4 does not double achieved warps.** It doubles CTAs while halving
warps/CTA, leaving total launched warps unchanged. K/inverse staging is
duplicated across V slices; the actual global-to-shared counter doubles.
BC falls only1.48%. Neither "occupancy doubled" nor "BC was eliminated"
describes its modest small-workload gain.

For B1/Hv64 (and independently B2/Hv32), the capacity geometry changes:

| State fact, B1/S2048/Hv64/g=-1 | V32/8 | V16/4 |
|---|---:|---:|
|CTAs|256|512|
|72CU x resident CTA ceiling|288|504|
|Reported waves/CU|0.89|1.02|
|Achieved active warps/CU|28.16|26.99|
|State time, us|116.201|159.108|
|Global-to-shared staging, MiB|224|448|

V32/8 fits in the aggregate resident capacity; V16/4 crosses it by eight
CTAs while staging twice the bytes and achieving no extra active warps.
This is a concrete capacity/traffic disadvantage consistent with the slower
state. The capture does **not** isolate how many microseconds come from
the final partially occupied wave versus duplicated staging or dependencies;
do not claim a measured causal allocation without a discriminating experiment.

The S2049/S2111 measurements use the registered generic tail-safe state,
not the separately retained mixed-full-prefix/final-tail optimization.
Do not extrapolate this geometry result to their unmeasured composition.

## Coverage and evidence boundaries

- 12/12 cells,60/60 arms,276/276 kernels are present. All raw and details
  kernel symbols, grids, blocks and durations agree exactly.
- 36 retained cases and256 edge cases were counted by unique keys, not just
  PASS footers. Each has the retained control and all three geometry arms,
  8 repetitions, RAW-BIT equality and unchanged independent2% admission.
  No duplicate case/delivery keys. Every captured cell also has matching
  independent preflight and subject receipts, identical input/reference and
  WY output fingerprints, and one public subject call.
- All760 archive members (759 payload checksums plus the checksum manifest),
  selected original capture manifests, source SHA,
  device/runtime/library identities and native audit logs reconcile. Native
  calibration audit reports identical instruction/operand sequences. No
  binaries, ISA or original `.acurep` bytes were uploaded or executed here;
  original report hashes remain capture-time declarations, not rehashed
  proof of the absent report bytes.
- Native resource records report STACK SIZE:0 for all six geometry images.
  ACU's separate per-thread launch stack metric is112/128/96B for these full
  arms. These are different records; the ACU launch field alone is not
  evidence of register spilling, and this upload contains no per-PC spill
  attribution. Box compiler HGCC2.2.0-dev, not the earlier local2.1.1 build.
- ACU raw text repeats some device attributes with different units/values
  (max-registers255 vs256). The auditor does not use those duplicates as
  actual launch resources; `launch__registers_per_thread` is unambiguous.
- Seven planted analysis defects are rejected after refreshing the fixture
  checksums: missing cell, missing arm export, omitted FLA fill, wrong
  geometry symbol, wrong duration unit, mixed device and wrong input shape.
  These are semantic negatives, not merely checksum failures.

Archive `geometry-light.tar.gz` SHA256:
`46f6015f07e26ea892ff39e0e9168e9e5d83c426c3b1bf3744d6c7cce45803fd`.
Library SHA256:
`cf47069ce87fca9a5dadd320bf5ac3b81f65b70896b03cfe14d698838b89bdf6`.
PPU-ZW810/72CU, device UUID `019ee024-8860-091c-0000-0000007aff6d`;
Torch2.9.0/runtime12.9. Original ACU executable remains
`/sim/eec/shared/junfu.qx/asight/bin/acu`.

Reproduce the read-only audit (no PPU SDK/device required):

```bash
python3 tools/analyze_ppu10_geometry.py /path/to/geometry-light.tar.gz --self-test
```

Structured result: [geometry_acu_20260929.json](../dev/ppu/results/geometry_acu_20260929.json).

## Decision and next step

The matched-geometry **implementation and migration experiment are closed**:
it is implemented, locally proven, device-correct and measured over the
registered space. It is not a universal tile improvement.

Next, confirm only V16/4 versus the retained V32/8 control for the four
promising full-chunk cells, using repeated same-cell paired ACU captures
under the existing disjoint-envelope criterion. Do not expand the sweep or
promote tail/large-workload configurations from these data. Preserve V32/8
for the other measured shapes. Shape-dependent performance admission and
the unified public selector remain the integration boundary; existing
default/public routing is unchanged.
