# ZW810 GDN versus FLA: user-operated KDA loop

Date: 2026-11-09 UTC. Parent `ee215dd919cc144ded85966f9ac4c08c3bddcb5d`;
branch `tune/zw810-gdn-fla-kda`. Production core remains `dbf0024`.
No kernel/selector change is made by this initial contract/evidence commit.

## Outcome and responsibilities

The user requests that actlizeLA beat FLA and will run intermediate box
experiments. The agent owns source/counter analysis, bounded hypotheses,
implementation, local tests and a pushed one-command handoff. The user owns
device execution and returns the small `results.tar.gz`. Do not SSH to another
device, reconfigure clocks/drivers, or claim a device win from local compilation.

Priority scope is the current eight ZW810 cells: B1, Hk16, Hv32/48,
T2048/8192, K=V128, natural-log g=-0.1/-1. The first milestone is eight
individual uncertainty-resolved wins over FLA, without losing admitted
per-shape winners. Keep reporting the previous 20%-lower-time target
separately; missing that target is not a reason to discard a smaller real win.
This is not a claim about the wider 54-cell matrix, ZW610 or PPU1.5.
No wall-clock deadline has been imposed; each implementation round ends at
the user-operated device handoff, not at an unverified promotion.

Keep the same processed inputs, BF16 QKV/beta, FP32 asymmetric nonzero KV
state, one query scale, independent sequential oracle and original tolerances.
Use the installed, hash-bound FLA Triton route, with backend dispatch disabled.
Do not improve performance by silently changing normalization, dropping final
state, approximating away old state on strong decay or weakening the oracle.

## Baselines and current evidence

Freeze the passed box library and old input files. Compare a candidate in a
separate build/runtime. No default promotion until its scope is admitted.
At phase 0 the latest three-core table was USER_REPORTED: default wins 8/8
against PLA, 2/8 against FLA. Its raw archive has now been audited (round 1
below); the pairwise conclusions are unchanged.

There is already a fully audited same-default baseline bundle, so the user
need not repeat it just to start analysis:

- archive SHA256 `8cf604490c529a6b8e91c88be453bc098d6c729313c7c47b3bf602938968cf3d`;
- harness `b810b6c707c4e2e6f69d3191925890a0eff18a3f`;
- library SHA256 `e86ea77e1bcda59c5d370cdfa3df9c21d3430a5aa7dc101e0fac5b77419489ab`;
- all imported CE frequencies 1.7 GHz; FLA 0.6.0/Triton 3.5.0, entry hash in receipts;
- new default medians in the returned table differ by less than 0.6% from it.

`dev/ppu/zw810_fla_kda/analyze_baseline.py` rechecks CSV inventories and
numerical/capture receipts, then constructs an additive mean-time phase budget.
It supports both this older bundle and the new three-core archive, reads tar
members as data and never executes uploaded source. Unknown work or a missing
registered cell fails. Means are used for additive accounting; summing phase
medians is not an exact full-call median decomposition.

Old audited phase differences (us, **ours minus FLA**, range across two g):

| Geometry | solve | state-equivalent | output | Interpretation |
|---|---:|---:|---:|---|
| T2048/Hv32 | +4.19 to+4.20 | +28.27 to+29.18 | +4.36 to+5.05 | State chain is the main excess. |
| T2048/Hv48 | +7.06 to+7.59 | +3.60 to+3.96 | +13.76 to+14.08 | Output matters more than state. |
| T8192/Hv32 | +39.41 to+40.02 | +73.92 to+77.80 | +1.74 to+3.37 | State first, solve also material. |
| T8192/Hv48 | +59.73 to+61.05 | **-67.69 to-65.18** | +17.86 to+19.64 | Preserve the residual-state advantage. |

State-equivalent is **our residual state versus FLA W/U + state**. Prefix
and FLA fill kernels remain in the full budget; the table is not a new timing
run and its rows must not be substituted for latest three-core medians.

The representative T2048/Hv32/g=-0.1 state launches both have grid 128.
Ours uses 256 threads, 128 regs/thread, 46,080 B shared, 14.19 active warps/CU;
FLA uses 128 threads, 224 regs/thread, 122,880 B shared, 7.11 active warps/CU.
Yet FLA W/U+state is faster. This rules out "we merely have fewer active
warps" as an explanation of this gap, not all occupancy/issue-readiness effects.
Keep the exact ACU metric identity when comparing instruction counts; do not
substitute a different per-opcode or per-scheduler denominator.

## Hypotheses to distinguish, not explanations to assume

1. **State critical path and work delivery.** Ours performs K@H, P@R, Kᵀ@scaledV
   inside the recurrent loop. FLA materializes W/U in chunk-parallel work,
   leaving W@H and Kᵀ@scaledV recurrent. Our lower global GEMM count need not
   imply a shorter critical path. First compare native delivery/round trips
   and the cost of P@R; a W/U decomposition is a controlled counterfactual,
   not an unconditional global replacement. It must include W/U time/traffic
   and paired snapshot/output layout changes. Hv48/T8192 is a mandatory control.
2. **Solve arithmetic and ownership.** Our off-diagonal products use TF32
   high/high plus two residual products; captured FLA source explicitly uses
   `input_precision='tf32'` where supported. The old C64 native model is three
   MMA per half-product versus one; actual current per-opcode counts are still
   needed before claiming a measured 3x count. TF32x1 is a separate, opt-in
   rounding experiment with the same numerical limits, not a scheduling edit.
   A same-precision register/warp-owner experiment is separate from it.
3. **Output at Hv48.** Compare selected tile/CTA geometry, H/Vnew delivery,
   gate evaluation and native waits. Do not change it in the first state-only
   experiment or infer that lower registers/BC alone must lower latency.

No chunk32 rewrite, blanket fusion, register cap or unrestricted config sweep
is preregistered. These are possible later hypotheses if measurements support
them. The PLA result does not prove C64 optimal or fusion universally bad.

## Per-round contract

1. Record parent SHA, one hypothesis, exact files/axes changed, unchanged
   controls, expected discriminating counters, and a falsification condition.
2. Perform local algebra/layout/ownership tests and negative controls. Preserve
   AIU.swzl / ld.swzl pairing; inspect native lowering for the actual SDK.
   Host/compiler absence is SKIP/NOT_RUN, never device PASS.
3. Push one candidate command. Reuse immutable inputs/baselines and unchanged
   build objects when source/SDK identity permits. Do not rebuild PLA each round.
4. User runs numerics before timing. Same-order edits must also be RAW-BIT equal;
   explicitly changed math gets the unchanged independent output/state oracle,
   nonzero asymmetric state, both gates, long sequence/tail/head-map coverage.
5. Compare baseline/candidate/FLA serially on the same idle ZW810, with rotated
   order and repeated full-call site-ACU captures. All work is counted; no
   API-timer substitution or faster single-stage promotion. Retain all samples.
6. Decide KEEP/REVISE/REJECT per registered cell. Overlapping envelopes are
   UNRESOLVED. A shape-specific winner can be retained while another shape
   keeps its old route. All 8 must be measured before claiming the milestone;
   the wider matrix and serving regression are required before promotion.

Stop each implementation round at the reproducible user-operated box handoff.
Resume from returned artifacts, not guesses from a screenshot or average wins.

## Round 1: parallel W/U counterfactual

The uploaded three-core archive is now audited: SHA256
`4b7d6d8c3e9c233fd4755264b6cf52fcb184ca54b91f5172b91feda43e8c043f`,
harness `ee215dd`, the same `e86ea77e` default library, 8 cells x 3 arms x 3
captures. The six losses/two wins are confirmed, not only user-reported.
The updated state-equivalent excess is 28.2–28.8 us / 74.1–75.5 us at Hv32;
the Hv48/T8192 residual state saves 70.3–70.8 us against FLA W/U + state.

Hypothesis: removing P@R from the recurrent chain can repay an additional
chunk-parallel W/U materialization. The counterfactual keeps C64, V32,
256-thread state ownership and the proved H/V publication layouts. Prefix,
solve (including TF32x3) and output reuse the current implementations.
W/U preparation reuses the retained split-WU atom order and BF16 conditioning.
State becomes Vnew=U-W@BF16(H), followed by the existing gated state update.
This changes BF16 association, not real-valued algebra; the independent
numerical gate is unchanged, but raw equality to residual is NOT expected.

Scope is an explicitly selected experimental entry, full C64 chunks only.
`ALA_PPU10_PARALLEL_WU_EXPERIMENT` defaults OFF and does not replace any public
route. It adds `gdn_wy_parallel_wu_materialize` and `gdn_wy_parallel_wu_state`.
All 5 kernel durations (prefix/solve/WU/state/output) count toward the verdict.
The unchanged public path in the candidate library must still match the old
admitted public path bit-for-bit before candidate profiling.

The existing inverse allocation has 16384 BF16 elements per group. Solve
writes P in its first 4096. One W/U CTA reads its entire P into shared before
overwriting its own group with W[64,128] and U[64,128]. Distinct groups never
overlap, so no inter-CTA barrier or extra allocation is needed. Snapshots
remain a separate live allocation. Reusing packed `tile_offset(group)` for
these writes is specifically forbidden and tested as a cross-group negative.

Costs to measure: one added kernel, materialized W/U global stores and state
reads; state shared storage 46080 -> 49664 B. The expected recurrent MMA sites
per warp fall 20 -> 16, while total work grows. No lower instruction count,
occupancy or latency is promised. Reject if state improves but the complete
call does not. Preserve Hv48/T8192 as a mandatory no-regression control;
admission remains per shape, not a blanket switch.

Local SDK 2.2.1 PPU1.0 compile/link has passed. Native recurrence changed
20 -> 16 BF16 MMA sites, 1264 -> 1024 total static sites, 128 -> 98 registers,
with zero native stack in both controls. AIU sites increased 4 -> 5 and all
4 recurrent CTA barriers remain. The materializer has 32 MMA sites per warp,
126 registers and zero native stack. These are compile facts, not measured
latency, dynamic instruction savings or a spill diagnosis from ACU's separate
stack-labeled metric. Native negatives include putting the old P@R body under
the new symbol. Independent float64 recurrence, wrong-state-transpose/sign,
padded workspace ownership and incomplete inventory negatives are local gates.
Local validation: 170 Python/host tests and 4 exhaustive C++ ownership suites
passed, including the unchanged W/U, eight-warp, H and H/V layouts. At the
initial handoff the native five-stage candidate was compiled and linked,
but PPU device correctness and speed were NOT_RUN (now reported below).
No claim of an eventual win follows from those local gates.

User-operated box command (same idle ZW810/SDK/inputs as the returned bundle):

```bash
INPUT_RUN=/workspace/actlize-la-zw810-recheck-b810b6c7-20261109T083458Z-2075017 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 JOBS=4 bash tools/run_ppu_gdn_parallel_wu_box.sh
```

The script builds the optional entry, runs a native-body gate, and requires
all three numerical arms plus unchanged default RAW-BIT checks before timing
each cell. Candidate admission poisons all scratch before each of 8 replays;
no poison/fill or verification kernels are inserted in its timed subject.
It captures all 8 cells, rotates default/candidate/FLA order, and retains a
second full cycle for unresolved/noisy comparisons. It never recompiles PLA.
Upload the printed `results.tar.gz`; binaries, operands and original acurep
remain on box. `COLLECTED` means the experiment completed, not that it won.

### Round 1 device verdict (2026-11-09 UTC)

Archive `11fe5c47e70bfea09dbe645e23bd80b507c02761604634c5476f482337b202d3`,
source `3d25e7e`, candidate library
`7df5a2ae614d22b502103090bf67063b70b81b87cafeddfe2c52671f7a4adc3e`.
Same original default library/device/operands; 291 source fingerprints match
the commit. All 8 numerical cells passed 8 replays and the candidate library's
unchanged public default passed RAW-BIT checks. All 90 captured calls match
their CSV inventories/receipts; CE frequency is 1.7 GHz throughout.
The two Hv48/T8192 cells retain six repeats because default/FLA overlap.
All other cells have three. Missing-cell, omitted-W/U and tampered-duration
offline negatives are rejected. Original binary/operands/acurep stay on box;
their hashes are receipts, not local rehashes of absent files.

**REJECT this candidate for promotion on all eight cells.** Full-call ACU
medians (us; W/U and final state included):

| T | Hv | g | Default | Parallel W/U | FLA |
|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | -0.1 | 237.388 | 297.519 | 206.715 |
| 2048 | 32 | -1.0 | 236.762 | 298.431 | 206.576 |
| 2048 | 48 | -0.1 | 305.726 | 393.134 | 291.898 |
| 2048 | 48 | -1.0 | 306.202 | 393.156 | 290.095 |
| 8192 | 32 | -0.1 | 920.615 | 1107.527 | 807.855 |
| 8192 | 32 | -1.0 | 922.468 | 1105.069 | 807.837 |
| 8192 | 48 | -0.1 | 1121.067 | 1414.877 | 1124.675 |
| 8192 | 48 | -1.0 | 1120.148 | 1413.241 | 1121.950 |

The candidate loses to both controls with disjoint envelopes in every cell;
it is 19.79–28.59% slower than default. Default/FLA at Hv48/T8192 is now
UNRESOLVED, not a repeated confirmed default win. Do not erase the earlier
two wins, but do not substitute them for this paired run either.

The intended state-chain shortening **did** save time, but did not repay
materialization. Additive means for g=-0.1 (the other g has the same outcome):

| T/Hv | State saving | Added W/U | Other stage variation | Net added time |
|---|---:|---:|---:|---:|
| 2048/32 | 18.323 | 78.722 | +0.127 | +60.526 |
| 2048/48 | 20.448 | 108.755 | -0.193 | +88.114 |
| 8192/32 | 71.014 | 258.985 | +0.030 | +188.002 |
| 8192/48 | 82.297 | 379.015 | -1.733 | +294.985 |

This is not a rejection of the abstract W/U algorithm: FLA uses it and is
faster. Our execution layer is not equivalent to FLA. At T2048/Hv32/g=-0.1,
our W/U is 78.722 us versus FLA 32.045 us; state 123.909 versus 82.366 us.
The same ACU `Executed Instructions` metric is 37,087,232 versus 7,438,336
for W/U and 32,530,560 versus 10,994,176 for state. These are not summed
per-opcode counts. State registers fell 128->98, but achieved active warps
remain 14.18->14.15: the unchanged grid128 still limits work supply.
Even matching FLA W/U alone would exceed the 18.323 us saved from state;
both implementations would need further changes. Do not spend the next
round treating a materializer-only speedup as a whole-call win.

Local reproducible audit: task `actlize-la-zw810-fla-kda-20261109`,
`audit_round1.py`, `round1-upload/audit/audit.json` and `audit.md`.
No default routing changed; retain this opt-in implementation as a rejected
counterfactual, not as a promoted performance path.

## Round 2 contract: isolated single-TF32 inverse merges

Return to the original four-stage residual algorithm; **do not stack round 1**.
Change only off-diagonal solve products from the retained high/high plus two
residual TF32 products to high/high TF32. Keep FP32 diagonal solve, BF16 KKT,
FP32 accumulator/state, final BF16 inverse, chunk geometry, solve thread
selection, state/output/prefix and scratch ownership unchanged.
This deliberately changes precision; it is NOT a bit-identical optimization
or a promise to match FLA rounding bit-for-bit. FLA source's single-TF32
products motivate the experiment, not its numerical admission.

Admission remains the original independent <2% output AND final-state gate,
8 repeats, both gates, all 8 cells, nonzero asymmetric state and unchanged
default RAW-BIT control. Do not relax that criterion after results arrive.
An explicit optional C entry and OFF-by-default build option keep the old
public route intact. Native gate must find 4 rather than 12 static TF32 MMA
sites in each solve body, unchanged BF16 KKT/diagonal/order/publication and
zero native stack; restoring the old triple-product body must turn it red.
CPU screens cover full/tail triangular extents and longer recurrent sequences;
they do not substitute for the user-operated device numerical gate.

Discriminating result: the changed solve must improve complete four-kernel
ACU time, not only reduce its instruction count. Compare incumbent/candidate/
FLA under the existing serial paired capture protocol. Reject on numerical
failure or any unresolved/non-winning scope; a per-shape win may be retained
without claiming that this one edit closes the larger Hv32 state gap.

Round 2 local closure: SDK 2.2.1 PPU1.0 compile/link passes. Native 128/256
thread solves have 1496 -> 1394/1393 static sites, 12 -> 4 TF32 MMA sites,
84 -> 66 registers and zero stack. Their BF16 KKT, 120 ordered diagonal
FMAs, AIU sites, barriers and vector stores stay fixed. All 36 retained
device bodies are instruction-identical to the round-one library; only
the two W/U experimental symbols are absent and two new solve symbols appear.
191 host tests pass, including long-sequence rounding screens, wrong-index/
sign/entry negatives and the existing capture/receipt regression. The first
source-gate negative caught a normalization hole: substituting the old
product could pass a normalized comparison. The gate now first requires both
actual single-TF32 call sites; restoring either old call fails. Native
restored-triple-product and omitted TF32/BF16/store negatives also fail.
These are local proofs; device numerical/performance status is **NOT_RUN**.

The preceding NOT_RUN status describes the initial handoff. The returned
device verdict below supersedes it; the original preregistration is retained.

```bash
INPUT_RUN=/workspace/actlize-la-zw810-recheck-b810b6c7-20261109T083458Z-2075017 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 JOBS=4 bash tools/run_ppu_gdn_solve_tf32x1_box.sh
```

This reuses the same eight-cell input files and old baseline library, builds
only the OFF-by-default single-TF32 experiment, and excludes W/U from this
candidate. The exact four-kernel inventory rejects the old solve or added W/U
work, while still counting unexpected durations before marking the call
unadjudicated. Use the printed small `results.tar.gz` for the next handoff.

## Round 2 device verdict and work-first next step

Returned archive SHA256
`9961c2362ed3485d52ef52020ae0b58fcebfbcfd0aa9bc64a5964aa65a55d419`,
source `38e8e91cc5ab07b3226f294c99d8584e5ce62b8f`, candidate library SHA256
`19fb1bbd565c01bee16d95d507aee23d8b3218d5f77712f522838b7fad049531`.
All 8 cells x 8 numerical repeats and unchanged-public-default RAW-BIT gates
pass. All 90 full-call captures, source/operand/FLA receipts and raw CSV
inventories are audited; CE frequency is 1.7 GHz throughout. Missing-cell,
missing-solve and changed-timing negatives fail. Original binaries and reports
remain on box; local rehashing of those absent files is not claimed.

Complete-call ACU medians in us, including every child and final state:

| T/Hv/g | Default | TF32x1 | FLA | TF32x1 vs default | TF32x1 vs FLA |
|---|---:|---:|---:|---|---|
| 2048/32/-0.1 | 237.831 | 236.111 | 206.535 | WIN | LOSS |
| 2048/32/-1.0 | 237.725 | 234.939 | 206.717 | UNRESOLVED | LOSS |
| 2048/48/-0.1 | 306.475 | 302.318 | 288.555 | WIN | LOSS |
| 2048/48/-1.0 | 306.090 | 302.094 | 291.334 | UNRESOLVED | LOSS |
| 8192/32/-0.1 | 921.719 | 912.962 | 812.307 | WIN | LOSS |
| 8192/32/-1.0 | 921.748 | 911.412 | 817.639 | WIN | LOSS |
| 8192/48/-0.1 | 1117.636 | 1110.432 | 1126.084 | WIN | WIN |
| 8192/48/-1.0 | 1117.749 | 1110.092 | 1124.341 | WIN | WIN |

Keep the measured candidate as an opt-in result on the six resolved scopes;
do not promote the two overlapping scopes or make a blanket lower-precision
default replacement. Time falls 0.64–1.36%, not remotely the 3x suggested by
looking only at the changed TF32 products. None meets the separate 20% target.

On the user's explicit direction, the next gate is **mathematical work and
executed instruction accounting before parallelism**, not a warp/grid sweep.
See [ZW810_GDN_WORK_AUDIT.md](ZW810_GDN_WORK_AUDIT.md) for the arithmetic DAG,
same-metric full-call instruction ledger, source-proven repetitions versus
unproved codegen attribution, and an offline existing-report export command.
No core, geometry or default-routing source changes accompany this audit.

## Round 3: conversion-only native BF16 candidate

Registered before editing at parent `bd9031543171de121f9dd95faa7f617b19deb380`.
The complete work export, archive `f7826334`, identifies the software BF16
constructor's 15-operation finite-rounding core as 42.6–44.0% of the default's
executed native PC instructions. That is an instruction count, not a time
fraction. See the work audit, section 5, for the independently closed rounding
denominator and the SDK compiler-version guard that selects software.

This candidate replaces only seven scalar FP32-to-BF16 constructor sites:
one in solve, four in state and two in output. The local adapter emits
`ppu.cvt.rtte.bf16.f32`, retaining the same BF16 storage and conversion
boundaries. It does not change actlize or pretend HGCC major 2 is CUDA major 11.
The old three kernel bodies remain the source authority: an opt-in generator
extracts their complete definitions and reversibly replaces the conversions
and candidate names. Generated source/authority hashes accompany the bundle.
There is no second hand-maintained algorithm or replacement public default.

Retain the original TF32x3 solve, FP32 state/accumulators, native AIU/SWZL
delivery, layouts, thread/grid geometry, inverse pitch, gates and prefix.
Do not stack the round-2 single-TF32 candidate. The extra C ABI and CMake
option `ALA_PPU10_NATIVE_BF16_EXPERIMENT` are opt-in. Reject tails and the
out-of-scope value64 route. The numerical/performance denominator remains all
eight B1/Hk16/Hv32-or-48/T2048-or-8192/g=-0.1-or-1 cells, K=V128/C64.

### Local closure, not device admission

Local SDK 2.2.1 compiles and links the real PPU1.0 library. The native gate
counts the entire selected static bodies, not estimated dynamic executions:

| Body | Static instructions, old → candidate | Registers | Software sites → native conversions |
|---|---:|---:|---:|
| Solve, 128 threads | 1496 → 1472 | 84 → 84 | 1 → 1 |
| Solve, 256 threads | 1496 → 1471 | 84 → 84 | 1 → 1 |
| State | 1942 → 1095 | 128 → 120 | 40 → 40 |
| Output | 2094 → 1408 | 72 → 70 | 32 → 32 |

All four have zero native stack. MMA, FP32 arithmetic, AIU, shared/global
load/store and barrier counts remain unchanged. Removing a conversion, MMA
or SWZL load from the exact candidate body fails the compile gate. All 36
retained control bodies have identical native instruction sequences to the
prior local build; this does not assert whole-DSO byte identity. Scoped host
regression: 292 tests PASS. The separately recorded legacy solve-static
source-equality failure is not included or hidden in that total.

Evidence in task `actlize-la-zw810-fla-kda-20261109`:
`native-bf16-build-02.log`, `native-bf16-gate.json`,
`native-bf16-control-check.json`, `native-bf16-tests-final.log`.
No GPU was run. Native instruction semantics and performance are **NOT_RUN**;
static instruction savings are not a predicted speedup.

### Box admission and command

Before any timing, a separate conversion probe compares both retained and
native results to an independent integer round-to-nearest-even oracle:
458,752 FP32 patterns, covering every high16 pattern at six low16 boundaries
plus deterministic random inputs. This includes ties, both zeros, subnormals,
finite overflow, infinities and the retained canonical-NaN contract. A one-bit
negative must fail. A semantic difference stops the round; do not relax it.

Each full-core cell then requires eight repeated independent numerical checks,
candidate output **and final state RAW-BIT equality to the original default**,
unchanged public-default bits in the candidate library, and poisoned-scratch
replay. Only admitted cells reach the existing serial, rotated 3/6-repeat
default/candidate/FLA full-call ACU capture. Include every FLA helper/fill and
final state as before; retain unresolved/lost cells and the eight-cell
denominator. A conversion mismatch is not a performance observation.

From the task branch on the same ZW810 box as the frozen inputs:

```bash
INPUT_RUN=/workspace/actlize-la-zw810-recheck-b810b6c7-20261109T083458Z-2075017 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 JOBS=4 ACU=acu \
bash tools/run_ppu_gdn_native_bf16_box.sh
```

Use PATH `acu`, not the obsolete absolute Asight path. The runner prints the
complete comparison and `/workspace/actlize-la-gdn-native-bf16-.../results.tar.gz`.
Upload that small archive; operands, binary and original reports stay on box.
`COLLECTED` does not mean a performance win, and no outcome auto-promotes main.
