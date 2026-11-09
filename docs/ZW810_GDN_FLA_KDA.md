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
passed, including the unchanged W/U, eight-warp, H and H/V layouts. The native
five-stage candidate is compiled and linked; PPU device correctness and speed
are still NOT_RUN. No claim of an eventual win follows from these local gates.

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
