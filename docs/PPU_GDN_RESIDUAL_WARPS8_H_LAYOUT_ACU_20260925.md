# Paired H layout: measured full-call improvement

At B1/S2048/Hk16/Hv32/K=V128/C64, g=-1, PPU-ZW810 at measured1.700GHz,
the paired private H layout reduces the **complete ACU kernel sum** from
205.73177 to194.46647us:5.48% less time,1.05793x versus the same-binary
eight-warp B-layout control. FLA's complete seven-kernel sum is224.35000us:
the candidate is1.15367x FLA, **not the1.5x goal**.

This is one capture per arm, not repeated statistical speed admission.
Both changed kernels improve; prefix/solve are unchanged. Keep this candidate
as the next experimental baseline; do not change automatic routing. The
retained non-B eight-warp incumbent was not captured in this bundle.

## Identity and numerical admission

- Source: `14c726806675324578d75c807dfafef0e9accad8`; implementation41ab78e.
- Archive SHA256: `2b08d40e3f1b87a17f2b34e68ab1e1aa6cd01001d03356046aafd9568f28383b`.
- 624 regular files,623 manifest hashes and148 source snapshots verified
  against the recorded commit. Tracked source diffs are empty; two loose
  operator-created `.acurep` files are recorded, not hidden as a clean tree.
- Binding SHA256: `56d58fe77e2d5bf261ed049ee0f87d1ab36e47f3d71564ff22326e56983c8512`.
- Library SHA256: `6b21e88df91a7074eaa4542fb9c44971e7e05ebf3a37d795e1ca26802c2a193e`.
- Same input, library, physical-device UUID and loaded-library receipts;
  UUID `019ee024-8860-091c-0000-0000007aff6d`.
- 30 scalar residual +30 B-layout +30 paired-H device cases,8 repeats,
  GVA/tails/nonzero state/output-only RAW-BIT checks pass. Independent
  recurrence2% gate and g=-0.1/-1 admission also pass. **Only g=-1 is timed.**
- Actual capture tool: `/sim/eec/shared/junfu.qx/asight/bin/acu`;
  SHA256 `7d750971b45b0bb1f4367df6b8b0f37656d1be89894d40b39ceb2010f865b2b6`.
  SDK2.1.1/logical targetppu0010, selected compiler flag `-arch=ppu_10`.
  Exact compiler/FLA metadata is in the recorded JSON.

No uploaded executable, PPU kernel or benchmark was run locally. Local
work only verifies the archive, parses its reports and runs host checks.

## Complete timing, not a selected-kernel comparison

| Stage | B-layout control,us | Paired H,us | Change,us |
|---|---:|---:|---:|
| Prefix,unchanged |2.41294|2.40882|-0.00412|
| Solve,unchanged |52.65824|52.56118|-0.09706|
| Residual state,changed |104.95235|95.41235|-9.54000|
| Output,changed |45.70824|44.08412|-1.62412|
| **All4 kernels** |**205.73177**|**194.46647**|**-11.26530**|

The changed stages account for11.16412us; the other0.10118us is observed
variation in unchanged stages, not credited to the H edit.

FLA includes every captured child, including both allocation-fill kernels:

| FLA stage |us|
|---|---:|
| Cumsum |2.76588|
| BF16 fill |4.34941|
| KKT/solve |44.34765|
| W/U |35.66412|
| FP32 fill |2.25882|
| State |89.41647|
| Output |45.54765|
| **All7 kernels** |**224.35000**|

Our residual algorithm absorbs the separate W/U work. Compare residual
state95.41235 with FLA W/U+state125.08059 when grouping work; state89.41647
alone is not an equivalent denominator. The full-call comparison needs no
such regrouping. These are ACU durations, not API-event medians or an
estimate of Python/launch overhead.

## What actually changed in the generated work

The state reader and paired output reader both ran; this is not a stale
binary or an unused branch. H's shared writer, contiguous16B private global
publisher and output AIU reader are the paired contract from
[the implementation](PPU_GDN_RESIDUAL_WARPS8_H_LAYOUT.md).

| Dynamic work | State control→H | Output control→H |
|---|---:|---:|
| BF16 MMA |655360→655360|524288→524288|
| BF16 conversion |1311744→1311744|393216→393216|
| Matrix loads,normal |655360→917504|393216→655360|
| Matrix loads,transpose |524288→262144|393216→131072|
| AIU bulk-copy instructions |16384→16384|8192→10240|
| Native per-PC instruction sum |15885440→15658112|11587584→11257856|
| Shared-load instructions |1708032→1708032|983040→983040|
| Shared-store instructions |1343488→1343488|395264→395264|

Each changed stage replaces262144 transpose loads one-for-one. The output's
extra2048AIU instructions are the predicted two additional cubes perCTA;
they are included in its improved time, not omitted from the account.
State `v.mov.v2s` falls32768; output falls163840. State wait instructions
increase64512 despite lower latency; opcode counts alone are not a verdict.

All15 kernel opcode sums close against ACU's built-in per-opcode total;
omitting an executed PC fails the check. `pu__inst_executed.sum` is a
different counter scope and is retained separately in the JSON. Both
state/output PC streams bind to native instructions and MMA/load operands;
each excludes only16 trailing alignment NOPs.8source and6native negative
controls remain red, including stale output reader/dispatch.

## BC and bytes: this is not a BC-only experiment

| Counter | State control→H | Output control→H |
|---|---:|---:|
| Read BC |2883584→2621440|1572864→524288|
| Write BC |2998272→1949696|1179648→1179648|
| Total BC |5881856→4571136(-22.28%)|2752512→1703936(-38.10%)|
| KVD→TSM bytes |112→112MiB|80→80MiB|
| Ordinary global-load KVD bytes |17→17MiB|0.25→0.25MiB|
| Global-store KVD bytes |98→66MiB|16→16MiB|
| L1/KVD→L2 store bytes |98→66MiB|16→16MiB|
| L2→LLC store bytes |50→50MiB|16→16MiB|

No useful data or rounding boundary was removed. State logical stores are
H snapshots32MiB +Vnew16MiB +final FP32 H2MiB =50MiB on both arms.
The paired H path removes32MiB of **inner-hierarchy store transaction
amplification**, not32MiB of unique outputs or compulsory DRAM writes.

The production publication plan explains a concrete address difference:
old H writes64B row segments at256B row pitch; paired H writes contiguous
256B rows. A host enumeration of every production-plan lane/vector gives:

| Plane | Payload |64B footprint |128B footprint |
|---|---:|---:|---:|
| Old H |32MiB|32MiB|64MiB|
| Paired H |32MiB|32MiB|32MiB|
| Vnew,both |16MiB|16MiB|32MiB|
| Final H,both |2MiB|2MiB|2MiB|

The128B per-warp/per-store footprint sums to the observed98→66MiB.
This is an **explicit, matching coalescing/partial-line hypothesis**, not a
measurement of per-PC transactions or proof of the hardware's service phase.
The raw KVD counter still counts64B transaction units. A16B store per lane
does not by itself establish efficient cross-lane/global-line coverage.

DRAM state reads also differ84,293,120→50,735,232B, but L2→L1 reads are
identical127,935,488B and L2↔LLC demand-read bytes are nearly equal. Do not
call that a proved input-load saving: cache/write-allocation/replay effects
were not isolated. Likewise, aggregate BC counters do not locate every
remaining conflict at K or assign the9.54us state gain uniquely to BC.

## Occupancy and waits: do not optimize one number in isolation

| Property | State control→H | Output control→H |
|---|---:|---:|
| Registers/thread;stack |124→122;0|94→78;0|
| Shared bytes |45568→45568|49408→49408|
| Register-limited CTA slots |4→4|5→6|
| Shared-limited CTA slots |5→5|5→5|
| Actual active warps/CU |14.09→14.19|37.00→36.98|
| Eligible warps/WE |0.35→0.38|0.86→0.79|
| Memory dependency/issue |1.71→1.49|2.02→1.79|
| TSM dependency/issue |1.37→1.17|1.60→1.29|
| Compute dependency/issue |0.79→0.63|0.59→0.75|
| TFU-WAR/issue |0.39→0.07|0.12→0.32|
| Sync/issue |1.16→1.10|2.85→3.13|

State grid128×8warps/72CU supplies14.22warps/CU on average, almost exactly
achieved. Output remains shared-limited to5CTAs despite its extra register
slot. Neither stage gains appreciable occupancy. State improves both
memory and compute dependency; output's memory improvement coexists with
more compute/TFU-WAR and sync waiting, yet output is faster overall.
Thus the new bottleneck does not completely cancel the gain, but neither
does lower BC guarantee proportional speed.

Per-PC sample totals support those directions: state memory7803→6668,
compute3697→2758,sync5911→5045; output memory2599→2153,compute721→860,
sync3444→3663. These are **samples,not elapsed cycles**. The strongest
remaining state memory samples sit at `vldcnt`/`tsmcnt` waits; inspect their
operand/global-publication context before naming them all bank conflicts.

## Decision and next step

Keep paired H as the next experimental control, not a default routing
promotion. Confirm against the retained non-B eight-warp incumbent when
making a production selection; keep the original numerical fallback.

Next bounded candidate: **paired Vnew publication/output layout**. It still
has the same64B-at256B row shape, a16MiB payload versus32MiB128B footprint,
and the output's remaining131072 transpose loads. Prove the input-V buffer
lifetime separately from its later Vnew publication, preserve BF16 rounding,
and pair private global layout with the output reader. Unlike H, its64×64
output tile need not add a cube. This is a concrete hypothesis, not a claim
of16MiB measured per-PC savings or a promised latency improvement.

Keep K's two orientations as a separate candidate; changing one reader or
adding a duplicate K view would require its own full cost/consumer proof.
Do not conflate this with doubling warps or imposing a128-register cap.

The1.5x goal requires149.56667us:another44.89980us(23.09%) must be removed.
The next layout alone is not promised to close that gap. Solve and the
serial recurrence still need independent algorithm/precision-safe work.

## Reproducible local evidence

Archive, validated extraction, all PC exports, parsers and host footprint
probe are retained in `/workspace/gdn-warps8-hlayout-acu-analysis-20260925`.
Machine-readable report:
[`residual_warps8_hlayout_acu_20260925.json`](../dev/ppu/results/residual_warps8_hlayout_acu_20260925.json).

Replay locally,without running a PPU:

```bash
python /workspace/gdn-warps8-hlayout-acu-analysis-20260925/analyze.py --native
python /workspace/gdn-warps8-hlayout-acu-analysis-20260925/summarize.py
python dev/ppu/check_residual_warps8_hlayout.py --self-test \
  --isa /workspace/gdn-warps8-hlayout-acu-analysis-20260925/input/acu/isa.txt
```
