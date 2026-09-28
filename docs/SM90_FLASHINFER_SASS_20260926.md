# SM90: actual SASS differences, not source-level guesses

Checkpoint: 2026-09-26. Physical H800, B1/T2048/Hqk16/Hv32/D128/C64,
BF16 Q/K/V/beta and scalar natural-log gates, FP32 final state. Forward only.
This is **not** a native PPU1.7 result. SM80/PPU1.0 is unchanged.

The closed inventory below is historical. The authorized continuation and
new S16 incumbent are in [SM90_SASS_FOLLOWUP_20260926.md](SM90_SASS_FOLLOWUP_20260926.md).

## Exact subjects

FlashInfer is pinned at `5d9f8c8d97fa53e22952ce8672f475d235f07478`, its
fastest admitted **no-CP** specialization, not its slower auto-CP wrapper.
The loaded CuTeDSL JIT object was retained in the original timing receipt.
Its complete embedded CUDA ELF is 94,096 bytes, SHA256
`2ac4802c189420aad6c016c1197cb64141848824fd7bdb015fc0617f6375e877`.
The initial truncated extraction is NOT evidence; only `fi-nocp-complete.cubin`
is disassembled. CUDA12.8 cuobjdump/nvdisasm can read the complete image.

Our matched subject is experimental S3, `1f22ac0`, not the old generic KDA
mainloop. Its actual measured shared object was downloaded and disassembled.
All native instructions of its four specializations exactly match the local
line-info cubin:31,432 static instructions, normalized instruction SHA256
`36b85b744ff7b5507d081f7e14fb27fe206bf21d338589429bc0bcce684be0df`.
Only **BF16 gate / no initial state** is used in the table below. The four
specializations must NOT be summed against one FlashInfer specialization.

Raw evidence root: `/workspace/gdn-sm90-win-20260926`;
`ours-native/{remote-extension.so,remote.sass,lineinfo.sass}`,
`fi-native/{extraction.json,fi-nocp-complete.cubin,lineinfo.sass}` and
`sass-accounting/intervals.json`. Exact-symbol/PC counter:
`dev/backends/sm90_sass_intervals.py` on branch `sm90-static-order-20260926`.
Its negative controls reject missing symbols/endpoints and mixed-symbol PCs.

## What is already the same

Both use TMA, four specialized warpgroups, resident FP32 state, WGMMA for
dense products and the same FP16 blocked inverse. Both wait on each dependent
WGMMA group; merely finding `wait_group(0)` is not evidence of an ours-only
serialization defect. The scalar auxiliary/state candidates removed the
inappropriate vector-KDA gated-TF32 work before this comparison.

The recurring complete-chunk instruction bodies have the same matrix work:

| Static PC interval | Our S3 | FlashInfer no-CP |
|---|---:|---:|
| Auxiliary PC range | `0x1b50..0x7460` | `0x3660..0x6fd0` |
| Auxiliary WGMMA / inverse HMMA |16 /14 |16 /14 |
| Auxiliary static instructions |1426 |920 |
| Auxiliary shared LDS sites |99 |21 |
| State PC range | `0xdb10..0x11ca0` | `0xcd10..0xf520` |
| State WGMMA |28 |28 |
| State static instructions |1050 |642 |
| State local LDL / STL sites |25 /14 |0 /0 |
| State MUFU sites |49 |16 |
| State scalar F2F / packed F2FP sites |32 /96 |0 /96 |
| State debug trap sites |21 |0 |

These are **static PC intervals**, not executed instruction totals or cycle
attribution. Predication, trips, first/tail paths and out-of-line wait paths
matter. In particular, a backward branch from an outlined wait block is not
another traversal of every instruction between its endpoints. No latency
percentage is inferred from these ratios.

## Four concrete differences

1. **Barrier IDs are runtime-indexed local storage in the generic C++ path.**
   S3 PCs `0x80d0` and `0x8130` load the wait/notify IDs with `LDL`, feeding
   `BAR.SYNC` at `0x81a0` and `BAR.ARV` at `0x8270`. Line info binds the
   operations to `math_order_barrier.hpp:76/94`. The IDs are just4 and5.
   FlashInfer selects fixed IDs. A stateless two-WG selector removes the
   indexed array while preserving256 participants, initial pre-arrival and
   every wait/notify. This does NOT promise that a selected ID or unrelated
   pipeline state can never spill under register pressure.

2. **Our build had CUTLASS pipeline DEBUG checks.** `sm90_pipeline.hpp`
   guards producer/consumer-role `brkpt` checks with `#ifndef NDEBUG`; the
   original `-O3` recipe did not define NDEBUG. S3 contains120 BPT sites in
   this one symbol. Release is a separate, hash-bound build experiment, not
   an excuse to remove public ABI validation or relax the numeric gate.

3. **Gate coefficients are recomputed in the state warpgroups.** O1, SK and
   state decay call `exp2f(prefix)` again; FlashInfer publishes coefficient
   channels once in its loader. Our standard exp2f also includes an underflow
   support path, whereas the reference requests fast exp2. Do not silently
   enable fast math or discard subnormal behavior to erase this difference.
   The first experiment caches the SAME exp2f values, adding1024 shared bytes
   under the existing alpha stage lifetime. Relative decay remains unchanged.

4. **NewV accumulator-to-operand conversion is scalarized.**
   `F2F.BF16.F32` occurs32 times in our middle-state interval. FlashInfer
   pairs the casts. The first hypothesis attributed these to SK residual
   subtraction; S8 explicitly paired that conversion but left all32 sites.
   It FAILED its native mechanism criterion and has no device speed claim.
   Source/PC tracing instead points to `acc_delta -> operand_delta` immediately
   after NewV WGMMA. S9 tests RNE preconversion before the existing operand
   retile. Whole-loop counts and parent raw equality remain mandatory.

## Measurement status

All timings use nsys **all GPU kernels per complete forward**,12 calls per
role and gate, same fixture, eight stable repeats and every captured output
checked. Device idle/concurrent-PID admission remains active. S3's first
strong run was rejected before timing when PID75882 was observed; only the
fresh `inverse-fi-strong-r2` run is admitted.

S3 weak/strong:191.2635/191.9865us. Paired FlashInfer no-CP:
112.767/112.737us. **Target unmet.** The earlier register-only R1 candidate
was slower (~401us) despite fewer warnings and must not be promoted alone.

S5 fixed-ID candidate:14/14 CPU-oracle PASS and parent O/state raw equality.
Weak paired nsys:190.177us[188.449,193.729] versus S3
190.9445us[189.569,192.993]. **UNRESOLVED**, not an established speedup.
This is a useful counterexample: deleting a visible local lookup did not by
itself materially close the performance gap.

Further causal tests, all medians from paired nsys forward-kernel sums:

| Candidate | Parent control | Candidate | Verdict |
|---|---:|---:|---|
| S6 explicit NDEBUG |188.5285us |181.2805us |WIN against parent only |
| S7 cache gate coefficients |180.2725us |184.2725us |LOSE |
| S9 paired NewV conversion |182.240us |182.9285us |UNRESOLVED |
| S10 separate gate producer |179.5845us |180.384us |UNRESOLVED |

S6/S7/S9/S10 each pass14/14 independent device cases AND exact parent
input/output/state hashes; repeats and all captured outputs also pass.
S8 is compile-only: its proposed seam did not remove the targeted F2F sites.
S9 did remove those sites, but that did not establish a speedup. S7 reduced
state MUFU49->16 but added loader work/shared storage and LOST. Moving gate
work to a separate warp in S10 did not establish a speedup either; therefore
the TMA-producer bottleneck hypothesis remains unproved, not a root-cause
verdict. Keep these negative results visible.

S11 tests the auxiliary repeated-read discrepancy directly. Mask-independent
in-range metadata loads plus a safe masked exponent input reduce its native
LDS99->21 and local load/store sites to zero within the auxiliary interval.
The interval shrinks1426->1243 instructions with the same16 WGMMA/14 HMMA.
S11 now passes14/14 device cases, parent O/state raw equality,8 stable
repeats and every captured forward. It establishes a material improvement:

| Paired window | Parent S6 | S11 | Fastest reference | Verdict vs reference |
|---|---:|---:|---:|---|
| FlashInfer, g=-0.1 |180.352 |148.736 |113.440 no-CP |REFERENCE-WINS |
| FlashInfer, g=-1.0 |180.5125 |148.480 |113.248 no-CP |REFERENCE-WINS |
| FlashQLA, g=-0.1 |178.416 |145.536 |165.072 auto-CP |CANDIDATE-WINS |
| FlashQLA, g=-1.0 |178.048 |145.424 |163.200 auto-CP |CANDIDATE-WINS |

Units:microseconds, nsys kernel sums. All four comparisons have disjoint
observed envelopes. Each family has its own paired window; do not subtract
the fastest sample/window of one from another. FlashQLA auto-CP additionally
has7.17-7.18us memory activity and host gaps, retained separately as requested.
S11 beats FlashQLA's fastest measured path by1.134x/1.122x, but remains
1.311x as slow as FlashInfer. **The two-library goal is NOT achieved.**
Old `[SM90 goal] beats-all-reference-paths=True` lines in the FlashQLA-only
captures mean that family ONLY. The display now explicitly names its scope;
no threshold, trace or original verdict was changed after measurement.

S12 retested cached gate coefficients on the newly cheaper auxiliary parent.
Fresh14/14 parent raw checks PASS. Weak paired nsys148.464us versus S11
146.416us, envelopes overlap. **UNRESOLVED, no promotion.** This composition
did not turn the earlier losing S7 observation into a success.

The final bounded S13 test removes recurring state beta/alpha-last protocols
that became unused when inverse ownership moved to aux. Counts are compiled
from the actual launcher expressions:alpha384 and beta128; the legacy
non-aux-owned case stays416/384. Retaining the deleted alpha consumer makes
the compile-time negative fail. All Q/K/V/QK/KK/O synchronization is kept.
The native middle-state SYNCS sites become14, matching FlashInfer's count,
but local spill sites and49 MUFU remain. All14 device cases pass the unchanged
CPU gate AND parent raw equality. Paired weak timing:146.832us
[145.312,148.032] versus S11 control147.984us[146.624,148.896].
**UNRESOLVED, no promotion.** FlashInfer no-CP is113.472us in that same window
and still wins with disjoint envelopes. Fewer synchronization sites did not
establish a full-forward improvement; no strong-regime win is inferred.

S11 also compiles all4 native CUDA SM90 bodies using the user's PPU CUTLASS
3.6.0 dependency in explicit **source-check** mode. This does not establish
native PPU1.7 execution or speed; that evidence tier remains SKIP.

The experiments remain isolated; device/timing closure is recorded in the
campaign handoff. The fastest reference remains roughly113us, not surpassed.
No production routing changes follow from compile-only improvements. Native
PPU1.7 remains SKIP because the available SDK/model cannot execute that target.

## Closed handoff and remaining gap

The bounded inventory is closed: R1 and S1-S13, with S4 compile-only and S8
rejected at native-mechanism admission. S11 is the best fully confirmed
both-gate candidate, on branch `sm90-aux-metadata-20260926` (kernel `457267e`,
report-scope follow-up `d3cf13a`). All experimental branches are retained and
pushed; main production routing and SM80 code are unchanged. This is an
experimental result, not a production promotion or a claim over all shapes.

The remaining state-side differences are real, but not yet a causal latency
decomposition: spills/local operations, coefficient evaluation/underflow
handling and accumulator-to-operand conversion/lifetimes. R1, S5, S7, S9,
S10, S12 and S13 show why reducing a selected opcode count alone is not enough.
The next bounded experiment should compare state accumulator/operand live
ranges against the reference and remove an actual spill lifetime while keeping
the gate arithmetic unchanged. Do not retry register budgets or cache-exp2 in
isolation, silently enable fast math, or delete required dependency waits.

Machine-readable ledger: `dev/backends/sm90_sass_campaign_20260926.json`,
20 admitted captures /1,416 complete forwards. Independent 14-case admission
and repeated correctness are additional to that timing denominator. Raw
remote campaign archive (sources, builds, cases, nsys, SQLite, JIT objects,
receipts, and explicitly rejected attempts):
`/workspace/gdn-sm90-win-20260926/remote-evidence-final.tar.gz`, SHA256
`79cd3c8e9bc87a2b75731da88a5b3bb82b229d1869575f50aeb8a7cffeccbaa4`.
The rejected idle capture is present for audit, never counted as a valid run.
The exact-SASS local evidence and pinned earlier reference artifacts remain at
the paths given above. No more GPU jobs are required for this handoff.
