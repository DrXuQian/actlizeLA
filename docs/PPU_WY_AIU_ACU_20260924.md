# Matched AIU/SWZL: verified ACU phase comparison

Current user decision scope is **ACU time only**. AIU totals304.048 us versus
FLA222.864 us: still36.4% slower, despite a12.2% improvement over shared346.278
us. The API samples below are retained evidence, not the decision metric.

## Evidence boundary

Archive `gdn-qsa-acu-20260924T031047Z-2637570.tar.gz`, SHA256
`eb13e3799ca84347d0062fad610a0e11723e2a767ae9b1d50e619d8581459400`.
All570 regular files and the exact569-file checksum denominator verify.
No archived program/library was executed. Native reports were re-imported
with the existing local SDK2.1.1 ACU and private compatible host loader.
All13 kernels' measured per-PC counts close against the independent native
opcode totals; omitting one executed PC fails closure in every kernel.
The62 host capture/benchmark contracts also pass.

Capture helper is clean `b75b25d`; measured binaries come from clean
`8340fc76d99344221023c5872d210592598d41a5`, run
`/workspace/gdn-wy-fla-8340fc7-20260924T023404Z`:

- Binding SHA256: `9b86f961d8c6637219d33af3b388e8c9b2d353eebb87b7cba4654e8edb1a13fb`.
- Device DSO SHA256: `f52466a3447787e63f2e497482d56dc27d0517421526910808044d47ff1783cb`.

Both WY captures loaded these exact binaries. Control is
`prepare-rows-shared`/1520; candidate is `aiu-state-output`/13808. Actual
native symbols match those choices. The preceding comparison, both independent
preflights and subject receipts agree on input, outputs and precision. All
three subjects use physical UUID `019ee024-8860-091c-0000-0000007aff6d`, PPU-ZW810,
72CU, and every kernel reports1.700 GHz. The preceding properties string
contains this same UUID, closing identity despite the old metadata caveat.

Capture-time tool banners: HGGC2.2.0-dev (June3 build), ACUv2.0.2/data15000,
Torch2.9.0/runtime12.9. Do not relabel the box compiler as the local SDK2.1.1
used for compile proofs. The original build log is not in this archive;
binary identity is established by hashes, not an assumed SDK-directory label.

Workload: B1/S2048/Hk16/Hv32/K=V128/C64, BF16 inputs/output, native GVA,
FP32 WY/FLA final state, zero initial state, final state requested, no QK
normalization. Original retains its BF16 final-state contract. The capture
is g=-1.0 only; preceding full-API samples cover both g=-0.1 and-1.0.

Correctness denominator:16 distinct cases x4 non-scalar deliveries=64 distinct
admissions, each scalar raw-bit match and8 repeated calls, GVA/output-only
checks and the unchanged independent2% oracle. No duplicate case was accepted
as a missing combination. The old `delivery_ab=false` metadata omission did
not erase the exact measured masks; the archived JSON was not rewritten.

## Full public-API event samples

All7 roles x14 samples x2 gates are present. Medians, fingerprints, selected
masks and every recorded pairwise envelope verdict recompute.

| g | Shared1520 us | AIU-state us | AIU-output us | AIU-both us | FLA us |
|---|---:|---:|---:|---:|---:|
| -0.1 |354.122|332.858|338.846|319.062|485.922|
| -1.0 |354.364|332.104|339.696|318.564|492.188|

AIU-both has disjoint winning envelopes against shared, each single change,
original, scalar WY and FLA at **both gates**. Strong AIU-both range is
`[317.048,322.244]` us, shared `[352.704,356.340]`, FLA `[485.648,505.552]`.
Weak AIU-both range is `[317.588,320.936]`, shared `[353.240,359.468]`, FLA
`[481.580,604.800]` us. This is an admitted experimental full-API win at this
workload, not an automatic global/default selector promotion.

The user table was the strong-gate result; its `ailu` spelling is a transcription
typo. Raw role is `aiu`; state median is332.104, not332.100 us. The old
`subject=wy ... FLA-WINS` line compares scalar716.770 with FLA492.188 us.
The separate candidate verdicts correctly report the new arms' wins.

## ACU per-phase comparison: same device, same frequency, same binary

| Phase | Shared1520 us | AIU-both us | Matched FLA us | AIU change vs shared |
|---|---:|---:|---:|---:|
| Prepare |135.465|135.854|83.428|+0.3%|
| State recurrence |145.836|121.071|90.198|**-17.0%**|
| Output |64.976|47.123|42.748|**-27.5%**|
| Separate fills |0|0|6.491|not applicable|
| Sum of profiled kernels |346.278|304.048|222.864|**-12.2%**|

Prepare is the exact same native body and dynamic instruction stream. Its
0.389 us variation is a control observation, not a prepare regression caused
by this change. State/output save24.765/17.853 us under this profiling protocol.

FLA prepare is prefix2.801 +KKT/solve44.241 +W/U36.386 us. Its two fills
4.350+2.141=6.491 us are retained separately. Profiled sums are shared346.278,
AIU304.048, FLA222.864 us including fills. They are **diagnostic replay sums,
not full-API latency**. Do not subtract them from API samples to measure
Python/allocation/launch overhead. AIU truly improves the kernels, but FLA
still has lower GPU-stage durations; these two facts do not contradict the API win.

## What the change removed

| Dynamic instruction count | Shared state | AIU state | FLA state | Shared output | AIU output | FLA output |
|---|---:|---:|---:|---:|---:|---:|
| All opcodes |21,741,184|11,967,104|10,040,320|21,560,320|11,612,160|9,166,848|
| v.madl.i32 |3,167,744|57,344|75,264|3,833,856|278,528|737,280|
| v.mov.v2s |756,736|461,824|0|851,968|688,128|0|
| BF16 MMA |524,288|524,288|524,288|524,288|524,288|524,288|

Total instructions fall45.0%/46.1%; the AIU/FLA ratios fall to1.192x/1.267x
from2.165x/2.352x. State `v.shrl` falls1,068,032->13,824 and `v.lop3`
804,864->47,616; output `v.shrl` falls1,048,576->73,728. This is substantial
address/operand work eliminated, not less GEMM math or weaker precision.

The native bulk writer really executes: state20,480/output8,192
`vmem.aiu.ld.tsm.l0...b16.kp1` operations replace294,912/163,840 ordinary
`vmem.ld.tsm` warp executions. These instruction granularities are different:
their ratio is **not** a byte reduction. DRAM reads stay about88.3 MiB for
state and64.3 MiB for output. KVD global writes remain98/16 MiB respectively.
Achieved active warps remain7.09->7.06 (state) and37.49->37.27 (output).
Grid, threads, shared allocation and MMA work stay fixed; increased occupancy
or lower HBM work is not the observed mechanism for this improvement.

Matching AIU/SWZL does not eliminate all descriptor/address transfers.
State/output still execute461,824/688,128 `v.mov.v2s`. Matrix shared loads
also remain720,896/786,432, versus FLA655,360/458,752. Output still computes
QK and two QH/value panels separately; the delivery experiment did not change
that reuse/scheduling structure. Do not replace the matching SWZL reader
with NCOM on these bytes without a separate mapping proof.

## Waiting did not disappear with the address instructions

| Counter (same definition in each arm) | Shared state | AIU state | Shared output | AIU output |
|---|---:|---:|---:|---:|
| s.blksyn.defer executions |66,048|66,048|81,920|81,920|
| ws__warps_issue_stalled_sync.avg |15,276.18|39,655.96|29,432.63|118,576.39|
| ws__warps_issue_stalled_commit_dependency.avg |25,562.50|11,021.92|68,587.40|12,674.89|

Barrier count is unchanged, while measured waiting shifts toward barriers.
This is not solely a smaller instruction denominator: the absolute named
stall counter grows too. Bulk single-issuer arrival/completion behavior and
less arithmetic between synchronization points are possible explanations,
not a proven per-PC latency attribution. The result **does not authorize
removing barriers**; they still protect producer/consumer and union lifetimes.
It explains why45% fewer instructions must not be forecast as45% less time.
Per-issue stall ratios are not wall-time fractions and are not summed into
an E2E time equation.

## Next structural priority (user requested ACU-only reassessment)

Remaining matched-math difference versus FLA is87.675 us:

- Prepare52.426 us (**59.8%**).
- State30.874 us (35.2%).
- Output4.375 us (5.0%).

**Prepare is the next structural target; W/U publication is one measured defect.** Its unchanged
source `gdn_wy_prepare_rows_ppu.cu` still writes each BF16 accumulator element
directly to `ws.w[at]` and `ws.u[at]`. The measured524,288 `vmem.st.b16`
warp executions represent32 MiB useful W/U, but their KVD write requests are
512 MiB; including row prefixes,537,133,056 B=512.25 MiB. L2 writes are only
33,816,576 B=32.25 MiB. Thus the16x amplification is at the KVD interface,
**not HBM**, and it is not removed by the new state/output readers.

Use the existing [FLA-structured design](PPU_WY_FLA_REWRITE.md): separate
prefix/KKT+solve/W/U responsibilities and resource budgets. Current fused
prepare uses70,144 shared bytes and128 threads, while FLA W/U uses25,600
bytes and256 threads. Redesign W/U tensor ownership and vector publication
with that independent budget. Splitting also adds launches/intermediate
traffic, so compare the complete preparation sum, not just its fastest child.
The52.426 us difference is a target to explain, not a guaranteed split gain.

Keep vector-only publication as a causal control, retaining the new AIU
state/output and old path. Prove scratch liveness and MMA-output/lane-to-vector
ownership first. Do not simply reuse the old losing tiled-prepare candidate.
The diagnostic is fewer scalar stores and KVD requests closer to useful
output; its isolated latency contribution remains unmeasured.

Keep the TF32 inverse's high/high plus two residual products. Its98,304
TF32 MMAs versus FLA32,768 are an intentional accuracy cost, not free removable
overhead. After publication, prepare AIU delivery/phase separation and state
load/wait overlap can be evaluated independently; output is no longer first
priority. This analysis changes no kernel, precision contract or default route.

Reproduction (host only): `/workspace/gdn-wy-aiu-acu-analysis-20260924/`
contains `analyze_bundle.py`, `reimport_native.sh`, `summarize.py`, original
reports, verified per-PC/native exports, `parsed.json`, `phase-table.csv` and
`analysis-verified.log`. No device rerun is required to inspect these results.
