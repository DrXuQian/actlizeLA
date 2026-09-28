# Closing the remaining SM90 GDN losses

Checkpoint:2026-09-27 04:18:49 UTC. Physical H800 PCIe,114SMs,CUDA12.8.93.
The fixed14-workload inventory and both gate regimes are unchanged. This is
not a native PPU1.7 result. No default routing, SM80, reference, clock, power
or numerical-tolerance change. **Expanded speed goal remains unmet.**

## S52: V64 plus independently live O2/KV operands

Source2d5e78d (kernel549ee4c), parent2b51b2f/S38. Apply the
[S50 overlap](SM90_PAIRED_STATE_20260927.md) to one state WG, not the old
two-WG geometry. Keep loader24/state192/aux232,384threads,152576B shared.
Actual delta map4096/4096 and two negatives pass. Actual state/output map
16384/8192,96 shape-stride maps and four negatives pass. The one-state
data-ring checker terminates for every reachable reduced state at1..8chunks;
this is not a memory-order proof.

All four native bodies keep matrix/TMA/publication work and four paired
O2/KV epochs, with distinct live operand registers. Old-body/missing-matrix/
serialized-WGMMA negatives fail. Initial-state spills increase; the candidate
is not a blanket resource improvement. PPU CUTLASS3.6 CUDA source-check
passes; unavailable native PPU1.7 SDK/model is **SKIP**.

All14 independent CPU cases, parent fingerprints, two extreme raw-byte
stress pairs, eight direct repeats and every graph/profiler output pass.
Eight graph cells give7parent wins/1unresolved. B2 remains much slower than
the retained V128 path, so V64 must not become an unconditional selector.

Eight registered nsys captures,528 complete forwards, have all been
re-extracted locally from SQLite to byte-identical result JSON. Units are
microseconds, **sum of every GPU kernel in one complete forward**, not graph
screen time or Python/API latency:

| Workload | Gate | Reference | Paired S38 | S52 | Fastest reference | S52 vs fastest |
|---|---:|---|---:|---:|---:|---|
| B1/T8192/Hv32 | -.1 | FlashInfer |361.4865|351.5185|354.654 auto|UNRESOLVED|
| B1/T8192/Hv32 | -1 | FlashInfer |361.903|352.559|353.439 auto|UNRESOLVED|
| B1/T8192/Hv32 | -.1 | FlashQLA |362.160|351.839|382.4955 auto|WIN|
| B1/T8192/Hv32 | -1 | FlashQLA |361.9515|351.7275|375.024 auto|WIN|
| B1/T2048/Hv16 | -.1 | FlashInfer |94.544|91.904|93.360 auto|UNRESOLVED|
| B1/T2048/Hv16 | -1 | FlashInfer |94.752|91.888|92.752 auto|UNRESOLVED|
| B1/T2048/Hv16 | -.1 | FlashQLA |93.536|91.792|92.3195 no-CP|UNRESOLVED|
| B1/T2048/Hv16 | -1 | FlashQLA |93.696|91.712|92.096 no-CP|UNRESOLVED|

Median-only reference leads do **not** pass the unchanged disjoint-range
criterion. E.g. T8192 weak S52[348.958,352.606] overlaps fastest
FI[351.679,357.951]. Strong S52[346.079,353.024] overlaps
FI[346.368,356.671]. Keep these as UNRESOLVED, not wins.

Evidence:/workspace/gdn-sm90-v64-paired-tail-20260927/s52-nsys.
Source and binary identities are in each receipt. Local native image SHA:
553b77feb975bce16201a669e40847c0f390c52abd1827a8be2a5e80c55abc77.

Retaining old winners plus only these confirmed per-cell improvements gives
FI16wins/8losses/4unresolved; QLA25wins/0losses/2unresolved/1numeric-invalid.
This is a composite best-so-far ledger across bound epochs, **not** a newly
rerun full matrix or permission to ship an automatic selector. The remaining
FI losses are B2/B4/Hv64 in both gate regimes. The known QLA weak-GVA1
numeric failure remains visible under the unchanged2% gate.

## S51: last inverse partials local to two row-owning warps

Source36c007d, independent S24 parent. The same last32->64 inverse level
keeps its two K16 FP32 products, separate FP16 casts and FP16 high+low sum.
It does not contract them into one differently-rounded K32 product.
Two row-owning warps compute twice the per-warp last-level work; CTA useful
HMMA count remains32. One input-C alias barrier stays, the cross-warp partial
publication barrier and exchange disappear. Caller publication remains.

Actual DC/output1024/1024 maps and wrong-half/omitted-warp/contraction negatives
pass. Four native bodies preserve state work and async protocols, remove two
barrier/four partial-store sites across full/tail clones, with noC7512.
14CPU parent-raw+2stress and eight-screen repeated/captured outputs pass.
Three narrow S24 graph wins/five unresolved are not reference speed admission.
S50 remains the confirmed B2 improvement; no S51 reference win is claimed.

## S55: local inverse on V64 closes four FI cells

Source6317ea4 composes the exact S51 inverse schedule with S52. Native math,
data ownership and rounding remain unchanged; all14 CPU/parent-RAW cases,
two extreme raw stresses,8direct repeats and every captured result pass.
Eight nsys captures/528forwards are independently re-extracted locally to
byte-identical JSON, including all samples, counts and verdicts. These are
full-forward kernel sums, not the faster-looking graph screen:

| Workload | Gate | Reference | S52 | S55 | Fastest reference | S55 verdict |
|---|---:|---|---:|---:|---:|---|
| T8192 | -.1 | FI |351.515|344.938|354.987 auto|WIN|
| T8192 | -1 | FI |349.256|340.920|352.585 auto|WIN|
| T8192 | -.1 | QLA |351.719|344.887|382.552 auto|WIN|
| T8192 | -1 | QLA |351.174|343.9255|373.893 auto|WIN|
| Hv16 | -.1 | FI |92.4175|91.314|93.2015 auto|WIN|
| Hv16 | -1 | FI |92.7215|91.5855|92.7695 auto|WIN|
| Hv16 | -.1 | QLA |91.409|90.817|92.225 no-CP|UNRESOLVED|
| Hv16 | -1 | QLA |91.473|91.249|91.937 no-CP|UNRESOLVED|

Four previously unresolved FI cells now pass the same disjoint-range rule.
This gives a composite retained FI20W/8L/0U, QLA25W/0L/2U/1numeric-invalid,
not a newly rerun entire matrix. The two small QLA gaps still overlap;
do not call their lower medians wins. B2/B4/Hv64 remain FI losses. No default
promotion: S55 has reference timing only for these two workloads, although
its broader numerical gate passes. Evidence:
/workspace/gdn-sm90-v64-inverse-tail-20260927/s55-nsys.
Native CUDA SHA0884b7134999088cd0966b21cd43f7fae5e41a98d867a9ecb0187aabb6bf5862.
PPU3.6CUDA source-check PASS; nativePPU1.7 remains SKIP, not an H800 inference.

## Bounded followups, not presumed additive wins

- S53 source8e89c8a: exactly compose S51 with S50 on V128. Four-arm,
  six-cell B2/B4/Hv64-GVA4 screen includes both single-change controls.
  Native/map gates and device14CPUparentraw+2stress pass. Six graph cells:
  three parent wins/three unresolved, no new best. Reject as a speed candidate.
- S54 sourcefa0dfa7: only remove optional alternating state issue on S50;
  every S50 data/lifetime source is bound unchanged. Actual physical H/O
  owners are disjoint in all four types; old inverse-owned-by-state retains
  ordering. Native33/41 issue sites become zero with all matrix/TMA/async/data
  families preserved. Device14CPUparentraw+2stress pass. Six graph cells:
  three parent wins/three unresolved. Reject as a speed candidate; removing
  a legal ordering constraint did not improve this complete forward.
- S55 source6317ea4: exactly compose S51 with S52 V64/aux232. Native/map
  gates and fixed14CPUparentraw+2stress pass. Four-cell T8192/Hv16 screen:
  three parent wins/one unresolved. Eight nsys captures closed above.
- S56 parent170f34f/S50: skip warp-uniform empty causal sectors in auxiliary
  QK/KK epilogue, preserving every live arithmetic expression. Actual-layout
  1048576-cell coverage and3negatives pass. CUDA12.8 if-converts the full
  chunk into22 genuinely predicated EX2 sites per body; this suppresses SFU
  evaluation, not necessarily fetch/issue slots. Four bodies preserve all
  matrix/TMA/barrier families with noC7512;4native negatives pass. Fixed14CPU/
  parentRAW,2extreme stresses and repeated/captured outputs pass. Six high-head
  screens all lose: B2~122..123 vs112us; B4~257..262 vs237..242us; Hv64~123vs112us.
  Reject. Less SFU evaluation did not shorten the complete pipeline; no claim
  that the saved issue slots were free or that one spill explains all cost.
  No approximate exponent/precision relaxation; no reference capture needed.
- S57 parentS50: separately publish completed inverse KK before acquiring the
  QK output slot. This explicitly retries S20's old losing axis after S50
  extended QK consumption through KV completion. Register pressure and native
  ordering were checked again. Actual4type offsets, native full/tail and
  outlined retry paths and all numerical/replay gates pass. Six high-head
  screens are all UNRESOLVED with slightly slower medians; no promotion.
- S58 source37f0975/kernelcf4ee8a, independent S50 parent: retire O2 with
  wait1, publish output/release QK while KV may continue, then retain wait0
  before H/input reuse. The [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
  guarantees completion of the older group, not an arbitrary completed one.
  Four actual native bodies preserve paired operands/math/data with4added
  wait1 sites/body and all finalwait0. Source/native negatives and actual
  8192operand map pass;14CPUparentRAW+2stress pass. Six high-head screens all
  UNRESOLVED with slightly slower medians. Reject as a speed candidate; no
  reference promotion. Earlier O publication did not improve complete forward.

S59/6ecd300 is rejected before device testing: the compiler if-converted the
full-chunk branches despite inline `bra.uni`. Tail branches alone do not
admit the intended experiment. No speed result is claimed.

S60/37e7d13 changes only that dispatch to indexed uniform branches. All four
native bodies retain eight real tuple branches; exact target-table bytes and
hoisted definitions are checked, with five negative controls. The actual
1,572,874-input seam is byte-identical to standard exp2f, including underflow;
197 producer fixtures/all32lanes pass. Full numerical/replay admission passes.
But six high-head graph screens give four parent losses/two unresolved:
B2 114.982/113.468 versusS50 112.822/111.438us, B4 249.140/244.520 versus
242.592/238.120us, Hv64 114.020/114.804 versus113.596/112.704us. Rejected;
real instruction suppression still is not a full-forward speedup.

S61/9f9c2e7 gives each next8->16 inverse-merge warp its two8x8 diagonals,
replacing the first cross-warp publication by warp-local synchronization.
Actual4096-cell physical ownership/four negatives pass. All four native
bodies remove two CTA barriers and preserve matrix work. The source-bound
PTX warp fence remains; native compilation elides its explicit opcode at
this straight-line seam. Stack24/112B versus16/104B is an explicit cost.
Numerical/replay admission passes, but all six high-head screens overlap
the S50 parent and have slightly slower medians. Rejected. Fewer barriers
alone did not improve the pipeline. No reference nsys capture for S60/S61.

S62 is independently registered on S50: issue next-chunk KK before current
inverse, retain its accumulator and K slot through next QK completion. All
1..8-chunk source-bound ring interleavings terminate; making K single-stage
or omitting final publication is rejected. Four native bodies keep KK live
across14inverse HMMA sites with disjoint registers and no premature retirement.
Its extra liveness increases stack to88/152B. All14CPUparentRAW+2stress,
8repeats and captured outputs pass. Six screens all lose: B2~154vs112us,
B4~318vs241..243us, Hv64~151vs113us. Reject; no reference capture. The native
overlap is real, but its added liveness, spill and earlier next-K wait are
costs, not proof of which single mechanism caused the loss. Local/remote
native instruction streams match SHAe1a8e225ecc1e4330969df87894134c720ce34e2f532e4e56a05806897ac3e2a.

The next separately registered inventory varies QK/KK result-ring depths:
(2,2) control, S63(1,2), S64(2,1), S65(1,1). These were not axes in the prior
32-cell input/metadata-stage sweep. Actual compiled layout checks caught that
KK storage used QK's depth while both were fixed2; no existing unequal-depth
runtime defect is claimed. Asymmetric configurations now independently bind
storage and pipeline. The wrong-QK-depth negative fails all4actual types.
Default2/2 reproduces every S50 native instruction/operand. All4tuples pass
native/progress gates. All three changed candidates pass14CPUparentRAW,
two extreme stresses and repeated/captured outputs. Eighteen screens close:
S63 three parent wins/three unresolved, S64 the same, S65 five parent wins/
one unresolved. No candidate wins; retain2/2. Smaller result rings also reduce
producer lead and are not a free scheduling simplification. Raw results are
local; no reference capture or promotion for these nonfinalists.

S66/1ac05a3 retains the original two-warp8x8 inverse ownership through the
16x16 and32x32 subtrees, leaving the final cross-warp publication intact.
Actual6144-cell maps/four negatives, source-bound PTX and native gates pass.
Four native bodies remove four CTA barrier sites but execute the middle-level
work on two rather than four warps. Stack8/104B is not a speed verdict.
All14CPUparentRAW+2stress/replay gates pass. Six high-head screens produce
four parent wins/two unresolved; every candidate median is slower. Reject.
Evidence:/workspace/gdn-sm90-local-inverse-prefix-20260927. Native instruction
SHA791b29629877095185bfb1210729325f4e803c06687bded4a4dd001452149fd1.
PPU3.6CUDA source-check PASS; nativePPU17 still SKIP.

S67/3eb5d75 (kernel1fd8c41) tests a structural interaction rather than
another synchronization tweak: whole V128 on one state WG, existing S42
shared BF16 H, with O1 delayed until NewV's BF16 conversion. No duplicated
QK/KK/inverse or additional public kernel. Source-only movement and bounded
one-state progress pass, but all four CUDA bodies still emit C7512.
Actual192/256HGMMA sites have192/256completion waits; stack568/704B exceeds
the predeclared104B ceiling. Reject before GPU; no measured slowdown or
numerical admission is claimed. Moving one source lifetime does not prove
the compiler kept all other large fragments disjoint. Evidence:
/workspace/gdn-sm90-single-state-late-output-20260927/native-rejection.json.

Each has a pre-edit plan, fixed deadline and independent worktree under
/workspace. GPU work is sequential and DeviceWatch-gated. H800 remains on.
