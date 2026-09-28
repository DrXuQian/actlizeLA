# Residual ACU verdict and remaining FLA gap

Source `1a843eb8d6f2f9fc5788bc447c1a3cbac3f806ea`; forward only. This is
analysis of the uploaded run, not a new kernel change or default-route promotion.

## Scope and evidence

- B1/S2048/Hk16/Hv32/K=V128/C64, g=-1.0, zero initial state, final state returned.
- Same PPU-ZW81072CU, UUID `019ee024-8860-091c-0000-0000007aff6d`;
  every captured kernel reports1.700GHz. Box hgcc2.1.1-a5c56e.
- Capture tool is the requested `/sim/eec/shared/junfu.qx/asight/bin/acu`,
  SHA256 `7d750971b45b0bb1f4367df6b8b0f37656d1be89894d40b39ceb2010f865b2b6`.
- Archive SHA256 `d1f0c05aacf91bb01005475adcb0f8a334f72fe9644f6483a2e7a5843d001151`:
  591 files/590 hashes verify;115 archived source files match the source commit.
  Tracked source diffs are empty. One untracked `wy-g-1.0.report.acurep` is
  recorded; the worktree is not literally empty, but no source change is hidden.
- Both ours arms load the same extension and library. Library SHA256
  `a6eb012526b7bf307af0f75e58a195931f8f47bee0bb87068f0dc736cb961ada`;
  extension `6939e8eb7726b0040e0744476bc431f87c6ba42a258f586477e8b4e9a13a43d4`.
-30 residual device cases x8 repeats pass the unchanged2% independent oracle;
  target g=-0.1/-1 both numerically admitted. Old WY16 cases and delivery
  RAW-BIT checks also pass. Residual is deliberately a different BF16 association.
  Only g=-1 is profiled here: no weak-gate speed claim.
- All5/4/7 kernels are included. No public-API timing is used. Each arm has one
  full capture: descriptive paired evidence, not a repeated-profile confidence bound.

Analysis artifacts and replay:
`/workspace/gdn-residual-acu-analysis-20260924`, `python analyze.py --native`.
Native reports were re-imported on the host with the available local SDK parser,
not reprofiled with a substitute tool. Per-PC sums agree with the opcode metric
for all16 kernels; omission of an executed PC is rejected. `pu__inst_executed`
and enumerated-opcode totals are different counters and are not silently equated.
The site export contains conflicting duplicate maximum-device-attribute entries;
both are preserved and excluded from numerical conclusions. The older parser's
missing KVD byte metric is not invented: actual transaction counts are checked.

## Complete kernel times (microseconds)

| Stage | Previous state-pipeline | Residual | FLA |
|---|---:|---:|---:|
| Prefix |2.59647|2.44706|2.69824|
| KKT + inverse solve |52.68294|52.69412|44.88118|
| W/U materialization |47.64353|absent|36.20059|
| State |107.27824|129.64353|90.27529|
| Output |45.02529|45.79941|43.77824|
| BF16 fill |0|0|4.27882|
| FP32 fill |0|0|2.19647|
| **All kernels** |**255.22647**|**230.58412**|**224.30883**|

Residual saves24.64235us (9.655%) over the previous same-binary path, but remains
2.798% slower than FLA. FLA1.5x requires149.53922us: another81.04490us /35.148%
reduction from residual. Target **NOT MET**; no selector change.

The reassociation deletes47.64353us of W/U but adds22.36529us to state. The
remaining0.63589us is variation in unchanged prefix/solve/output, not a source gain.
Do not compare residual state with FLA state alone: residual includes P@R, while
FLA materializes W/U separately. Like-scope combined WU+state is129.64353 vs
126.47588us, a3.16765us gap, not39.36824us of pure implementation overhead.

The net residual-minus-FLA ledger is:
prefix-0.25118 + solve7.81294 + combined-state3.16765 + output2.02117
- FLA fills6.47529 = **6.27529us**.

## Solve: numerical algorithm and execution organization both differ

Actual launch dimensions: ours1024 CTAs x128threads; FLA1024 x32threads.
Both do the same ten lower-triangular BF16 KKT block products (81920 MMA).
The block inverse merge precision is **not the same**:

- Ours: `include/gdn_qsa/ppu/wy_mma.cuh::tf32_product` computes
  low(A)*high(B) + high(A)*low(B) + high(A)*high(B), with conversion/residual
  preparation. Sixteen16x16 products per group yield98304 TF32 MMA.
- Captured FLA `ops/gated_delta_rule/chunk_fwd.py` uses
  `SOLVE_TRIL_DOT_PRECISION='tf32'`, single-term products:32768 TF32 MMA.

This is exactly3x **for TF32 merge MMA**, not3x solve time or all math. Our
extra terms are a precision choice, not a demonstrated compatibility-emulation
bug. Removing them would require a separate numerical experiment/contract,
not a silent delivery optimization or a relaxed oracle.

| Solve metric | Ours | FLA |
|---|---:|---:|
| Enumerated dynamic instructions |14,767,104|10,306,560|
| `s.wait` |1,637,376|507,904|
| `v.mov.b32` + `s.mov.b32` |1,711,104|454,656|
| `tsm.ld.b32` |811,008|180,224|
| TF32 MMA |98,304|32,768|
| Registers/thread |84|256|
| Shared bytes/CTA |49,664|6,144|
| Achieved active warps/CU |18.99|13.67|
| Average active lanes/warp metric |12.33|23.24|
| Stack bytes/thread |0|8|

FLA retains its ten blocks as distributed register tensors and uses warp
reductions/shuffles in diagonal substitution. Ours publishes FP32 lower/inverse
matrices to shared, solves four diagonal blocks with four warps, then runs
progressively fewer warps at off-diagonal gaps. KKT ownership also gives the
four warps4/3/2/1 products. More threads do not mean more useful parallel work.
Actual occupancy is higher on our side: **low occupancy alone is not an
explanation of this solve gap**. FLA also has an8B stack, so do not describe
it as a zero-spill ideal or blindly copy its256-register budget.

### What the observed wait/move instructions actually do

The native diagonal region between the key-reader-retirement barrier and the
inverse-publication barrier has612 static instructions and5,013,504 dynamic
instructions. Within it:196608 `v.mov ... ivreg` reads and466944 `pipe_flush`
waits. A representative generated loop is:

```
s.wait pipe_flush
s.add / s.min / s.bfi x0       // select register-array entry
v.mov.b32 ..., ivreg          // column[k]
tsm.ld.b32 ...                // lower coefficient
s.add / s.cmp                // loop induction
s.wait tsmcnt(0)
v.fma.f32 ...
s.cbr.nz ...                 // backedge
```

The source has `CUTE_UNROLL`, but the actual body still contains inner
backedges and indexed register reads. A pragma is not proof of static register
selection. This region is about34% of our enumerated solve instructions;
it is **not** a measured34% wall-time share.

Whole solve wait decomposition:704512 `pipe_flush`,907264 `tsmcnt(*)`, and
25600 other waits. `s.wait` is not synonymous with a CTA barrier. Ours executes
28672 `s.blksyn.defer`, FLA308224, yet ours has the greater sync-stall ratio;
counts alone cannot establish how long participants wait.

Of the1,711,104 ordinary scalar/vector moves,1,479,680 load literal operands;
196608 of the remainder read `ivreg`. These are not all software unswizzling.
The separate102400 `v.mov.v2s` need operand-use attribution; do not relabel
them collectively as bad SWZL. The existing AIU.swzl/ld.swzl pair stays intact.

## State and output: where the1.5x target still lives

Residual state is56.22% of our total. Its32 sequential chunks now each perform
KH -> BF16 residual -> P@R -> BF16 scaled value -> state update, with shared
handoffs. Deleting W/U removes parallel work but lengthens this dependency chain.
Compared with our pipeline control, state MMA increases524288->655360 and
dynamic instructions11,819,648->14,680,704; latency increases20.85%.
Combined WU+state work decreases, explaining the net gain without claiming
that the serial state kernel became faster.

State launch/parallelism is not uniquely worse than FLA: both use128 CTAs x128
threads, four independent V32 slices/head, about7.1 active warps/CU. Source
inspection shows residual currently waits for initial inputs before KH and
has conservative shared handoffs; it has not inherited all control-path
prefetch overlap. Improving the recurrence's register lifetimes and delivery
overlap is an experiment, not a promise that its remaining39us versus FLA
state alone is removable overhead.

The grid-limited occupancy ceiling is128/72*4=7.111warps/CU, matching measured
7.09 (FLA7.10), versus resource-allowed16warps/CU. Residual tensor-pipe active
is11.053% of elapsed peak (11.508% of active peak), distinct from achieved
occupancy11.07%. This makes finer independent V tiling a meaningful experiment,
not proof it wins: V32->V16 with a re-proved128-thread layout could give256CTAs,
but duplicates K/P delivery across twice as many CTAs. Doubling grid without
changing ownership is wrong; splitting dependent chunks is not an independent
work redistribution.

Memory dependency does not establish bank conflicts as the cause. Actual
shared bank-conflict counters: residual6,930,432 (load2,883,584/store4,046,848),
FLA state8,550,400 (load3,815,424/store4,734,976). Different native access paths
and work scopes prevent a causal conclusion from those aggregates. A conflict
can stall a dependency without saturating the shared pipe; conversely a larger
aggregate conflict count does not establish a slower call. Localize PCs and
prove the actual bank mapping before modifying the AIU/SWZL contract.

The1,310,720 executed FP32->BF16 conversion instructions close an exact value
ledger (warp instructions,32 valid lanes at these full-tile sites):
524288 for H snapshots,262144 for residual,262144 for Vnew and262144 for scaledV.
Old WY/FLA state have1048576 FP32->BF16 conversions; residual adds the262144
residual conversions required by its declared BF16 P@R operands. These are
precision boundaries, not unexplained duplicate extraction. They may admit
packing or cheaper register-to-operand delivery, but cannot be silently removed
or changed to scaled(BF16(Vnew)) in place of BF16(scale*FP32(Vnew)). The measured
1,310,720 scalar BF16 shared stores expose the associated materialization cost.

Output has equal524288 BF16 MMA, but11,587,584 vs9,166,848 dynamic instructions;
45.79941 vs43.77824us. Our higher achieved occupancy37.44 vs22.69warps/CU does
not eliminate descriptor/address/wait overhead. This gap is real but only2.02us
in this capture; it is not the first lever for an81us target reduction.

## Thread dimension is not tile dimension

| Stage | Ours blockDim.x | FLA blockDim.x |
|---|---:|---:|
| prefix |64|32|
| solve |128|32|
| state |128|128|
| output |256|256|

Our C64 is64 tokens/chunk. FLA's `h_blockdim64` names64-wide K-state panels,
not64 CUDA threads. Its selected state has V32, K128 covered as two K64 panels.
Our state CTA owns K128 x V32, four warps each K64 x V16 via four16x16 fragments.

## Next experiments, not implemented by this analysis

1. Solve's diagonal register delivery is the most isolated like-scope gap: make the intended static column
   selection real, retain arithmetic/order, check native `ivreg`/loop/flush
   reduction and resource cost before device timing. Then evaluate smaller
   cooperative groups/register-resident block organization with explicit mapping.
2. If trying FLA's single TF32 merge, give it a separate precision identity and
   independent weak/strong/tail/initial-state gate. Never fold it into step1.
3. Residual state is the largest total-time target. Retain its measured net gain
   while shortening serial handoffs / overlapping inputs. Split independent
   V-slices as a separate geometry experiment, not combined with pipeline or
   precision changes. This and solve must both improve substantially for1.5x;
   merely matching FLA's solve would yield222.77118us, nowhere near149.53922us.
4. Output last for the current workload. Reprofile all children and report
   losses as well as gains. Keep original reset/scan and old WY controls.
