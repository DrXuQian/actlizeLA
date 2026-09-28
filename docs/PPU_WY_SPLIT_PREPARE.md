# Structural prepare: first step toward FLA1.5x (ACU only)

Parent8b64bd3; opt-in `delivery="split-prepare"` / mask30192. No default
or original reset/scan routing change. Device timing and numerics were
NOT_RUN at local handoff. The [verified uploaded result](PPU_WY_SPLIT_ACU_20260924.md)
now closes correctness: ACU304.995 ->269.306 us, versus FLA223.583 us.
Prepare improves25.51%; full profiled sum improves11.70%. The1.5x target
is **NOT MET**; weak-gate speed and production promotion are not established.

## Target fixed before measurement

Same B1/S2048/Hk16/Hv32/K=V128/C64, BF16 Q/K/V/beta/output, BF16 or FP32
natural-log gate, FP32 initial/final state, native GVA, scale1/sqrt128.
Numerics include weak/strong gates, tails and output-only. Primary ACU capture
is g=-1.0, same as the verified before; it does not establish weak-gate speed.

Metric: **FLA's sum of ALL kernel durations / our sum >=1.5**. Include
prefix, solve, W/U, state, output and any fill/helper kernels. Do not compare
one candidate stage to the whole reference or substitute API-event medians.
The previous same-device ACU sums304.04824/222.86413 us give a historical
target148.57609us. Admission uses the new same-run FLA measurement, not a
frozen old latency if the device clock/environment changes.

Current state+output already costs168.19412us. Therefore prepare alone
**cannot** reach1.5x even if its cost were zero. This bounded handoff attacks
the largest remaining excess first; recurrent-state pipelining/resource work
is still required. No numerical relaxation or speed prediction fills that gap.

## What changes

| Stage | Threads | Shared bytes | Local SDK2.1.1 registers | Stack |
|---|---:|---:|---:|---:|
| Old fused prepare (counterfactual) |128|70144|86|0|
| Prefix |64|256|32|0|
| KKT + FP32 block solve |128|49664|84|0|
| W/U |256|41472|128|0|

One new TU, `csrc/gdn_chunk/gdn_wy_split_prepare_ppu.cu`, owns the three
stages. `include/gdn_qsa/ppu/wy_split_prepare.cuh` declares their geometry,
element-unit workspace view and small arithmetic seams. The solve's complete
diagonal/gap-ordered arithmetic is checked against the old source. It retains
TF32 high/high and both residual terms; no precision-for-speed substitution.

W/U assigns32x32 to each of eight warps, retaining K order0,16,32,48 for
every accumulator. Native `AIU.swzl -> ld.swzl` supplies inverse/K/V, never an
unmatched plain/NCOM consumer. Inputs are conditioned with the same
`BF16((K * beta) * exp(prefix))` / `BF16(V * beta)` boundaries. A shared
exchange publishes complete16-byte vectors, not scalar fragment stores.
Local native code also reuses B fragments across the two row fragments.

The state/output kernel bodies and their resources remain unchanged, as do
all18 old device bodies. Python adds one exact opt-in selector; existing
masks cannot silently select it. The C ABI and input layouts do not change.

### Workspace and traffic, not free fusion

Snapshot allocation is already `[groups,128,128]` BF16. Before state begins,
each group's first4096 elements temporarily hold its row-major64x64 BF16
inverse. The rest is untouched. Launch order on ONE stream is:

`prefix -> solve writes inverse -> WU reads inverse -> state overwrites snapshots -> output reads snapshots`.

No new allocation, flag/counter or host synchronization is required. This is
an explicit lifetime alias, not permission to overlap W/U and state. Every
inverse upper-triangle entry is written zero; tail conditioning remains zero.
Shared K storage becomes solve scratch only after all K readers retire;
W/U inputs become output exchange only after its MMA readers retire.

Splitting adds8MiB inverse writes +8MiB inverse reads and rereads16MiB K at
the priority shape, plus small row metadata. Those are logical payloads, not
measured HBM misses. Conversely old W/U had32MiB useful output but512MiB
KVD-interface stores. New vector publication is intended to remove that
amplification; the subsequent device report confirms W/U KVD512 ->32 MiB,
while total prepare DRAM reads increase24.269 ->40.907 MiB. The complete
prepare sum nevertheless falls136.341 ->101.566 us. These are measurements
of the combined change, not isolated publication/splitting speedups.
Report these costs and the two added launches, including losing outcomes.

## Local admission

- Actual SDK hgcc PPU0010 compilation and CPython3.12/Torch2.9 link,21 WY
  images, all stack0.18/18 old native instruction+operand sequences identical
  to the same-SDK AIU control (no csrc/include changes5cbe8db..8b64bd3).
- Ten compiled host tests, including the actual actlize load simulator and
  MMA traits. New gate:32768 prefix rows,4194304 conditioned values,16384
  W/U output owners/order traces,49373184 inverse scratch cells. Tests do
  not claim to emulate GPU floating-point execution.
- Six new fault plants: wrong scan carry, reassociated multiplication, lost
  warp, misplaced vector, reversed reduction order and wrong scratch pitch.
- Nine source negatives cover inverse sign/gap order, unwritten upper zeros,
  scratch read pitch, stream/order, lifetime barriers and ignored selector.
- New native negatives cover lost TF32 residual, mismatched consumer,
  scalar publication, missing barrier/scan and missing cross-TU launch symbol.
- Device16-case independent2% oracle + scalar RAW-BIT,8 repeats,
  GVA expansion, nonzero state, tails and output-only remain mandatory.

Evidence: `/workspace/gdn-wy-split-prepare-evidence-20260924/`.
Native resources are local compiler facts, not profiler counters or timings.

## One box command

From GDN-QSA-sm80, after pulling `ppu-backend`:

```bash
DEVICE=0 PPU_SDK=/usr/local/PPU_SDK JOBS=16 \
  bash tools/run_ppu_wy_split_prepare_acu_box.sh
```

Set DEVICE to the intended physical card and PPU_SDK to the working SDK.
An explicit ACU path remains supported; otherwise the SDK's ACU is preferred.
The wrapper builds, performs full device admission, records same-input FLA
admission at both gates **without API timing**, then directly invokes ACU
sequentially for AIU13808, split-prepare30192 and FLA once. No PPUProfiler.
Artifacts live under `/workspace/gdn-wy-split-prepare-<sha>-<UTC>-<pid>/`;
upload the printed `acu.tar.gz`. No CSV copy/paste or manual hashes required.

Interpret the uploaded reports in three parts: old/new prepare SUM; old/new
whole-call ACU SUM and FLA ratio; remaining gap to1.5x. Verify candidate has
all five math kernels, incumbent three, and preserve all FLA helpers. A script
PASS means admission/capture completed, never that1.5x has been achieved.
