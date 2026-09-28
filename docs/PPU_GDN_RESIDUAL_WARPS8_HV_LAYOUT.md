# Paired Vnew layout on the measured H-layout

Candidate `residual-warps8-hvlayout`, control `residual-warps8-hlayout`.
The existing H path is retained verbatim. Only Vnew publication and its
paired output reader change; no automatic routing or old-kernel changes.

## Goal and cost boundary

At B1/S2048/Hk16/Hv32/K=V128/C64,g=-1, the previous paired-H capture measured
194.46647us across all4 kernels versus FLA224.35us across all7. That is
context, not this new candidate's performance. The1.5x target remains open.

Vnew still writes64B row segments at256B pitch. The previous production-plan
128B footprint model accounted for32MiB of store transactions for16MiB of
logical Vnew. This candidate instead stores private Vnew as `[group,V,T]`,
making the rows128B contiguous and matching the output's native B operand.
The hoped-for reduction is inner-hierarchy transaction amplification,
**not a reduction in logical results or compulsory bytes**.

At the priority shape, expected dynamic changes to check on box:

- Output's remaining131072 transpose matrix loads become normal loads,
  one-for-one; state's remaining K transpose loads are not changed.
- AIU staging remains112MiB for state and80MiB for output. Vnew output
  staging is64x64 before and after, so it still uses one AIU cube. Unlike
  the earlier H change, no extra copy instruction is needed.
- Logical state stores remain50MiB. A128B footprint hypothesis predicts
  KVD/L2 store transactions66→50MiB; this is **not** per-PC measured fact.
- Grid128x256(state),1024x256(output), shared45568/49408B and all
  synchronization/rounding/math remain fixed. Judge the complete call,
  not only output BC, registers, or one stage's time.

## The two lifetime/ABI seams

State's `sm.value` has two successive roles:

```
AIU.swzl input V -> old Value::offset scalar readers
  -> existing RESIDUAL_READY barrier retires all input-V readers
  -> PR BF16 result directly stored in B-oriented microcubes
  -> existing VALUES_READY barrier
  -> contiguous16B publication of private Vnew[group,V,T]
  -> existing RETIRE barrier -> next input-V AIU overwrite
```

No additional buffer, transpose, shuffle, copy or barrier is introduced.
The same BF16 cast of PR's result and the separate scaledV cast remain.
H snapshot, K/P staging, gates and FP32 initial/final state are unchanged.

The private offset is in **BF16 elements**:
`tile_offset(group) + v*Chunk + t`, with `Chunk=64` and group pitch8192.
State-slice base is `v0*64`; output-panel base is `panel*64`. The host's
opaque allocation extent is unchanged; its old tensor shape is not a
public layout promise for this scratch buffer.

Output's Vnew operand type is now separate from its final-output
exchange/publisher. The operand stages physical `[V,T]` through AIU.swzl
and reads it with normal ld.swzl. Final output remains the old `[T,V]`
exchange and canonical external stride, including GVA and row tails.
The square64x64 geometry must not be used as evidence of semantic equality:
the coordinates, physical bytes and native B fragment are tested explicitly.

## Implementation boundaries

- `wy_residual_warps8_hvlayout.cuh`: reuses the existing eight-warp
  BIntermediate producer map; adds the contiguous private Vnew publisher
  and paired AIU/SWZL output reader. H and storage types are aliases to
  the admitted H-layout, not copied definitions.
- `gdn_wy_residual_warps8_hvlayout_ppu.cu`: isolated state/output bodies and
  paired launcher. Reversing only the declared Vnew edits must reproduce
  the old H-layout bodies exactly after comment/whitespace normalization.
- Explicit API variant9/profile entry; default remains scalar residual.
  A missing candidate backend is an error, never fallback to old-H output.
- The shared runner uses the registry to select **H versus HV versus FLA**,
  not B-layout versus HV or a different warp geometry.

## Local proof

`l030_wy_residual_warps8_hvlayout` enumerates all64 valid-row tails, two
groups, four V slices, eight warps,32lanes, both output panels and every
reduction tile. The actual native C/B MMA traits and actlize matrix-load
simulator are independent of the production writer's address expression.

Fixed denominators:

| Checked domain | Count |
|---|---:|
| Original input-V reads |1048576|
| Vnew producer values |1048576|
| Contiguous16B private publication vectors |131072|
| Output native-B reader values |4194304|
| Valid public-output values |532480|

It also checks global exact-once, zero padding, untouched output tails,
interleaved two-head public strides, and a64-bit group-offset witness.
Thirteen negative controls cover stale shared layout, one-bit permutation,
old global/publication coordinates, wrong pitch/output coordinates,
missing warp/slice/group/K/tail, early input-buffer overwrite and an
accidentally transposed public-output exchange. Omitting a reduction tile
or tail is rejected by the denominator even if all remaining values match.

Twelve source/binding negatives include the old paired-output dispatch,
changed rounding and missing input-reader retirement. In particular, a
newly exported function can coexist with a Python binding still calling
variant8; the old-variant and old-native-launcher plants reject this even
though an old H-layout run would remain numerically correct. Six native negatives target
the new body only: stale transpose, scalar store, missing AIU/wait/barrier/MMA.
The native gate checks whole bodies AND actual state/panel backedges,
not a source pragma or a total mnemonic count alone.

## Local SDK2.1.1 code generation

| Property | H control | HV candidate |
|---|---:|---:|
| State registers/thread;stack |122;0|122;0|
| Output registers/thread;stack |78;0|70;0|
| State static instructions |1215|1194|
| State recurring bodies |512/526|503/517|
| Output static instructions |1418|1405|
| Output panel bodies |247/254|241/248|
| State normal/trans matrix-load sites |28/8|28/8|
| Output normal/trans matrix-load sites |52/8|60/0|
| State MMA/AIU/barrier sites |20/4/5|20/4/5|
| Output MMA/AIU/barrier sites |40/7/6|40/7/6|

Ordinary shared/global instruction widths/counts, BF16 conversions and
async completion counts are unchanged. No NCOM replacement, software
transpose, indirect register slots or new spills. Lower registers do not
prove greater occupancy: output is already limited by shared memory, and
state remains underfilled by its grid.

Full local regression passes:21/21CTest,91host contracts,7compiler-dialect
checks,45WY+61residual algebra cases and305original source controls. The
34-image WY library contains all32 old native instruction-and-operand
sequences unchanged. The15 original backend images are also audited; the
pre-existing experimental C32 stack144B remains separately classified,
while production C16 and both new kernels have zero stack.

Compile-only DSO SHA256:
`01665ad8310766f1acccf499e7b1d9ee54d12c5235e193328fa541fa78e542de`;
binding SHA256:
`8a7c301e1448cc75de5d6f9a307125d69cb6b4789251d89d2731960b56407748`.
Final clean replay authority is `final-closure.log` in the evidence directory.
Device numerics and performance are **NOT_RUN locally**.

## Same-binary box command and fixed interpretation

After pulling the published change in `GDN-QSA-sm80`:

```bash
git pull --ff-only &&
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
DEVICE=0 JOBS=16 CANDIDATE=residual-warps8-hvlayout \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Keep the working SDK path if it differs. The script performs correctness
first, then captures all4/4/7 kernels using
`/sim/eec/shared/junfu.qx/asight/bin/acu`; there is no PPUProfiler or API
event timing in this experiment. Artifacts go into a fresh `/workspace`
directory, with source/binary/input/device receipts and a final upload tar.

Require all30 scalar +30 H +30 HV device cases x8 RAW-BIT and independent
recurrence2% admission, both g=-0.1/-1. Then profile default g=-1 (weak-gate
speed is not established by the numerical gate). The control and candidate
must show their exact paired state AND output symbols, same geometry and
clocks, with no missing FLA fill kernels.

A complete-call improvement supports retaining the candidate; lower BC or
fewer registers without lower time is not a speed win. Count both changed
stages and treat prefix/solve variation separately. No automatic routing
promotion, new tolerance or changed1.5x target follows from one capture.

Source worktree: `/workspace/gdn-wy-warps8-hvlayout-20260925`.
Plan/build/logs: `/workspace/gdn-wy-warps8-hvlayout-evidence-20260925`.
