# WY compute-tile alternatives (opt-in, not a device speed claim)

Parent diagnosis: [delivery verdict](PPU_WY_DELIVERY_VERDICT_20260921.md).
The 98 MiB state write target matched FLA but did not close the instruction
or latency gap. This candidate changes MMA operand reuse and useful warp
work distribution, not only the epilogue. Original reset/scan routing,
scalar WY, and the previous packed-only candidates remain available.

## Separate stages and arithmetic boundary

| Stage | Change | Threads before / tiled | Shared bytes scalar / packed / tiled |
|---|---|---:|---:|
| Prepare | Reuse each inverse fragment across four W/U N16 fragments; write N64 panels through dead FP32 inverse storage. | 128 / 128 | 70,144 / 70,144 / 70,144 |
| State | Four warps own independent K64xV16 state; keep the BF16 snapshot live for W@H AIU loads, eliminating C-to-B register shuffle. Reuse each V operand across four independent K output fragments. | 64 / 128 | 37,120 / 41,216 / 49,408 |
| Output | Eight warps, two N64 panels, Q/attention operands reused across two N16 fragments; CTA-wide stores reuse the dead H panel. | 128 / 256 | 73,984 / 76,032 / 49,408 |

State shared storage **increases**, because H must remain live while U is
prefetched; do not describe this as a shared-memory reduction. The extra
warps own real independent outputs. No CTA splits an individual reduction
or adds a cross-CTA synchronization protocol. All three launch grids stay
unchanged: 1024 / 128 / 1024 for B1/S2048/Hk16/Hv32/D128.

FP32 state, C64 recurrence, BF16 cast boundaries and each output's ordered
MMA contributions stay unchanged. State's independent-output K loop is
interchanged with the time-reduction loop, but each accumulator still sees
r=0,16,32,48 in order. Output performs QH, the same gate multiplication, PV,
then the same final scale/cast; caching a row gate does not change its formula.

Prepare's prefix/solve/scaling body is shared in `gdn_wy_prepare.cuh` and is
source-identical to the corresponding 1b1f508 body (SHA256
`f31e9eab9f2d3788882f5daa71251cf20468fffbd6803dd891326ea8c1938436`).
It retains all three TF32 high/residual terms. **No plain-TF32 shortcut.**

`gdn_wy_tiles_ppu.cu` contains only the three new kernels and their launch
helpers. `gdn_wy_common.cuh` holds shared input/workspace ABI and load helpers.
`wy_tiles.cuh` owns the device/host-tested geometry and storage definitions.
The same unsigned delivery ABI is retained: low bits 1/2/4 select legacy
packing; high bits 8/16/32 select tiling. Both implementations of one stage
are rejected, not silently prioritized. Scalar/default remains zero.

## Next combination: scalar prepare + tiled state/output

The follow-up device result and exact reuse command are now recorded in
[the state/output result](PPU_WY_STATE_OUTPUT_RESULT_20260922.md). The reported
pair is 439.486 us at g=-1.0; it wins against scalar/state/FLA but remains
UNRESOLVED against all and original. The experiment design below is retained.

`delivery="tiled-state-output"` selects **mask48 = 16 | 32**. Prepare stays
scalar (neither bit1 nor bit8 set). These are existing device kernels and
the existing C ABI; this follow-up changes only the Python named
experiment, timing inventory and contracts. No C++/CUDA, numerical threshold,
default or original strong/reset routing change.

Motivation is the user-reported df90c61 strong-decay run: scalar WY 705.644 us,
tiled-state 453.986 us, tiled-all 443.362 us, FLA 485.896 us, original 424.072 us.
Prepare alone loses at 765.836 us. The isolated deltas would predict all at
497.886 us, not its observed 443.362 us: **do not subtract prepare's isolated
penalty from all to predict this combination's time**. Weak results and the
raw bundle from that run have not yet been supplied/verified locally.

The new arm prints direct envelope comparisons against scalar WY, FLA,
tiled-state, tiled-all and original. If the pair wins against all, keep scalar
prepare for this measured candidate; if it loses, retain all; if envelopes
overlap, report UNRESOLVED. Apply this separately at each gate. In every case
preserve the original strong winner and precision scope (original final state
BF16, WY/FLA FP32). No automatic selection follows from this experiment.

Local negative controls reject a wrongly forwarded mask56, omission of the
new role, reuse of the old unbalanced 14-sample cycle and an ACU capture bound
to tiled-all's numerical fingerprint instead of the exact new role. The same
16 device cases now check scalar plus five candidates; raw output/state
equality and eight repeats remain required before timing.

Follow-up local checks: 53 Python contracts and 45 algebra cases/five negatives
PASS, including a CPU-mocked complete comparison with the real admission and
sample loops. Unchanged five compiled host gates and the prior real-SDK
nine-image binary/seven-negative gate rechecked read-only. Device sources,
host geometry sources and actlize gitlink match a712a7d; no new build or
hardware execution is claimed. Evidence:
`/workspace/gdn-wy-state-output-evidence-20260922`.

## Local admission and its limits

Real SDK 2.1.1 compiles and links the original 15 device images and WY's six
old plus three new images, including Python bindings and cross-TU launchers.
New prepare/state/output registers: **160 / 232 / 98**, all zero stack.
The codegen gate requires native AIU loads, vector stores, expected MMA
bodies and no scalar BF16 output stores. New state has **zero**
`v.shuffle` instructions; scalar state has 128 static shuffle instructions.
Static code size is not a device execution count or a performance verdict.

The gate explicitly rejects spills, missing TF32 work, missing C entrypoint,
missing tiled device image, missing cross-TU host launcher, scalar-store
regression and a reintroduced shuffle. The linked new TU cannot be omitted
while the old six images make the gate green.

Five compiled host gates include:

- 9,516 vector ownership cases, all row tails, padded strides and offsets;
  27 planted word/permutation/coverage/tail failures.
- 8,192 native H producer-to-AIU-consumer reads, independently anchored by
  MMA traits and the hardware cube map, not a second producer-layout call.
- Every one of 34,816 output coordinates across WH, state update, QK, QH/PV,
  W and U (counted separately) gets the same ordered reduction contributions; gate multiplication
  is explicitly represented between QH and PV.
- 256 selector values against an independent 27-valid-combination denominator;
  eight additional missing-warp/K-offset/transpose/contribution/panel/selector/U-plane
  negative controls.

Final post-edit suite: **5/5 CTests, 48 Python contracts, 45 algebra cases/
5 numerical negatives PASS**, plus the original 305-control preservation
check and all original 15 device images. Seven new/retained binary negatives
fail as expected. Complete log: `sealed-local-r2.log` in the task directory.
These are host/codegen checks, **not device arithmetic or race admission**.
The box must still pass 16 cases, raw output AND FP32 state equality against
scalar, native/expanded GVA, tails, initial-state/output-only and eight repeats
before timing. The independent 2% recurrent-oracle gate is unchanged.

Control provenance: extracting the common prepare body changes prepare's
register allocation/codegen (packed 78 -> 82 registers; scalar stays 84).
The four scalar/packed state/output machine instruction sequences match the
preceding local build exactly. Do not call the complete control byte-identical
to 1b1f508 or subtract its historical API timings. Use the recompiled scalar
in the same new library and the same balanced experiment.

Local artifacts: `/workspace/gdn-wy-tiles-evidence-20260921`.

## One box command

```bash
git pull --ff-only
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 TILE_AB=1 DELIVERY_AB=0 SAMPLES=16 \
  bash tools/run_ppu_wy_fla_box.sh
```

Eight sequential roles: original, scalar WY, tiled prepare-only, state-only,
output-only, state+output, all-tiled, FLA. Gates -0.1 and -1.0; five warmups, 16 balanced
samples x 10 calls per arm. No concurrent timed kernels; full API includes
allocation/submission, not just one stage. The script records all samples,
identities and correctness before reporting disjoint-envelope verdicts.
Overlaps remain UNRESOLVED; terminal PASS does not mean a performance win.
The <=1.10x FLA target is unchanged. No auto-selection or routing promotion.
Without `SAMPLES`, the benchmark derives a complete two-pass order cycle
from the role inventory (16 for this family). An explicit 14 is now an error;
legacy `DELIVERY_AB` still defaults to 14 and three-role mode to 12.
Both gates and all samples are saved to `comparison.json`. The new result
key is `cases[].arms.wy-tiled-state-output`, with `delivery_mask=48` and
five `versus` entries. Its named selection is recorded, not inferred from
the equal numerical fingerprint; hardware execution remains a box check.

Optional subsequent ACU capture **reuses** the completed run and explicitly
selects the new family:

```bash
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 \
  bash tools/run_ppu_gdn_fla_acu_box.sh \
  --wy-run /workspace/<completed-comparison-run> \
  --wy-delivery tiled-state-output --gate -0.1
```

The collector binds to `wy-tiled-state-output` in that run, never to
`wy-tiled-all`, `wy-tiled-state` or old `wy-all` just because output
fingerprints happen to match. `--wy-delivery tiled-all` remains available.
Keep ACU times separate from
the full-API benchmark. Check all three stage instruction/traffic costs.

## Follow-up

The admitted tile bodies are retained. The next opt-in experiment isolates
[state address arithmetic and exact row-gate reuse](PPU_WY_STATE_ADDRESS.md)
with `STATE_AB=1`; it does not change the `TILE_AB` inventory or routing.
