# V32 residual with eight CTA-local warps

Device follow-up: [2026-09-24 ACU verdict](PPU_GDN_RESIDUAL_WARPS8_ACU_20260924.md).
RAW-BIT admission passes; state 127.74 -> 104.81 us, all-kernel sum
228.64 -> 207.00 us versus FLA 226.97 us. Registers162/stack0 confirmed,
active warps7.07 -> 14.12, global staging112MiB unchanged; shared matrix
reads+28.57% and BC+15.13%. Keep as experimental control, routing unchanged.
The local handoff below is historical; its device-pending statements are
superseded by that report, not its original contracts or predicted budgets.

Parent c537f42. This opt-in experiment retains the V32 CTA tile, grid128,
shared layout, BF16 boundaries, FP32 state, arithmetic/K order, and existing
barriers. Only independent output-row ownership changes4->8 warps. It does
not include B-layout, V16, a new prefetch schedule, or auto-routing changes.
Priority B1/S2048/Hk16/Hv32/K=V128/C64; g=-1 capture, both-.1/-1 admission.

## Two different kinds of shared traffic

The earlier proposal promised no duplicated **global-to-shared K/P/V staging**,
not unchanged **shared-to-register operand reads**. They must not both be
called simply "shared traffic".

| Per V32 CTA / chunk, source budget | 4 warps | 8 warps |
|---|---:|---:|
| K/P/V global-to-shared input | 28 KiB | 28 KiB |
| KH shared matrix loads | 96 | 128 |
| P@residual shared matrix loads | 48 | 64 |
| K-transpose@scaled-V shared matrix loads | 80 | 96 |
| Total shared matrix loads | 224 | 288 |
| BF16 MMA instructions across all warps | 160 | 160 |

K/P operand loads remain unchanged. Additional H/residual/scaled-V reads
come from splitting the output fragments that formerly reused one warp's
loaded B registers across two fragments. Across the priority workload,
global input payload remains112MiB, whereas matrix loads917,504->1,179,648
(+28.57%). These are source/native-inventory-derived budgets, not measured
bank conflicts or transaction bytes. There is no already-proven alternative
that doubles compute warps while keeping both traffic directions unchanged.

The experiment asks whether more independent resident warps and shorter
per-thread work offset that known cost. Occupancy alone is not acceptance.

## Production ownership and lifetime contract

The new shared header is used by both the real kernel and host proof:

```text
column(w)      = (w % 2)*16
state_row(w,f) = (w / 2)*32 + f*16   f=0,1
value_row(w)   = (w / 2)*16
```

This splits output rows, never reduction K. Each output retains its ascending
K16 sequence and every BF16 boundary. Snapshot/residual/scaled/Vnew writers
remain unique. Global snapshot/Vnew/final-state publication uses the same
StateVectorPlan with256 threads; retaining its old128-thread iteration
stride is a tested error. K/P/V input descriptors and single-issuer AIU
copies do not change. All8 warps participate in the existing full-CTA
publication and retirement barriers.

The new TU is source-canonical-equal to residual after symbol/header
renaming only; the actual owner traits carry the geometry change. This
deliberately leaves the admitted control's source/body untouched.

## Actual SDK2.1.1 compile result

| Local native property | Residual | Eight-warps |
|---|---:|---:|
| Threads / priority grid | 128 /128 | 256 /128 |
| Registers per thread | 242 | 162 |
| Stack B | 0 | 0 |
| Shared B | 45,568 | 45,568 |
| Whole-body static instructions | 2,092 | 1,316 |
| BF16 MMA static sites per warp path | 40 | 20 |
| Nontransposed/transposed SWZL sites | 24 /32 | 12 /24 |
| AIU / CTA-barrier / async-wait sites | 4 /5 /1 | 4 /5 /1 |

Static instruction sites are not dynamic totals: twice as many warps execute
the new body. No performance win follows from the smaller body.

162 is still above128. The preceding device report records131,072 registers
per CU. At256 threads the raw register budgets imply3 CTA slots at162 versus
4 at128, subject to actual allocation/runtime limits. Current grid128 on72CU
has only1.78 CTA/CU on average; the global supply ceiling is14.22 warps/CU
for8 warps per CTA. A fourth register-limited slot does not lift that ceiling.
This is not a claim that registers never matter: larger batch/head grids can
need more slots, and compiler scheduling matters independently of occupancy.
No forced register cap or spill trade was introduced. Check actual ACU
occupancy/resource counters on the user's device.

## Local validation and outstanding device gate

Complete local run:17/17 compiled host tests,82 Python contracts,7 compiler
dialect tests,45 WY and61 residual algebra cases,305 original source controls.
All28 WY images and15 original images are audited;27/27 old native instruction
and operand sequences are identical to the same-SDK parent build. The old
experimental C32 spill remains labeled separately, not attributed here.

The new real-native-trait test checks6,144 producer/consumer values,
163,840 output/K-atom cells across all4 V slices, exact old K order and1,792
publication vectors. Eight host negatives reject omitted upper warps,
duplicate owners, wrong swizzle, omitted reader lane, missing K atom,
reversed K order, missing V slice and stale publisher stride. Six source and
four native negatives reject wrong launch/publication threads, changed math,
duplicate input copies, missing waits/barriers/MMAs and non-paired readers.
Missing backend entry cannot silently fall back to the scalar control.

Evidence: `/workspace/gdn-wy-residual-warps8-evidence-20260924`, with isolated
worktree, plan, configure/verify scripts, complete logs and native dumps.
Local device library SHA256:
`ef418fde4ec4f12d9930ef40f220c13c502c794ee3704ca9956358d442acca4c`.
Local binding SHA256:
`67d975308db18add6ab14f18cae3f1491ea04fa4784b06119f0a694d95f10cb3`.
Neither local compilation nor host proofs certify PPU numeric/performance
results. **Device RAW-BIT / occupancy / latency remain NOT_RUN.**

## Box command

From GDN-QSA-sm80 on `ppu-backend`:

```bash
git pull --ff-only &&
DEVICE=0 JOBS=16 CANDIDATE=residual-warps8 \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

The runner builds with the installed matching SDK, runs the new compiled host
proof and native negatives, retains the original/WY device gates, then tests30
residual cases x8 repeats including tail/GVA/state/output-only. The candidate
must match residual output AND state bytes and the unchanged independent2%
oracle before profiling. Both gate values are admitted, default capture is-1.
Set `GATE=-0.1` only for a separate weak-gate run.

Control residual, candidate warps8 and FLA are captured sequentially with
`/sim/eec/shared/junfu.qx/asight/bin/acu --set full`. Expected state grids:
control128x128, candidate128x256; complete inventories4/4/7 kernels, including
FLA fills. The returned bundle binds source, loaded binaries, fixture,
device and codegen. API_TIMING=NOT_RUN; upload the printed tar path.

Judge complete ACU sums and state time, then traffic/BC/instructions,
eligible/active warps, resources and clocks. Increased warp population without
lower correct full-call latency is not a win. The external goal remains
1.5x FLA; no automatic promotion or speed claim accompanies this handoff.
