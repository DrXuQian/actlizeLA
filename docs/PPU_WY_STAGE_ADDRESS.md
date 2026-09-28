# Prepare/output addressing on the frozen state-both control

Follow-up: [prepare row-factor reuse](PPU_WY_PREPARE_ROWS.md) records the
user-reported result and retains prepare-address with old output as its
incumbent. The table below remains this experiment's original prior evidence.

## Why this experiment

The preceding state ablation has a **user-reported** strong-gate result:

| Full-API role, g=-1.0 | Median us | Observed range us |
|---|---:|---:|
| Old tiled state+output, mask48 | 436.296 | 435.100–438.492 |
| State address only | 401.968 | 400.896–403.968 |
| State row-gates only | 413.482 | 412.388–416.148 |
| State both, mask240 | 378.696 | 377.644–380.476 |
| Original | 424.998 | 416.932–430.620 |
| FLA | 490.140 | 477.336–515.060 |

All three state candidates win against mask48 under the unchanged envelope
criterion; both wins against each single change. This is a pasted excerpt,
not a locally verified complete raw JSON/binary/device identity. The new
weak-gate result is not supplied and is not inferred from the strong case.
Original final state is BF16, WY/FLA state FP32. No automatic routing change.

The [earlier verified mask48 ACU](PPU_WY_STATE_OUTPUT_ACU_20260922.md) measured
prepare/output at 161.331/63.963 us versus matched FLA math at 83.073/43.643 us.
Those stages were untouched by the state ablation. They independently retain
address-work gaps. Do not subtract those profiled durations from the new API
median or assume the difference is host overhead.

## Three new calls, only two new device kernels

| Delivery | Mask | Prepare | State | Output |
|---|---:|---|---|---|
| `tiled-state-output` | 48 | scalar control | old tiled | old tiled |
| `tiled-state-output-both` | 240 | scalar control | address+row gates | old tiled |
| `stage-address-prepare` | 496 | address variant | same state-both | old tiled |
| `stage-address-output` | 752 | scalar control | same state-both | address variant |
| `stage-address-both` | 1008 | address variant | same state-both | address variant |

Prepare reuses the state-proven nonnegative, compile-time 16-byte K/V copy
map. BF16 inverse conversion and K/V conditioning preserve the original
per-thread element order with explicitly nonnegative coordinates. The
conditioning loop stays rolled; it is not a 64-fold instruction expansion.
`prepare_inverse<Address>` is the shared arithmetic authority. The FP32 solve,
TF32 high/high plus two residual terms, exponent arguments, beta multiply
order, BF16 boundaries, W/U arithmetic and scalar W/U publication stay intact.
This is **not** the previously slower tiled-prepare kernel or another store-
vectorization attempt.

Output reuses the same copy map for Q/K/H/V and uses an independently checked
tail-aware 16-byte publisher. Inactive rows form no output pointer and write
nothing. Panel64, 256 threads, reduction order, row gates, union lifetimes,
barriers and output rounding do not change. Both new kernels live in
`gdn_wy_stage_ab_ppu.cu`; no state body is changed.

The unsigned public C ABI selector is unchanged. PrepareAddress bit256 is
valid only with scalar prepare, OutputAddress bit512 only with tiled output.
Conflicting/unconsumed options fail closed. Independent Cartesian census:
4 prepare choices x6 state choices x4 output choices =96 valid masks among
2048 tested values. The old27 and old54 subsets retain their exact semantics.

## Local evidence, not device speed admission

Real PPU SDK2.1.1-a5c56e compilation and native library link pass. All **12
retained native instruction and operand sequences** match the immutable
same-SDK parent library exactly, including the state-both winner.

| Kernel | Registers | Stack B | Static instructions | Static copy sites | BF16 / TF32 MMA sites | Exponent sites |
|---|---:|---:|---:|---:|---:|---:|
| Scalar prepare control | 84 | 0 | 2188 | 2 | 16 / 12 | 9 |
| Address prepare | 86 | 0 | 2431 | 16 | 16 / 12 | 9 |
| Tiled output control | 98 | 0 | 1725 | 4 | 40 / 0 | 18 |
| Address output | 120 | 0 | 1785 | 14 | 40 / 0 | 18 |

Copy sites expand rolled iterations; transferred bytes do not increase.
These are static counts, not executed instruction totals or latency. The
output register increase can reduce residency: do not assume the address
variant wins. Both losing and unresolved results must remain visible.

Local gates: 7/7 CTests; 60 Python contracts (21 WY,24 ACU,8 FLA,7 hgcc);
45 CPU algebra cases plus five negatives. The new gate exhausts 28,672
coordinates, 1,492,480 copy/tail conditions across five strides including
64-bit strides, and 12,288 prepare element iterations. It anchors the map
against both native CuTe layout and an independent hardware cube formula.
Eight negatives cover swizzle, byte/element confusion, omitted vector,
tail write, inverse diagonal, omitted conditioning row, ignored option bit
and missing selector combination. The full census preserves old masks.

Native audit:14 images, zero stack, fixed arithmetic/copy body counts and
cross-TU links. Seventeen negatives include mutations restricted to the new
prepare/output bodies; an unrelated old-control error cannot satisfy those
tests. Python tests exercise actual API mask forwarding, the complete timing
loop, sub-2%-but-not-raw-equal failure, the eight-role denominator and exact
ACU role rebinding even when output fingerprints are equal.

Full unchanged PyTorch binding: **SKIP/environment locally**, because the
private test torch is CPU-only and lacks generated CUDA headers/libraries.
Native hgcc device compile/link is a separate completed check. The box builds
the full binding before device admission. No device execution or speed
result for these new candidates is claimed here.

Plan/logs: `/workspace/gdn-wy-stage-address-evidence-20260922/`.

## One-command box handoff

In `GDN-QSA-sm80`, on the same physical PPU as the control:

```bash
git pull --ff-only origin ppu-backend &&
env -u OUT PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 \
  STAGE_AB=1 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 SAMPLES=16 \
  bash tools/run_ppu_wy_fla_box.sh
```

Old STATE_AB/TILE_AB/DELIVERY_AB inventories remain available and unchanged.
STAGE_AB keeps eight roles: original, scalar WY, old mask48, state-both mask240,
the three new calls and FLA. Both g=-0.1/-1.0 use B1/S2048/Hk16/Hv32/D128,
zero initial state and returned final state. Five warmups,16 balanced samples,
10 full public-API calls/sample, sequential arms and no trimming.

Before timing:16 device cases retain tails/GVA/nonzero state/FP32 gates/
output-only; every candidate must satisfy the independent2% oracle, raw output
**and state** equality to scalar WY, and8 repeated launches. Failures are not
UNRESOLVED timing cells. Paired verdicts remain disjoint observed envelopes;
each new call compares with mask240, not merely old scalar. Combined also
compares with both singles. No promotion from a single strong-gate result.

`[WY verdict] subject=wy` labels the old scalar comparison explicitly; new
candidate verdicts are under `[WY delivery verdict]`. The printed workspace
directory contains `comparison.log`, `comparison.json`, codegen/correctness
logs and source/binary identity. Existing ACU collection accepts the three
new delivery names and binds to the exact compared binary without rebuilding.
