# Eight-warps: paired residual/scaledV B layout

Device follow-up: [2026-09-24 ACU verdict](PPU_GDN_RESIDUAL_WARPS8_B_LAYOUT_ACU_20260924.md)
passes numerics and lowers BC26.3%, but full-call205.359->205.854us shows
no speed gain. Keep the eight-warp control; routing unchanged. The following
records the original local handoff, before that measurement.

Parent a2c8617. This is a layout-only branch of the measured eight-warp
residual implementation, not a new arithmetic algorithm or default route.
Priority: B1/S2048/Hk16/Hv32/K=V128/C64, g=-1.0 profile; both -0.1/-1.0
numeric admission. Device numerics, BC and performance are **NOT_RUN** locally.

## Exact change and unchanged work

Only R and scaledV, which have no global-publication consumer, are written
directly in native B orientation and read by nontransposed `ld.swzl`.
No transpose kernel, software shuffle, extra copy or barrier is added.
H snapshots, K, P inverse, Vnew, their global layouts, all four AIU input
sites, 256 threads, V32/grid128, chunk/K order and BF16 boundaries stay fixed.
In particular scaledV remains `BF16(x * row_decay)`, NOT scaling `BF16(x)`.
H has another vector publisher and K has consumers in both orientations;
this change deliberately does not alter them.

The physical microcube layout and native reader are inherited from the
four-warp B-layout branch, with one shared source of truth. The producer
base is newly derived from eight-warp ownership:

```text
microcube = cube(value_row(warp,0), column(warp)) = warp
base      = microcube*256 + (lane%4)*16 + lane/4
```

The old four-warp producer formula is not reused: it changes upper-warp
destinations and goes out of bounds. The actual C-layout writers and actual
actlize native-reader simulator are exhaustively checked against independent
B-fragment coordinate tags. No empirical BC coefficient is a host test gate.

## Local SDK2.1.1-a5c56e / PPU0010 result

| Native property | Eight-warp control | Eight-warp B layout |
|---|---:|---:|
| Threads / priority grid | 256 /128 | 256 /128 |
| Registers/thread | 162 | 124 |
| Stack bytes | 0 | 0 |
| Shared bytes | 45,568 | 45,568 |
| Whole-body static sites | 1,316 | 1,313 |
| Recurrence backedge bodies, static sites | 523 /537 | 516 /530 |
| Normal / transposed matrix-load sites | 12 /24 | 20 /16 |
| BF16 MMA / AIU / CTA barrier / commit-wait sites | 20 /4 /5 /1 | 20 /4 /5 /1 |

The compiler naturally allocated124 registers; no register cap was applied.
This is a resource observation, not a proof that removing transpose alone
caused all38 registers to disappear. Current grid128/72CU still supplies only
1.78 CTAs/CU on average, so a fourth register-limited slot need not increase
achieved occupancy. Inspect the user's actual compiler/resources/counters.

Static/source inventory predicts exactly262,144 transposed reads replaced
by normal reads, out of786,432 in the control. Total matrix-load count stays
1,179,648 and global-to-shared payload stays112MiB for the priority shape.
These are workload budgets, not new device measurements. Counted TSM bytes,
read/write BC and elapsed time must be measured. The earlier four-warp
B-layout lowered BC without improving latency; this combination has no
presumed performance admission.

## Proofs and fail-closed controls

`l027_wy_residual_warps8_blayout` covers both planes, all four V slices,
all64 valid-row counts, all8 warps and all32 lanes. It checks1,048,576 writer
values and4,194,304 reader values; writers are unique and tails are zero.
Eight planted errors must fail: old four-warp base, old Value placement,
wrong swizzle bit, wrong consumer cube, missing lane, missing warp, missing
K tile and missing plane. Coverage denominators are fixed independently.

Six source negatives reject stale writer/reader, altered rounding, old
publication thread count, missing retirement and wrong owner mapping.
Four native negatives reject wrong transpose, removed MMA, wait or barrier.
After reversing only the permitted layout edits, the candidate source is
equal to the eight-warp control. Native math, transfer/publication and
synchronization inventories remain equal, including recurring backedges.
All28 previous device bodies have identical native instruction/operand
sequences in the same SDK build. All29 WY images are audited and linked.

Full local run:18/18 CTest,85 Python host contracts,7 compiler-dialect tests,
45 WY +61 residual algebra cases,305 original source controls; original15
images also audited. The historical experimental C32 stack is separately
labeled and not part of this candidate. Local device execution is NOT_RUN.
Box must still pass30 fixtures x8 for both eight-warp branches, independent
2% recurrence numerics, GVA expansion, nonzero state, tails and output-only.

The timed/profiled control is now explicitly registered per candidate in
`RESIDUAL_CONTROLS`. For this branch it is `residual-warps8`, NOT four-warp
`residual`. Scalar residual remains the independent RAW-BIT numeric anchor.
Wrong control geometry, wrong output, missing selected backend and missing
ACU are tested failures. Both controls are admitted in the same binary and
bundle; no silently substituted path may produce an apparent speedup.

Evidence: `/workspace/gdn-wy-warps8-blayout-evidence-20260924` contains the
pre-edit plan, configure/verify scripts, full logs and native dumps.
Initial local library SHA256:
`c3339f04cafb5d4abf3a9452f289939a32da41046f2d45e5de293507e01085b2`.
Binding SHA256:
`798332ce02308741348e36a34258bcb6454b3bf7cb270d06773c2078ef046e29`.
These compile artifacts are not presented as device-tested binaries.

## Box command and fixed decision

In GDN-QSA-sm80, branch `ppu-backend`:

```bash
git pull --ff-only &&
DEVICE=0 JOBS=16 CANDIDATE=residual-warps8-blayout \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Use the already working SDK, or set `PPU_SDK=/path/to/PPU_SDK`. The runner
creates a fresh directory under `/workspace`, compiles/links and runs local
layout/native gates, then device numerics before sequential captures:
eight-warp control, eight-warp B-layout, FLA. Both state grids are128x256;
complete-call inventories are4/4/7 kernels including FLA helpers. It uses
`/sim/eec/shared/junfu.qx/asight/bin/acu --set full`, no API performance timing.
Upload the printed tar path. `GATE=-0.1` selects a separate weak-gate capture;
never overlap timed or profiled jobs on the same device.

Judge full-call ACU sums first, then state time. Report unchanged-stage
variation separately, followed by BC/request/traffic counts, register/spill,
active/eligible warps and clocks. Lower BC/registers without lower correct
complete-call latency is not a win. A single near-parity capture is not a
stable speedup. The1.5x FLA target remains a target, and default routing
stays unchanged regardless of this compile result.
