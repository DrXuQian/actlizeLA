# Matched state geometry, registered before box measurement

Parent: `8777067bb4785bd36de1f4991bd0029513895818`. No device results yet.
Purpose: finish the shape-dependent, architecture-independent part of the
SM90 migration. The old 1.5x FLA target is not an admission requirement.

## What is held fixed

BF16 Q/K/V/beta, BF16 or FP32 log gate, K=V128, C64; FP32 recurrent state;
identical ordered MMA, scalar BF16 rounding, paired H/V private layout,
gate-coefficient cache and static solve. The output kernel is shared unchanged.
All old kernels and default/public routing are retained. This is not a scan
over the older V16/4-warp implementation: that would also change layouts.

Only the state ownership geometry varies:

| Arm | V columns / warps | State shared bytes | Local SDK2.1.1 full / tail registers | Stack |
|---|---|---:|---:|---:|
| retained full-chunk control |32 / 8|46,080|120 /124|0|
| geometry-v32-w8, calibration |32 / 8|46,080|120 /124|0|
| geometry-v32-w4 |32 / 4|46,080|162 /168|0|
| geometry-v16-w4 |16 / 4|35,840|96 /102|0|

Calibration is native instruction-and-operand identical locally, including
tail after normalizing only the function ordinal in BB labels. Keep it in
the box capture: the box compiler can differ. V16/8 is excluded: C64xV16 has
only four complete native output warp fragments. Eight active warps need
a different algorithm (partial/idle warps or K reduction).

V32/4 keeps CTA count and shared capacity but gives each warp twice as many
outputs. It reuses B operand fragments across more row accumulators, with
more registers per thread and fewer runnable warps per CTA. V16/4 doubles
state CTAs but also duplicates K/inverse input traffic; it is not free
occupancy. Registers alone, or bank conflicts alone, do not select a winner.

## Fixed workload and evidence

`dev/ppu/geometry_shapes.json` is the one executable shape inventory:
B/S/Hk/Hv = 1/2048/16/32, 1/2048/8/16, 2/2048/16/32,
1/2048/16/64, 1/2049/16/32, 1/2111/16/32. Both g=-1 and -.1;
no initial state, final state required:12 cells. Every cell captures the
same-binary control, all three candidates and FLA once:60 subject calls.
Each subject is one public call under external ACU, with independent
preflight and the same inputs. No API-event timing selects a winner.

Local L040 proves every lane/warp/slice in3 geometries x64 extents x2 planes
x2 groups:9,437,184 values written,31,457,280 state operand values read,
37,748,736 output operand values read. Real native C/B traits and actlize's
ld.swzl simulation are the oracle.38 negative cases catch wrong ownership,
stale pitch, reader corruption, wrong publication and missing coverage.
Device gate: retained36 cases plus256 two-chunk extent/gate/initial cases,
each candidate8 RAW-BIT repetitions versus residual scalar and unchanged
independent2% oracle. GVA expansion and output-only are also checked.
Every performance shape is separately admitted before profiling.

## Decision registered now

Use the sum of **all** kernels for that call (including fills), and also
report state-only time. Verify exact shape/arm/symbol and numerator/denominator
before interpreting. Register/shared/warp occupancy limits, achieved active
warps, traffic, BC and stalls explain results; none alone is a speed verdict.

One capture per arm/cell is descriptive only. Positive or negative outcomes
both remain in the report. The calibration arm checks timing/codegen noise.
No default winner is admitted from this initial sweep. Finalist selection
requires repeated same-cell paired captures, disjoint observed envelopes;
overlap is UNRESOLVED. Never claim an unmeasured shape is proven. Changes in
untouched stages and clock/device/compiler changes must be reported separately.

## Run and upload

```bash
git fetch origin
git switch agent/ppu10-geometry-20260928
git pull --ff-only
DEVICE=0 JOBS=16 bash tools/run_ppu10_geometry_box.sh
```

Use your current SDK via `PPU_SDK=/path/to/PPU_SDK` if it is not installed at
`/usr/local/PPU_SDK`. ACU defaults to `/sim/eec/shared/junfu.qx/asight/bin/acu`.
Fresh output directories are created below `/workspace`; the caller's shell
is not exited. Builds and kernels run sequentially, not concurrently across
arms. Upload only the printed `geometry.tar.gz`. It contains all12 captures,
source/binary identity, native resource audit and numerical receipts.
Failure preserves evidence and prevents performance/default admission.

## Local verification (not device admission)

SDK2.1.1 real PPU build and host link completed:52 WY device images;
all46 retained native instruction/operand sequences and resource records
identical to parent. All six new full/generic bodies use native AIU.swzl
+ ld.swzl, direct fragments and vector stores, with zero stack spill.
L040 and31/31 CTest pass;285 CPU tests pass. Four NVIDIA device-only test
modules are separately SKIP, not counted as CPU passes. PPU execution and
speed admission remain NOT_RUN. Source/native/linked negative controls
reject changed rounding, wrong owner/grid, full-for-tail dispatch, missing
MMA and wrong C ABI targets. Five-arm collector and archive negatives reject
missing coverage, changed input shape/output, stale library and mixed device.
