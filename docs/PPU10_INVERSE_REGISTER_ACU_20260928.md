# PPU1.0 inverse register delivery: device verdict

Measured source `3227b188fe708eafe2001e30635dc973cbcf38e9`. The decision
rule was registered in [PPU10_INVERSE_REGISTER.md](PPU10_INVERSE_REGISTER.md)
before implementation/capture. Both registered cells are present.

**Correctness passes; no observed performance benefit. Keep the full-chunk
control, leave inverse-register opt-in, and do not change default routing.**
This closes the inverse-retention migration investigation. It does not mean
that all register-delivery algorithms are slower, or that a sub-1% regression
is statistically established by one capture per arm/gate.

## Complete-call ACU results

B1/S2048/Hk16/Hv32/K128/V128/C64, BF16 inputs, FP32 final state, initial=None.
Times are in microseconds, summing **all actual kernels** in the selected
call: four control, four candidate, seven FLA, including both FLA fills.
These are profiled kernel sums, not API-event times or inter-launch gaps.

| Gate | Full-chunk control | Inverse-register | FLA | Candidate latency change |
|---:|---:|---:|---:|---:|
| -1.0 | 169.28529 | 170.20471 | 225.28706 | +0.543% |
| -0.1 | 168.67236 | 168.96294 | 225.51236 | +0.172% |

The retained control is 1.331x/1.337x FLA in these captures. Neither that
comparison nor the suspended 1.5x goal changes the registered A/B decision.

| Gate | Stage | Control | Candidate | Candidate minus control |
|---:|---|---:|---:|---:|
| -1.0 | prefix | 2.36941 | 2.35765 | -0.01176 |
| -1.0 | solve (changed) | 44.59412 | 45.29706 | +0.70294 |
| -1.0 | state | 79.43000 | 78.93412 | -0.49588 |
| -1.0 | output | 42.89176 | 43.61588 | +0.72412 |
| -0.1 | prefix | 2.34118 | 2.36588 | +0.02470 |
| -0.1 | solve (changed) | 44.36412 | 44.45941 | +0.09529 |
| -0.1 | state | 79.40000 | 79.82353 | +0.42353 |
| -0.1 | output | 42.56706 | 42.31412 | -0.25294 |

Unchanged stages contribute +0.21648/+0.19529 us of variation. Their exact
native bodies and executed opcode dictionaries are identical between arms.
Do not attribute this variation to the solve edit. The small solve-time
increases alone do not identify a cycle-by-cycle critical-path penalty.

## The intended change did execute, but it is not a free removal

The uploaded binary executes `gdn_wy_split_solve_register`, not a relabeled
old solve. Each of its 16 native shuffle sites executes 6,144 times. The
98,304 total warp shuffles are exactly 96 per chunk across 1,024 chunks.

| Solve work/resource, same at both gates | Control | Candidate |
|---|---:|---:|
| Scalar FP32 shared stores, warp instructions | 249,856 | 200,704 |
| Scalar FP32 shared loads, warp instructions | 614,400 | 565,248 |
| Indexed warp shuffles | 0 | 98,304 |
| `v.csel.b32` increment | — | +49,152 |
| `s.nop` increment | — | +43,008 |
| Total executed per-opcode instructions | 11,416,576 | 11,449,344 (+0.287%) |
| Total shared bank conflicts | 2,408,448 | 2,211,840 (-8.163%) |
| Shared read transactions | 3,588,096 | 3,489,792 |
| Shared write transactions | 761,856 | 663,552 |
| Registers/thread | 84 | 76 |
| Stack bytes | 0 | 0 |
| Shared allocation bytes | 49,664 | 49,664 |
| Register-limited / shared-limited CTA slots | 12 / 5 | 12 / 5 |

The removed warp-private FP32 round trip is logically 6 MiB stores plus
6 MiB loads for this shape. The table's transaction counts are separate
hardware metrics, not another name for those logical bytes. Global traffic
does not change: solve KVD load/store/staging bytes are respectively
4,456,448 / 8,388,608 / 16,777,216 in both arms.

The PPU C-to-B fragment map needs values from two source slots before a
destination-lane selection. Register retention therefore replaces memory
work with shuffles, selection and scheduling/address changes; it does not
delete those operations at zero cost. All opcode deltas are in the JSON
record. Static native sites fall 1,472 -> 1,468 (including trailing padding),
yet dynamic instructions increase. Per-PC counts cover 1,456/1,452 sites;
the omitted 16 sites in each body are trailing padding, not missing work.

BF16 MMA 81,920, TF32 MMA 98,304, TF32 conversions 524,288, BF16 conversions
131,072, ordered FP32 FMA, AIU/SWZL operations and CTA barrier counts stay
fixed. The three-product TF32 precision algorithm has not been reduced.

Lower registers do not lift this shared-limited capacity. Active warps/CU
are 19.09 -> 18.79 at g=-1 and 19.02 -> 18.93 at g=-0.1; eligible warps/WE
are 0.66 -> 0.65. Memory-dependency per-issue ratios fall 0.63/0.60 -> 0.57,
while compute-dependency rises 0.43 -> 0.47. These are not disjoint wall-time
shares, nor proof that one ratio exactly cancels another. The verified result
is **less shared traffic/BC and fewer registers, without an observed latency
benefit**, not a quantified shuffle-latency attribution.

The opcode sum reconciles with `sass__inst_executed_per_opcode`. Preserve
the distinct `pu__inst_executed.sum` denominator: 11,570,176 -> 11,602,944.

## Admission and provenance

- 36 retained cases x8 plus 256 extent/gate/initial-state cases x8 pass
  RAW-BIT residual comparison and the unchanged independent 2% recurrence
  criterion. The actual case keys were checked, not just the printed count.
  All 64 extents execute the new solve, including tail cases.
- Two cells, 30 kernels; 9 outer manifest entries and 822 entries per inner
  bundle verified. All 347 captured source files per cell match measured SHA.
- Same binary, device, input and preflight/subject receipt in each paired
  cell; FLA source hashes retained and checked. All 30 kernels report 1.700GHz.
- All 43 old native instruction/operand/resource controls match the earlier
  same-box-compiler bundle. The whole uploaded native/resource/linked-symbol
  audit and the inverse-specific native gate pass locally, without execution.
- Eight analysis negative controls reject missing kernels, old solve under
  candidate identity, wrong device, and missing/duplicate cells. The uploaded
  native/source checks also pass their registered mutation tests.

Device: PPU-ZW810, 72 CU, UUID `019ee024-8860-091c-0000-0000007aff6d`.
Driver `2.1.2-r7b50d071022`; actual HGCC compiler `2.2.0-dev`.
`ppu-smi` separately reports SDK `2.0.2-d2568e060312`; do not substitute the
local compile SDK2.1.1 for either box identity. Torch2.9.0/runtime12.9;
FLA0.6.0/Triton3.4.0, native GVA mapping, backend override disabled and the
existing process-local CUDA13 version-parser backport.

Capture ACU: `/sim/eec/shared/junfu.qx/asight/bin/acu`. Local SDK ACU was used
only to import existing reports and export per-PC counters; no device work
or uploaded code was executed during analysis.

Archive SHA256:
`4aaaa64b3d1f3a929f316f2a75d8658b8b744a12343d23edd752e4fd72f5f05c`.
Library SHA256:
`3d30584dcfcccf4943faf78bff1f581161697fe100eccb0cc80d70e3538dae7e`.
Full structured record:
[`inverse_register_acu_20260928.json`](../dev/ppu/results/inverse_register_acu_20260928.json).
Local raw imports/auditor:
`/workspace/actlizeLA-migration-20260928/ppu10-tuning/inverse-register/acu-analysis-20260928`.

Next: close paired-conversion applicability; then PPU geometry selection and
unified public routing. This experiment needs no extra box run to establish
that there is currently no evidence to promote it. Any future speed claim
would require repeated performance admission. No H800 restart.
