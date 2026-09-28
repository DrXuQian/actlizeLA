# V32 eight-warps: more useful issue supply without duplicating global staging

## Verdict and scope

Keep `residual-warps8` as the next experimental control. This capture shows a
real reduction in the changed state stage and in the complete kernel sum.
It is not a default-routing promotion or a claim of winning other shapes.
The target of 1.5x FLA is **not met**.

Source `dfe76c3d745586bd008ddfca52edcd2f0500f8b0` (implementation `1750cf1`),
B1/S2048/Hk16/Hv32/K=V128/C64, g=-1.0, zero initial state, final state enabled.
PPU-ZW810, 72 CU. All 15 profiled kernels report 1.700 GHz. These are **ACU
kernel durations**, not API event spans. There is one full capture per arm;
do not label these medians or a repeated performance-confidence interval.

| Stage, microseconds | Residual, 4 warps | V32, 8 warps | FLA |
|---|---:|---:|---:|
| Prefix | 2.40765 | 2.41941 | 2.81765 |
| KKT/inverse | 52.38529 | 52.09882 | 44.14882 |
| W/U + state (combined algebraically in ours) | 127.74471 | 104.81000 | 36.49529 + 91.45176 |
| Output | 46.10294 | 47.67412 | 45.45765 |
| Required separate fills | none | none | 4.32353 + 2.27471 |
| **Complete kernel sum** | **228.64059** | **207.00235** | **226.96941** |

State falls 22.93471 us, or 17.95% (1.219x). The full sum falls 9.46%
(1.105x versus control), and is **1.096x FLA**. The three unchanged stages
contribute +1.29647 us of variation, chiefly output +1.57118 us. Their exact
symbols and native opcode counts are unchanged in the same DSO. Do not
attribute that variation to the changed state implementation.

Our state performs the work corresponding to **FLA W/U + H**, not just H.
Calling 104.81 > 91.45 us a remaining state regression omits FLA's 36.50 us
W/U preparation. Ours still has different BF16 association from FLA; the
independent 2% numerical contract, not cross-library RAW-BIT, admits that.
Eight-warps versus the residual control does retain RAW-BIT equality.

At this capture's FLA time, 1.5x means 151.31294 us. Another 55.68941 us
(26.90% of the current sum) must disappear. With all other stages fixed,
state would need 49.12059 us. That is budget arithmetic, not a prediction;
minor counter improvements must not be presented as closing this target.

## The register concern: 162 is real, but 128 is not this grid's threshold

| State resource / issue metric | Control | Eight-warps |
|---|---:|---:|
| Grid / CTA threads | 128 / 128 | 128 / 256 |
| Registers per thread | 242 | 162 |
| Stack bytes per thread | 0 | 0 |
| Shared bytes per CTA | 45,568 | 45,568 |
| Block limit, registers / shared | 4 / 5 | 3 / 5 |
| Theoretical active warps/CU | 16 | 24 |
| Achieved active warps/CU | 7.07 | 14.12 |
| Achieved occupancy | 11.05% | 22.06% |
| Eligible warps / WE active cycle | 0.25 | 0.36 |

There are only 128 CTAs over 72 CUs. The eight-warp global supply ceiling
is 128*8/72 = 14.222 warps/CU, and 14.12 is already close. At the recorded
131,072 registers/CU, lowering 162 to 128 might change register capacity
from three to four CTAs, but it cannot create more CTAs in this grid.
No register cap, spill trade or hidden reduction in useful MMA was used.
Lower register demand can still help code scheduling or larger grids;
this result only rejects treating <=128 as a prerequisite for this experiment.

The source change also shortens each warp's fragment work and changes codegen.
Thus the speedup is not a measurement of occupancy alone. Higher eligible
warp supply, unchanged useful work and shorter time are consistent with
improved latency hiding; they do not assign every saved cycle to that cause.

## Two traffic directions, and the explicit cost that remains

| State measurement | Control | Eight-warps | Change |
|---|---:|---:|---:|
| KVD -> TSM transaction bytes | 112 MiB | 112 MiB | unchanged |
| Ordinary KVD global-load bytes | 17 MiB | 17 MiB | unchanged |
| KVD global-store bytes | 98 MiB | 98 MiB | unchanged |
| AIU load instructions | 16,384 | 16,384 | unchanged |
| Shared matrix-load instructions | 917,504 | 1,179,648 | +28.57% |
| All shared-load instructions | 1,429,504 | 1,708,032 | +19.48% |
| Shared-store instructions | 1,343,488 | 1,343,488 | unchanged |
| Read bank-conflict counter | 2,883,584 | 3,932,160 | +36.36% |
| Write bank-conflict counter | 4,046,848 | 4,046,848 | unchanged |
| Total BC | 6,930,432 | 7,979,008 | +15.13% |
| BC / useful BF16 MMA | 10.575 | 12.175 | +15.13% |
| Read BC / all shared-load instructions | 2.01719 | 2.30216 | +14.13% |
| Useful BF16 MMA | 655,360 | 655,360 | unchanged |

The promised non-duplication applies to **global-to-shared staging**, unlike
the earlier V16 experiment's measured 112 -> 224 MiB. It does not apply to
shared-to-register operand reads. Non-transposed matrix loads remain 393,216;
transposed loads rise 524,288 -> 786,432, exactly closing the preregistered
224 -> 288 loads per CTA/chunk budget. Extra scalar shared loads account for
the other 16,384 instructions. More warps split the H/residual/scaled-V
register reuse. Global DRAM reads remain nearly equal:
84,298,752 vs 84,291,968 B. These hierarchy metrics are not all unique payload.

BC elimination did **not** happen. This is a counterexample to rejecting a
mapping just because aggregate BC rises: here the full correct call improves
despite that cost. Conversely, this does not establish that remaining BC is
free or that increasing warps again would improve performance.

## Dynamic instructions and stalls: count the right denominator

| Native per-PC opcode count | Control | Eight-warps |
|---|---:|---:|
| Complete state opcode sum | 14,680,704 | 16,107,648 |
| `s.wait` | 1,537,536 | 2,078,720 |
| `v.mov.v2s` | 674,816 | 1,153,024 |
| `s.blksyn.defer` | 66,048 | 132,096 |
| `v.reg.dchk` | 294,912 | 139,264 |
| `v.cnvt.bf16.f32.rtte` | 1,310,720 | 1,311,744 |

Total native instructions rise 9.72% while state time falls 17.95%. The
distinct `pu__inst_executed.sum` metric is 14,746,240 -> 16,664,704 (+13.01%);
do not mix these counter scopes. Static body size 2,092 -> 1,316 sites is not
the dynamic count: eight instead of four warps execute the body.

The additional 1,024 BF16 conversions come from one setup PC executing once
per launched warp: `v.cnvt.bf16.f32.rtte vreg100, sreg19`, whose source was
set to zero. The recurring conversion work remains 1,310,720. This is not a
new per-output precision boundary. The five static CTA barrier sites also
remain unchanged; twice as many warp participants explain the doubled barrier
instruction count, not twice as many source synchronization phases.

Memory-dependency per-issue ratio increases 1.02 -> 1.72, TSM dependency
0.74 -> 1.38, sync 0.52 -> 1.10, instruction fetch 0.20 -> 0.24. Therefore
"it is faster because each warp stalls less" is unsupported. More runnable
work can coexist with more aggregate/per-issue waiting. These ratios are
not disjoint wall-time percentages and must not be summed into a time budget.

## Next decision, not another blind register cap

Use this exact eight-warp branch as the new experimental control, retain the
old residual and all existing numerical gates, and leave default routing
unchanged. The next bounded layout experiment can combine the already-proved
B-oriented intermediate layout with **this** ownership geometry, after
re-proving producer/reader maps. The former four-warp B-layout result was
negative; it is not an admission of the combination. Its falsifiable target
is lower correct full-call time versus 207.00 us in a contemporaneous capture,
not merely fewer conflicts or lower registers. Stop this direction if only
the counters improve again.

The remaining 1.5x gap also needs stage budgets, not only state tuning:
inverse is 52.10 us versus FLA 44.15 us, output 47.67 versus 45.46 us. Solve
still executes 98,304 TF32 MMAs versus FLA 32,768, with equal 81,920 BF16 KKT
MMAs. That is an existing precision-algorithm difference; simplifying it
would require its own numerical experiment, not a layout-only permission.
No kernel, math, selector or runner changes were made during this analysis.

## Integrity, correctness and replay

Archive SHA256
`b86b02f9ec67ffcc666f42945ff7e298a074f0b0651c7c588b93ed7b731cf612`:
3,703,105 B; 611 files, 610 checksums, 135 captured source files match the SHA.
The first truncated upload was retained separately and never used for a
performance verdict. Tracked source/diffs are clean; loose operator-created
`acu.tar.gz` and two reports are explicitly recorded, not source modifications.

Device UUID `019ee024-8860-091c-0000-0000007aff6d`, driver
`2.1.2-r7b50d071022`, hgcc `2.1.1-a5c56e`.
Device DSO SHA256
`e017ebd988523df4e7debc0dd8d916c070d8da9fb38ced47a83312e8c817bf97`;
binding SHA256
`5abcd0302f3d72adeed1c96a0e0fd80095b71a84990f71b58e5ca129f1d74b59`.
All paired receipts bind the same fixture, physical device and loaded binary.
Before/after device snapshots say no processes; this is not a continuous
trace excluding all possible external interference.

Thirty residual + 30 eight-warp device cases x8 repeats pass, including
tail/GVA/initial-state/output-only and output AND state byte equality. The
old 16 WY cases and 32 delivery admissions pass. Both g=-0.1/-1.0 have
independent 2% numerical admission; **only -1.0 is profiled here**.
FLA 0.6.0 / Triton 3.4.0 source hashes and process-local CUDA13 version-parser
backport are retained in the machine-readable result; no algorithm patch.

Capture tool: `/sim/eec/shared/junfu.qx/asight/bin/acu`, banner
`v2.0.0_20251231-4f7cd70`, data version 12006. Local SDK report import is
host-only, not a device rerun and not a claim that capture used the SDK tool.
All 15 per-PC opcode sums close; omitted-PC negative controls fail. The exact
uploaded native bodies also pass the trusted source/native inventory audit
and its six source/four native negative controls. Ambiguous duplicate device
attribute fields in the text exporter are retained and not used.

Evidence/replay: `/workspace/gdn-residual-warps8-acu-analysis-20260924`
(`capture.tar.gz`, `analyze.py`, `reimport.sh`, `analysis-verified.log`,
`summarize.py`, `summary.json`, `benchmark.csv`, native audit and per-PC exports).
No uploaded source, binary or shell command was executed locally.
Condensed result:
[residual_warps8_acu_20260924.json](../dev/ppu/results/residual_warps8_acu_20260924.json).
