# Paired Vnew layout: measured improvement, not complete BC elimination

At B1/S2048/Hk16/Hv32/K=V128/C64, g=-1, the complete site-ACU kernel sum
falls **194.53824 -> 186.15942 us**, a 4.31% time reduction versus the
same-binary H-layout control. FLA is **224.85000 us**: **1.20784x FLA**,
not the 1.5x goal. All arms measured 1.700 GHz on the same PPU-ZW810.

Keep HV as the next experimental control. This is one capture per arm,
not repeated statistical admission, a weak-gate speed result, or permission
to change default routing. No kernel/default was edited for this analysis.

## Evidence and numerical admission

- Captured source `58b2e29e09013ba2770116a44ab01561a4f15e8f`;
  implementation `fd25bcf59a44c73b26b0b0d4053866d7ccfbbb5e`.
- Archive SHA256:
  `76dfc9d0022b87e0696142b04da250b105eabfa1feb6c4624c725e821e829bf6`.
  628 regular files, 627 manifest hashes and 152 source snapshots verified.
  Tracked source diffs are empty; the two untracked operator-created report
  files are recorded explicitly, not described as an empty worktree.
- Binding SHA256:
  `555513154d4384aa350f33080b7f1ecd58cf52a4dc399b9547816f249a6f9f28`.
  Library SHA256:
  `8dac435d19a7eab24728635f6b20425ee31895b960e1765e50c475e081aecaa7`.
- Same binary, inputs and physical-device UUID across current capture arms:
  `019ee024-8860-091c-0000-0000007aff6d`, 72 CU, 64 MiB L2.
- Device numerics: 30 scalar + 30 H + 30 HV cases, each eight repeats,
  tails/GVA/nonzero state/output-only RAW-BIT gates pass. Independent 2%
  recurrence gate passes for g=-0.1 and -1; **only g=-1 is profiled**.
  The old WY 16-case/32-delivery-admission regression also passes.
- HGGC `2.1.1-a5c56e`, driver `2.1.2-r7b50d071022`, HGGC runtime13.0.
  Capture uses `/sim/eec/shared/junfu.qx/asight/bin/acu`, hash
  `7d750971b45b0bb1f4367df6b8b0f37656d1be89894d40b39ceb2010f865b2b6`.
  FLA0.6.0/Triton3.4.0 source hashes and process-local CUDA13 parser patch
  are bound in the JSON; alternate backend dispatch is disabled.

No uploaded executable or device code was run locally. Reports were reimported
using the compatible local SDK parser, retaining the actual site capture identity.

## Full-call timing

All numbers below are ACU kernel durations in microseconds, not API latency.

| Stage | H control | HV candidate | FLA |
|---|---:|---:|---:|
| Prefix/cumsum | 2.47588 | 2.39471 | 2.75765 |
| BF16 fill | — | — | 4.50941 |
| Solve | 52.26824 | 52.32059 | 44.81294 |
| W/U | in residual state | in residual state | 36.52706 |
| FP32 fill | — | — | 2.25941 |
| State | 95.72765 | 88.40706 | 90.44471 |
| Output | 44.06647 | 43.03706 | 43.53882 |
| **Complete call** | **194.53824 (4 kernels)** | **186.15942 (4)** | **224.85000 (7)** |

State saves 7.32059 us (-7.65%); output saves 1.02941 us (-2.34%).
Unchanged prefix/solve contribute another -0.02882 us of observed variation,
not credited to Vnew. Residual state incorporates the W/U algebra; the FLA
work-grouping counterpart is W/U+state = 126.97177 us, not state alone.

The 1.5x target is 149.90000 us: **36.25942 us (19.48%) remains**.

## Mechanism: the registered publication prediction materialized

The [paired implementation](PPU_GDN_RESIDUAL_WARPS8_HV_LAYOUT.md) changes only
private Vnew publication and its output reader. Input V, public output,
public FP32 state, H layout, math order and rounding boundaries are unchanged.

| Dynamic work | State H -> HV | Output H -> HV |
|---|---:|---:|
| BF16 MMA | 655360 -> same | 524288 -> same |
| BF16 conversion | 1311744 -> same | 393216 -> same |
| Normal matrix loads | 917504 -> same | 655360 -> 786432 |
| Transpose matrix loads | 262144 -> same | 131072 -> 0 |
| AIU instructions | 16384 -> same | 10240 -> same |
| Shared-load instructions | 1708032 -> same | 983040 -> same |
| Shared-store instructions | 1343488 -> same | 395264 -> same |
| Native per-PC instruction sum | 15658112 -> 15350912 | 11257856 -> 11044864 |

Output replaces its remaining 131072 transpose loads one-for-one. A square
64x64 Vnew tile stays one native AIU cube; unlike the prior H change, there
is no additional bulk-copy instruction. State removes 164864 wait instructions;
output adds 12288 waits but removes 163840 vector-to-scalar moves. Neither
wait instruction count alone nor fewer registers predicts the latency result.

Every one of the 15 opcode sums closes against ACU's built-in opcode total;
omitting an executed PC fails. Both changed native bodies and MMA/matrix-load
operands match their report PC streams (excluding 16 alignment NOPs each).
Twelve source/binding and six native negative controls remain red. The
separate `pu__inst_executed.sum` metric has a different counting scope; it is
retained in JSON and is not silently substituted for the opcode denominator.

| Traffic | State H -> HV | Output H -> HV |
|---|---:|---:|
| KVD -> TSM | 112 -> 112 MiB | 80 -> 80 MiB |
| Ordinary global-load KVD bytes | 17 -> 17 MiB | 0.25 -> 0.25 MiB |
| Global-store KVD transactions | **66 -> 50 MiB** | 16 -> 16 MiB |
| L1/KVD -> L2 stores | **66 -> 50 MiB** | 16 -> 16 MiB |
| L2 -> LLC stores | 50 -> 50 MiB | 16 -> 16 MiB |

State logical stores remain H32 MiB + Vnew16 MiB + final FP32 H2 MiB =50 MiB.
The former Vnew row was 64B at256B pitch; private [V,T] writes contiguous
128B rows. The production publication-map 128B footprint model predicts
H32 + Vnew32 + final2 -> H32 + Vnew16 + final2 = **66 -> 50 MiB**, matching
both measured inner-hierarchy store counters. This is a matching footprint
inference, not a per-PC transaction measurement or a universal hardware
service-granularity assertion. The raw transaction unit remains64B.

State DRAM reads fall50,735,360 ->33,957,760B, but L2 -> L1 input reads
remain127,935,488B and LLC demand-read bytes differ only2,432B. Do not claim
16 MiB of removed compulsory input: cache/write-allocation/replay effects
are not isolated. DRAM writes actually rise8,468,736 ->10,160,640B.

## BC and occupancy: important counterexamples

| Counter | State H -> HV | Output H -> HV |
|---|---:|---:|
| Read BC | **2621440 -> 3014656 (+15%)** | **524288 -> 0** |
| Write BC | 1949696 -> 1425408 (-26.89%) | 1179648 -> same |
| Total BC | 4571136 -> 4440064 (-2.87%) | 1703936 -> 1179648 (-30.77%) |
| Registers/thread; stack | 122 -> 122; 0 | 78 -> 70; 0 |
| Shared bytes | 45568 -> same | 49408 -> same |
| Register-limited CTA slots | 4 -> 4 | 6 -> 7 |
| Shared-limited CTA slots | 5 -> 5 | 5 -> 5 |
| Achieved active warps/CU | 14.18 -> 14.16 | 36.82 -> 36.74 |
| Eligible warps/WE | 0.38 -> 0.40 | 0.80 -> 0.83 |

Output's counted read BC is zero, but this is **not a conflict-free whole
format**: state reads have more conflicts, although its transpose/scalar
load counts are unchanged. Vnew's shared publication addresses changed.
The earlier opcode-count-only empirical BC fit is insufficient for this
new layout; native service phases and the vector publisher still matter.
Do not label all remaining state BC as K-transpose or all gain as BC removal.

State is still near its grid supply ceiling128*8/72=14.22 warps/CU.
Output remains shared-limited to five CTAs despite another register slot.
This gain is not an occupancy increase.

Per-issue state ratios: memory1.51->1.48, TSM1.18->1.21,
compute0.62->0.55, TFU-WAR0.07->0.04, sync1.10->0.95.
Output: memory1.82->1.81, TSM1.33->1.28, compute0.74->0.80,
TFU-WAR0.32->0.34, sync3.05->3.03. These are not wall-time shares.
Lower time can coexist with increased TSM or compute waiting ratios.

## Next bounded experiment, not another global layout rewrite

Use HV as the same-shape experimental control; keep existing fallbacks and
default routing unchanged. Output read-BC elimination has already happened;
another experiment aimed only at that counter has no remaining target here.

First investigate **state input-ready/metadata overlap**, preserving the
current K/H/V layouts, grid, useful traffic, math and rounding. Candidate PC
0x5480def8 is the first H-publication vector load after INPUTS_READY and has
4116 of4587 state sync samples. Before that region, the32-bit gate load at
0x5480dcb8 and16-bit beta load at0x5480dcf8 have `vldcnt` waits with471/412
memory samples. This identifies an arrival/metadata-latency hypothesis,
not proof that the gate loads caused all barrier waiting or that the samples
represent microseconds. Any prefetch must be checked in the actual loop CFG,
retain completion and old-reader retirement, and account for registers.

K's two-orientation delivery remains a separate possible optimization; no
unproved no-trans replacement, duplicate K copy or added shared footprint
is folded into the metadata experiment. The52.32us solve also remains a
substantial target, but its triple-TF32 arithmetic is not a layout-only knob.
No promise that one of these changes alone reaches1.5x is made.

## Replay and machine-readable evidence

All raw data, verified extraction, host parsers and PC exports are retained
at `/workspace/gdn-warps8-hvlayout-acu-analysis-20260925`.
Summary: [residual_warps8_hvlayout_acu_20260925.json](../dev/ppu/results/residual_warps8_hvlayout_acu_20260925.json).

```bash
python /workspace/gdn-warps8-hvlayout-acu-analysis-20260925/analyze.py --native
python /workspace/gdn-warps8-hvlayout-acu-analysis-20260925/summarize.py
python dev/ppu/check_residual_warps8_hvlayout.py --self-test \
  --isa /workspace/gdn-warps8-hvlayout-acu-analysis-20260925/input/acu/isa.txt
```
