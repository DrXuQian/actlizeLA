# H800: FlashQLA / FlashInfer full GDN forward kernel time

## Result and target

**FlashInfer with CP disabled is the fastest measured implementation on this
workload. Our current SM90 CUDA kernel is about 3.19x slower; the PPU-CUTLASS
CUDA source-check build is about 2.99x slower. Beating the previously measured
cuLA C++ entry is not sufficient.**

User target: strongest admitted SM90 forward against the best comparable
implementations. Feed portable algorithm/dataflow improvements back into
SM80/PPU1.0 afterward; do not force Hopper instruction-level compatibility.
This report establishes the target, **not achievement of that target**, and
does not establish a universal winner across all shapes.

No production kernel, default dispatch, tolerance or hardware settings changed.
All measurements below are physical NVIDIA H800 PCIe, **not native PPU1.7**.

## Same-input results

B1/T2048/Hqk16/Hv32/K128/V128, BF16 Q/K/V, scalar natural-log gate
g=-0.1/-1.0, identical BF16-quantized beta, zero initial state, FP32 final
state, scale=1/sqrt(128), no internal Q/K normalization. Independent recurrent
CPU reference and unchanged **<2% max-error/max-reference** on both O and state.

Metric: median **sum of all GPU kernel durations per complete forward** from
nsys, us. Twelve calls/role/gate, interleaved with unchanged incumbent binaries.
Brackets are observed ranges. These are not API or CUDA-event timings.

| Implementation / selected path | g=-0.1 | g=-1.0 | Kernels/call |
|---|---:|---:|---:|
| FlashInfer **CP off** | **112.014 [109.758,113.022]** | **111.279 [109.631,113.119]** | 1 |
| FlashInfer default auto CP | 130.430 [129.022,131.806] | 129.374 [128.991,132.924] | 4 |
| FlashInfer auto CP + GPU log-gate/beta adapter | 139.439 [138.046,141.757] | 138.766 [136.863,140.254] | 7 |
| FlashQLA default auto CP, forward-only | 166.252 [163.228,167.517] | 163.598 [160.543,164.990] | 17 |
| FlashQLA CP off | 175.853 [174.843,177.628] | 174.750 [173.694,176.479] | 3 |
| Ours CUDA, FlashInfer measurement window | 357.548 [356.347,359.675] | 355.020 [354.268,357.180] | 1 |
| Ours PPU-fork **CUDA source-check**, same window | 335.020 [331.612,338.747] | 332.172 [329.820,335.356] | 1 |

All listed competing-path ranges are disjoint within each gate. FlashInfer
CP-off is 1.484x/1.470x faster than FlashQLA auto by descriptive median ratios;
their dependency environments differ, as explained below. FlashInfer's own
auto-CP heuristic loses to CP-off here by about16.4%/16.3% in kernel work time.
FlashQLA auto-CP saves about5.5%/6.4% versus its CP-off path.

Our contemporaneous controls in the FlashQLA environment were CUDA
357.897/354.043us and PPU-fork332.697/330.460us. Cross-environment incumbent
drift is <0.71%, far smaller than the cross-library gap. Environments were
measured sequentially, not misrepresented as one interleaved process.

Historical context only: the earlier original cuLA **C++** entry measured
447.249/447.376us in [the preceding report](SM90_CULA_NSYS_20260926.md).
It was not remeasured here; that result is not a claim about cuLA's fastest
possible DSL/backend/configuration.

## What actually ran

FlashInfer CP-off is `FullyFusedDeltaRuleSm90`: grid32, block512,128 reported
registers/thread,185728B dynamic shared. Both our kernels also run grid32,
block512,128 registers,167936B dynamic shared. **Grid underfill alone therefore
cannot explain our roughly3x gap.** Resource counts alone do not establish
equal active occupancy, spills, instruction count, or pipeline efficiency.

FlashInfer auto CP uses four kernels. Per-component medians (these need not
sum exactly to the median of the per-call total):

| Stage | g=-0.1 us | g=-1.0 us | Grid / threads |
|---|---:|---:|---|
| T precompute | 16.495 | 16.432 | 1024 /128 |
| MN precompute | 50.975 | 50.144 | 64 /384 |
| State fixup | 12.064 | 11.888 | 64 /256 |
| CP prefill | 50.671 | 50.991 | 64 /512 |

The CP prefill by itself is much faster than111-112us; the extra precompute
and fixup work makes the **complete** CP path slower. No stage was omitted.

FlashQLA auto CP splits the sequence at0,512,1024,1536,2048. Its important
weak-gate components are prefix3.728us, KKT solve23.407us, warmup selection
2.272us, prepare-state26.112us, correct-initial-state8.416us and fused
forward86.590us. All auxiliary PyTorch metadata kernels are also counted,
bringing the actual denominator to17 kernels/forward.

FlashQLA CP-off contains prefix3.744us + KKT solve23.951us + fused
forward148.157us. **CP-off does not mean no extra parallelism:** upstream
chooses `block_DV=32`, creating128 CTAs by splitting V. Auto CP instead uses
four subsequences and `block_DV=128`, also128 CTAs. Its inverse precompute
is a separate1024-CTA kernel. Our current head-only launch has32 CTAs.

Source confirmations:

- [FlashQLA pinned Hopper forward / V-split selection](https://github.com/QwenLM/FlashQLA/blob/a97c9783bbcc42fa8fbfe895dfc674131e376b5c/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py).
- [FlashQLA pinned CP policy and correction](https://github.com/QwenLM/FlashQLA/blob/a97c9783bbcc42fa8fbfe895dfc674131e376b5c/flash_qla/ops/gated_delta_rule/chunk/cp_context.py).
- [FlashInfer pinned public dispatch](https://github.com/flashinfer-ai/flashinfer/blob/5d9f8c8d97fa53e22952ce8672f475d235f07478/flashinfer/gdn_prefill.py),
  [SM90 fully fused implementation](https://github.com/flashinfer-ai/flashinfer/blob/5d9f8c8d97fa53e22952ce8672f475d235f07478/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py).

This is not evidence that a DSL is inherently faster than C++/CUTLASS, or that
one particular barrier/precision choice causes the gap. Algorithm, layout,
register lifetime, pipeline and compiler differ. Both our current inverse and
FlashInfer's non-CP inverse use FP16 storage; do not invent a TF32-vs-FP16
explanation for that comparison. Instruction/cycle attribution remains open.

## Accuracy and ABI adapters

| Path | Weak O/state max-relative error | Strong O/state max-relative error |
|---|---:|---:|
| Ours, both dependencies | 0.4854% /0.2769% | 0.6329% /0.3574% |
| FlashQLA, auto and CP-off | 1.4563% /1.1505% | 0.6329% /0.3561% |
| FlashInfer auto | 0.4854% /0.3810% | 0.6329% /0.3561% |
| FlashInfer CP-off | 0.4854% /0.3362% | 0.6329% /0.3561% |

Every arm passes the unchanged criterion,8/8 repeated fingerprints, and
post-capture checking of every result. Libraries need not be bit-identical to
one another. FlashQLA's larger weak-gate error is material and retained in the
comparison; no tolerance was relaxed to admit it.

FlashInfer requires FP32 **alpha=exp(g)** and FP32 beta, flattened Q/K/V and
VK state. Native-ABI timings use CPU-prepared representations outside capture;
alpha log-roundtrip error is2.24e-8/0 for weak/strong inputs. The additional
GPU-adapter arm includes both casts and exp inside forward, not free hidden
device work. CPU/GPU exp may round differently (different fingerprints), so
both arms are independently checked against the same recurrence. VK->KV is
only a CPU oracle interpretation. FlashQLA receives log gate/KV state directly;
backward-only CP cache is disabled because this project is forward-only.

Numerical negatives: treating exp(g) as log-gate, transposing state incorrectly,
and replacing O with zero each fail the real numerical checker.

## Timing limits, integrity and artifacts

All216 captured forwards /864 GPU kernels are assigned exactly once.312
concurrency/telemetry observations across the four final runs found no foreign
GPU PID. The first warmup run saw PID44469 and was invalidated, not selected
for timing. Two provenance-helper failures before capture were fixed and
retained as failed runs; no speed result came from them. Polling every0.2s is
not an absolute reservation against shorter foreign work. GPU idle at finish.

The kernel-sum ranking is **not an eager API latency ranking**. FlashQLA auto
has median GPU activity span1121.3/1123.4us, including approximately949/952us
of internal gaps and7.14/7.18us of separate memory operations. Its CP-off spans
404.2/398.2us. FlashInfer auto spans162.0/167.9us; CP-off is a single kernel.
Host/submission gaps in this synchronized eager nsys protocol are reported,
not used as kernel time, and not extrapolated to a graph-captured deployment.

Physical device:H800 PCIe114 SMs,
UUID`GPU-1d5fdef3-4899-79d9-19e6-c9c815b2a59c`, driver595.71.05,
Torch2.8.0+cu128, CUDA12.8.93, nsys2025.1.1. FlashQLA pinned
`a97c9783` uses TileLang0.1.9/TVM-FFI0.1.9. FlashInfer pinned`5d9f8c8d`
(version.txt0.7.0) uses CuTeDSL4.7.0/TVM-FFI0.1.11. Distinct isolated venvs
preserve both declared dependencies; system Torch and original cuLA environment
were not replaced. Full package inventories are archived.

Official git source archives were verified against every extracted regular
file. Hashes bind our original binaries,7 loaded TileLang executables, and
5 FlashInfer JIT objects containing actual CUDA ELF images. Exporting those
objects is outside capture and changes no kernel/compiler options. Full
symbol names and raw traces are preserved even where the committed summary
uses symbol prefixes plus SHA256.19 host/measurement tests pass, including
missing/duplicate denominator, hidden helper, no-kernel, foreign-work,
wrong binary/target and host-object-without-GPU-code negatives. Local Python3.10
lacks Torch for the legacy identity test; its actual Torch3.12-environment
rerun passes, not a fabricated local PASS.

Machine-readable evidence: `dev/backends/sm90_library_nsys_20260926.json`.
Raw local root:`/workspace/gdn-sm90-library-compare-artifacts-20260926`.
Final runs:`qla-weak-final`, `qla-strong-final`, `fi-weak-final3`,
`fi-strong-final3`. Archive:`nsys-library-comparison-evidence.tar.gz`, SHA256
`1a45f3c8507e94513c65e8fd7da63c96ab52d5f57fbc72adf4d38c032667ab8e`.
Remote root:`/workspace/gdn-sm90-library-compare-20260926`.
Runner:`tools/run_sm90_library_nsys.sh`; preregistration:
`docs/SM90_LIBRARY_NSYS_PLAN.md`. Set explicit WORK/FAMILY/GATE/OUT and the two
admitted extension paths; output directories must be fresh. No device build or
code change is implicit in the runner.

## Next implementation boundary

1. Make FlashInfer's **GDN-specific non-CP complete dataflow** the first
   comparison target, not just its WGMMA instruction or a bigger grid. Audit
   math, gate placement, inverse algorithm, stage DAG, accumulator lifetime,
   register spill/codegen and memory delivery against our scalar-KDA-derived
   loop. Keep the current binary as a controlled baseline.
2. Implement admitted changes in a separate pure-C++/CUTLASS SM90 algorithm
   provider. Retain the old cuLA-derived provider for A/B; do not entangle
   algorithm selection with architecture dispatch. Target this workload's
   best reference,111-112us kernel sum; no promised speed before measurement.
3. Treat V-splitting, inverse precompute and CP as explicit alternatives.
   Count all helpers/fixup and select by complete forward. Expand to a registered
   shape/gate/initial-state matrix before claiming the strongest SM90 backend;
   one constant-gate fixture does not prove that claim.
4. Transfer proven **algorithmic** changes (work decomposition, state/value
   tiling, intermediate lifetime, gate reuse) into independent SM80/PPU1.0
   candidates. Implement delivery with native AIU/TSM contracts. Keep TMA,
   WGMMA, Hopper barriers and warpgroup role scheduling in the SM90 backend.
   PPU1.0 still needs its own numerical/native-code and device-speed admission;
   an H800 win does not admit a PPU default.
