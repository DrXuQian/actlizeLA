# Non-CP residual reassociation (opt-in, forward only)

Parent: `3641826`; incumbent `state-pipeline` / mask62960. The original
reset/scan dispatch and every WY delivery remain unchanged. New public entry:
`gdn_chunk_residual`, backed by `_gdn_wy_ppu.residual`, not another WY mask.
PPU device correctness/performance are **NOT RUN** at local implementation.

Device follow-up: [same-binary residual/control/FLA ACU analysis](PPU_GDN_RESIDUAL_ACU_20260924.md).
Numerics PASS; all-kernel sums230.58412us residual vs255.22647us control and
224.30883us FLA. The1.5x target is NOT MET; default routing remains unchanged.

## Reference and exact scope

[FlashQLA pinned reference](https://github.com/QwenLM/FlashQLA/tree/a97c9783bbcc42fa8fbfe895dfc674131e376b5c/flash_qla/ops/gated_delta_rule/chunk)
uses gated residuals rather than materialized W/U (`hopper/fused_fwd.py`).
This experiment adopts that reassociation only. It retains our already-admitted
gated inverse solve, TF32x3 off-diagonal merges, separate output kernel and
non-CP serial chunk recurrence. It does **not** claim to port FlashQLA's entire
ungated-solve / fused-output / CP / Hopper execution structure.

Let P be the inverse of `I + strict_lower(beta_i * dot(K_i,K_j) * exp(g_i-g_j))`,
B diagonal beta and D diagonal cumulative gate exp. Real arithmetic gives:

```
old: W=PBDK; U=PBV; Vnew=U-WH
new: R=B(V-D(KH)); Vnew=PR
```

P and B are not commuted. No reset approximation, history truncation, gate
threshold, normalization or input quantization is added. Initial-state and GVA
semantics are unchanged. Natural-log gates remain BF16 or FP32 inputs.

## Explicit numerical contract

Identifier: `gated-inverse-residual-bf16-v1`.

- Reuse BF16(P) from the same gated solve and BF16(H) snapshots; keep recurrent
  H accumulators and final state in FP32.
- KH accumulates BF16 inputs into FP32, R rounds **once** to BF16 after
  `beta * (V - exp(prefix)*KH)`; PR accumulates into FP32.
- Publish BF16(Vnew); form BF16(scaledV) from **FP32** Vnew, not the published
  BF16 value. Retain the old output kernel's rounding and scale1/sqrt(128).
- This is not raw-bit equivalent to BF16(W)/BF16(U) materialization. Admission
  is the original independent recurrent output/state oracle and unchanged2%
  max/max relative-error gate, plus eight-repeat bit stability. Existing WY
  delivery variants must still equal scalar WY bits.
- Local rounded matrix reference is numerical feasibility evidence, not a
  simulation of exact device MMA/exp/FMA instruction rounding. Device gate is
  mandatory. Passing finite fixtures is not a universal BF16 error bound.

## Execution, memory and costs

Priority workload: B1/S2048/Hk16/Hv32/K=V128/C64, g=-0.1 and-1.0; no auto route.

| Stage | Incumbent | Residual |
|---|---|---|
| prefix | chunk-parallel, unchanged | same native kernel |
| solve | chunk-parallel, unchanged | same native kernel |
| W/U | two chunk-parallel matrix products | absent |
| state | W@H, Kᵀ@scaledV | K@H, P@R, Kᵀ@scaledV |
| output | chunk-parallel | same native kernel |

Five launches become four. Per logical head/chunk, one C64×C64 @ C64×K128
product is removed in total (1,048,576 logical FLOPs). Across1024 chunks this
is1,073,741,824 FLOPs. W/U's32MiB logical vector stores disappear. This is NOT
the net hierarchy-traffic reduction: P, snapshots and Vnew still exist.

Inverse has its **own allocation**, separate from H snapshots. Sharing them
would race across four independent V-slice CTAs even if each CTA loads P before
writing its own H slice. v1 retains the unchanged solve's padded BF16-element
pitch16384 (only4096 inverse elements used). Inverse allocation32MiB replaces
W16MiB+U16MiB; total temporary allocation is therefore NOT reduced in v1.
No U allocation or W/U producer is hidden in the residual launch path.

Residual state:128threads,45568B shared (incumbent49408B). AIU.swzl is paired
with matching ld.swzl. Register-produced snapshot/R/scaledV stores use that
same proved physical layout. Four V-slice CTAs each own all128 K rows for32
independent output columns; no inter-CTA reduction or lock is introduced.
Async input completion and all four CTA lifetime handoffs are conservative;
this experiment does not also retune the pipeline.

**Risk:** P@R is now on the cross-chunk serial dependency chain. Fewer total
products and stores do not guarantee lower latency. A slower residual arm is
a valid result and must stay visible; it cannot replace the incumbent by fiat.

## Local verification

Evidence: `/workspace/gdn-wy-residual-evidence-20260924` (not device results).

- SDK2.1.1 real hgcc compile/link:23 native WY images. All22 parent native
  instruction+operand sequences exactly unchanged, including solve/output.
- New state:242 vector registers, zero stack,2092 static instructions,
  40 BF16 MMA sites,4 AIU SWZL sites,56 matrix-load sites,5 barrier sites
  (includes optional final publication), one async commit wait. SDK-dependent
  resource numbers are not box measurements or dynamic instruction counts.
-61 CPU algebra/boundary cases: tails, GVA, nonzero H, no/weak/strong/variable
  decay, normalized K and variable beta. FP64 agrees with independent token
  recurrence at rtol1e-9/atol1e-11; rounded maximum output/state errors across
  the base+target inventory0.008878/0.004194 (<0.02).49/56 base fixtures differ
  from old WY, constructively confirming that old RAW-BIT cannot be claimed.
- Four semantic negatives: omitted gate, beta moved outside P, omitted inverse,
  reset history; all rejected by the same numerical gate.
- l020 uses real MMA traits / actlize load simulator:6144 owned shared values,
  163840 product/output/reduction-atom cells,1064960 group/tail cases; missing
  warp/slice, wrong swizzle/K order/pitch/tail all red. Existing l016 covers
  paired AIU cube descriptors, including the inverse64×64 tile.
- Source/native negatives bind allocation separation, actual new API, async
  wait, reader retirement, rounding, native load pair and extra PR arithmetic.
  Final full rerun:12/12 CTest,74 Python contracts,7 compiler-dialect tests,
  45 old-WY algebra cases,61 residual cases,305 original source controls,
  all23 WY and15 original native images. Evidence `verify-final.log`.

## Box command and preregistered interpretation

From this repository, with the normal PPU SDK and installed FLA environment:

```bash
git pull --ff-only
JOBS=16 DEVICE=0 bash tools/run_ppu_residual_acu_box.sh
```

The script builds locally on the box, runs the original and old WY RAW-BIT
gates, then30 residual cases and same-input8-repeat scalar/pipeline/residual/FLA
admission at both gates. API timing is not run. It then captures pipeline,
residual and FLA sequentially, one public call per arm, g=-1 by default.
`GATE=-0.1` selects the weak-gate capture; do not overlap the two runs.
Default profiler is exactly `/sim/eec/shared/junfu.qx/asight/bin/acu`.

Reports, receipts, source/binary/tool/device identities and native ISA/resources
are saved under a fresh `/workspace/gdn-residual-...`; upload the printed
`UPLOAD=.../acu/acu.tar.gz`. No manual identity fields, PPUProfiler or pasted
CSV are required. Missing profiler/failed numerics/changed loaded binary fail
closed. Compile/numerical failures retain their logs; capture failures retain
an INCOMPLETE diagnostic archive, not a measurement verdict.

Compare **sums of all kernel durations**: expected main stage counts5 control,
4 residual,7 FLA on the currently measured FLA version (verify actual names;
never silently truncate other required launches). Target FLA/residual>=1.5.
Also report state and total latency, MMA/instruction counts, shared/global
traffic, registers/stack, CU frequency and missing/repeated kernels. A lone
profile is descriptive evidence, not an uncertainty-resolved universal winner.
Any loss, numerical failure, altered reference selection or incomplete stage
inventory stays explicit; no threshold or default routing is changed afterward.
