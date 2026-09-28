# Named SM90 implementations and PPU1.0 transfer

This cleanup integrates the three retained H800 implementations into one
source tree. The evidence remains the fixed 14-workload campaign in
[the closure report](SM90_H800_CLOSURE_20260927.md), not a new device benchmark.
H800 has been shut down. No device was restarted for this refactor.

## Architecture and ownership

| Layer | Responsibility | Must not own |
|---|---|---|
| Common Python forward contract | Inputs, state semantics, explicit backend/algorithm/configuration | GPU-specific intermediate layouts |
| `cuda_sm80` | Original scan/reset implementation | Hopper primitives |
| `ppu_aiu` / PPU1.0 | Native AIU + paired shared readers, PPU-specific experiments | TMA/WGMMA emulation |
| `sm90` | Scalar-GDN algorithm, TMA, async WGMMA, warpgroup role scheduling | PPU1.0 collective compatibility |
| SM90 CUDA dependency | Explicit hash-bound CUTLASS headers | Implicit actlize fallback |
| PPU1.7 dependency | PPU CUTLASS3.6.0/10700 | Claiming CUDA source-check is PPU execution |

The existing original, WY and residual APIs/defaults remain unchanged.
No forward-only task adds backward code. Experimental sources/evidence in
older branches remain archived; they are not copied into the shipping tree.

## Four named builds, one tuned collective

One configuration is compiled into each binary: no runtime device switch,
no three-way copied mainloop, no experimental S-number macro.

| Build configuration | Source authority | CTA work and distinguishing feature |
|---|---|---|
| `control` (default) | pre-cleanup main | Original nonsliced SM90 control |
| `value64` | c490c8d / S38 | Two independent V64 slices; 232-register auxiliary role |
| `value64-local-inverse` | 6317ea4 / S55 | V64; local final inverse reduction; paired O2/KV tail |
| `value128-paired` | d481a42 / S69 | V128; packed NewV BF16 conversion before retile; paired O2/KV tail |

`configuration.cuh` owns compile-time traits. `value_types.cuh` assembles the
collective/scheduler/options. `scalar_gdn_aux.cuh` owns QK/KK, conditioning,
inverse publication and gate production; `scalar_gdn_state.cuh` owns state
and output math. Small helpers own the value-slice map, relative-gate map,
fragment conversion, inverse reduction and ordered barriers.

Async lifetime is part of the algorithm: paired O2/KV uses distinct unscaled
and scaled operands until their shared wait. The local inverse retains two
K16 FP32 partials and their **separate FP16 roundings**. Neither is replaced
with a numerically different summation merely to shorten the code.

The fixed H800 inventory selected V64-local-inverse for seq8192/heads16, V128
for the four high-head workloads, and V64 for the other low-head cases.
That historical table is not a new general-purpose selector. Unknown shapes
are not automatically assigned a measured winner, and PPU1.0 never inherits
H800's winning configuration.

## Explicit build and use

```bash
python tools/build_gdn_sm90.py \
  --target cuda_sm90 --configuration value64 \
  --compiler /usr/local/cuda/bin/nvcc \
  --cutlass-root /path/to/cutlass \
  --out /workspace/gdn-sm90-value64
```

The measured campaign used CUTLASS **4.3.2** headers. Rebuilding against
another release is another build, not measured-code identity.
Full compilation, host linking and import precede a complete receipt.
`--device-only` is explicitly compile/codegen-only.
CMake exposes `GDN_SM90_CONFIGURATION` and `GDN_SM90_CUTLASS_ROOT`.
For PPU1.7 use the separate `PPU_CUTLASS_ROOT` and source-check/native target
gate; do not point it at actlize.

```python
# GDN_QSA_SM90_EXTENSION identifies the exact newly built DSO.
output, state = gdn_forward(
    q, k, v, g, beta, initial_state=initial_state,
    algorithm="fused_sm90", backend="cuda_sm90",
    configuration="value64",
)
```

The module and build receipt both expose the configuration. A wrong/missing
module label fails before launch. Old experimental modules without that
field are **not** silently interpreted as control. The same explicit name
is accepted by `tools/run_sm90_gdn.py` and `tests/run_sm90_hopper_cases.py`.
Profilers obtain it from the hash-bound build receipt, not an operator guess.

Dependency identity hashes all actual include files, including untracked
headers and archive snapshots. An enclosing GDN Git HEAD is no longer
mislabeled a CUTLASS revision. Changed source/headers invalidate reuse.

## Reproducible local gates

`tools/check_sm90_host_maps.sh` compiles and executes four real CuTe host
proofs (no device invocation): value-slice ownership/public strides, local
inverse row/K-half ownership and rounding, paired-tail operand coordinates,
and the two-stage relative-gate producer with every tail length. Set
`NVCC`, `CUTLASS_ROOT`, and a fresh `OUT` under /workspace.

`dev/backends/check_sm90_configuration.py BUILD --self-test` re-disassembles
the exact hash-bound object and compares all four complete instruction
streams, including predicates, operands, ordering and resource records,
against the committed anchors. Missing specialization, changed operand,
reordered instructions and changed resources must fail.
This is not a new device-performance certificate.

Three integration mistakes were rejected by that gate:

1. A ternary between CuTe `C<128>` and `int` erased the static extent in
   the nonsliced TMA view. The restored V128/control branch retains its
   static type; the measured V64 path is unchanged.
2. Factoring the tail output address into a temporary changed native
   scheduling/register allocation even with identical opcode totals.
   The nonsliced tail retains its original assignment expression.
3. Importing a tuned auxiliary/load register budget into the base kernel
   also changed the original control (176 to 128). Configuration traits
   now preserve the control's LD/ST24 + auxiliary152 + two state168 roles;
   only tuned builds redistribute that budget.

The checker was not weakened to hide these differences. We preserve the
admitted SM90 stack/resource footprints, including existing spills; the
cleanup is not a claim that every SM90 configuration is spill-free.

The final local evidence is recorded in
[`backend_cleanup_20260927.json`](../dev/backends/backend_cleanup_20260927.json):

- All four CUDA configurations compile, link and import; **16/16** complete
  native bodies and resource records match their respective original builds.
  Four constructive native negatives per configuration are rejected.
- **171/171** Python tests, **24/24** PPU host CTests, and four real CuTe map
  proofs pass. CMake forwards all four names and the explicit dependency.
- Three tuned configurations pass CUDA source checks using PPU CUTLASS3.6.0.
  Native PPU1.7 verification is **SKIP**: its SDK/model is unavailable.
- PPU1.0 native compilation and link pass; runtime import is **SKIP** on this
  host because the SDK needs GLIBC_2.38. Device correctness/performance must
  still be admitted on the box; there is no new speed claim.

## What transfers to PPU1.0

| Technique | Decision |
|---|---|
| Once-per-token gate coefficient production | Implemented as opt-in `gate-cache`; exact expf, existing barrier; needs PPU device A/B |
| Eliminating scalar address/repeated coefficient work | Reusable method; inspect real PPU native code and charge shared/register costs |
| Packed BF16 conversion before operand retile | Do not blindly transplant: PPU's fragment layout/instructions differ; prove the actual paired reader first |
| Local final inverse reduction | Do not copy FP16 Hopper math over the current PPU TF32 inverse; preserve/verify its own rounding boundaries first |
| Async O2/KV pairing and warpgroup register redistribution | Hopper-specific; PPU1.0 synchronous MMA cannot inherit the overlap/resource claim |
| TMA and Hopper ordered barriers | Keep in SM90; PPU1.0 retains native AIU.swzl + matching ld.swzl |
| H800 fast-exp2 experiment | Not retained by performance evidence; no PPU fastmath change |

The first PPU transfer and ready-to-run ACU command are documented in
[PPU10_GATE_COEFFICIENT_REUSE.md](PPU10_GATE_COEFFICIENT_REUSE.md).
Primary performance metric remains full-forward kernel-sum, with independent
numerical admission. Static instruction reductions are diagnostic, not wins.
