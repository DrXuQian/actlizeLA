# Local admission: fused scalar-GDN source integration

Parent `b9a870b`; isolated worktree `/workspace/gdn-sm90-algorithm-20260926`.
Evidence `/workspace/gdn-sm90-evidence-20260926`. No GPU, simulator or remote
device execution was performed. This report is **not a speed verdict**.
This is the compile/local checkpoint. The user subsequently supplied a physical
H800; its separate checks are in `SM90_FUSED_GDN_H800_20260926.md`.

## Completed checks

| Check | Result / meaning |
|---|---|
| Independent FP64 chunk formula vs token recurrence | D128, 72 combinations: 12 lengths (1..257, including chunk/tail seams), three decay regimes, zero/nonzero initial state. PASS. |
| Mathematical negatives | Wrong log base, cross-chunk prefix carry, transposed state, wrong beta axis: four expected reds. |
| Actual CuTe layouts and consumer copies, with ASan | PASS against both CUDA CUTLASS and PPU3.6. All 16,384 logical gate values, state/output ownership, all64 tail sizes, auxiliary A **and** B copy layouts checked. A shifted gate row is red on both dependencies. |
| Python contracts | 104 PASS (includes eight new SM90 contracts), plus 26 backend-boundary tests and seven compiler-dialect tests PASS. No GPU dependency for these checks. |
| CUDA SM90a full body | All four BF16/FP32-gate × initial-state-present/absent kernels assembled; live WGMMA, TF32 MMA, TMA input/output, FP32 state stores and BF16 tail stores in every body. |
| PPU CUTLASS source check | Same four full bodies assembled with stock CUDA12.8 against PPU CUTLASS3.6.0 / ACOMPUTE10700. This is CUDA code, **not native PPU code**. |
| Native-code negative controls | Per image: omitted specialization, removed WGMMA/TMA-load/TMA-store/global-store, removed only FP32 state store, removed only BF16 tail store: seven reds. |
| Actual host link/import | CUDA target and PPU-fork source-check extensions link and import under Python3.12/Torch2.9. No GPU query/launch in the import check. |
| CMake selection | Both new target graphs configure successfully without the legacy AIU test/build graph. Full compilation/link above uses their registered Python builder. |
| Actual missing-TU negative | Omitting launch.o produces `undefined symbol: gdn::sm90::launch(...)` at import. FAIL, not SKIP or a successful stub. |
| Target negative | Compiling the real target receipt for SM80 fails. No fabricated architecture macros. |
| Legacy preservation | No diff in `csrc/gdn_chunk`, `include/gdn_qsa`, legacy CUDA/PPU primitives or PPU build leaf. Default API routing remains unchanged. |

The expanded-gate prototype first compiled successfully, but the final scalar
gate optimization required preserving CuTE's nested K shape. A flat-shape
attempt failed the real device-body compilation; the admitted version uses
`(32,4)` and is separately tested through the actual A/B consumers. The first
host swizzle test also exposed an **unaligned test buffer**, repaired by using
the production SharedStorage type; production shared alignment was not relaxed.
Final ASan layout checks use `-O1`. Redundant, slow `-O0` host builds were stopped
after the two `-O1`/ASan executions passed; they are not recorded as passes.

## Resource results: risks remain visible

Dynamic shared: **167,936 B**, 512 threads. Gate buffers: **512 B**, down from
65,536 B in the vector-gate prototype. The image additionally reports 1,024 B
static shared. Actual occupancy must be queried on the target device; dynamic
shared alone is not an occupancy measurement.

| CUDA-assembled dependency | Initial state | Registers | Stack | Spill store/load (ptxas bytes) |
|---|---:|---:|---:|---:|
| CUDA CUTLASS4.3.2 | absent | 128 | 264 B | 608 / 676 |
| CUDA CUTLASS4.3.2 | present | 128 | 296 B | 674 / 1,128 |
| PPU CUTLASS3.6.0 source-check | absent | 128 | 224 B | 472 / 580 |
| PPU CUTLASS3.6.0 source-check | present | 128 | 312 B | 726 / 1,216 |

Both gate dtypes have these respective resource counts. These are **CUDA
compiler results**, not PPU register counts, dynamic spill traffic or ACU
instruction counts. Spills prevent a performance-ready claim. No register
budget, precision threshold or legacy routing was changed to hide them.

Native-code check counts are static sites: no-initial kernels have80 WGMMA
sites, initial-state kernels112; all have216 TF32-MMA,6 TMA-load and8 TMA-store
sites. Both dependencies retain all four complete entries. These are liveness
checks, not executed-work estimates or a CUDA/PPU throughput comparison.

## Toolchain/identity

- Reference cuLA: `79be249e61453808e18e5cef7702b363239e7d8d`.
- PPU fork used in the final source check:
  `1436fd98ed81e9099ff95c4fbe5accf4b2ded9ba`, CUTLASS3.6.0.
- CUDA compiler: `/usr/local/cuda-12.8/bin/nvcc`,12.8.93; real
  `-gencode=arch=compute_90a,code=sm_90a`, assembly included.
- Complete per-source/extension hashes, compiler hash, flags, dependency
  version/revision/diff and target identity are in each `build.json`.
- Final CUDA extension SHA256:
  `0518dd5352450977bd9e7f5281ce2df4cf2619686e0bb9d3726cd5e521eabe82`.
- Final PPU-fork **source-check** extension SHA256:
  `4151c7c64b5ec6e702d8f61432d363de75af305c4c46dcf094514a68f00851a0`.

Local files: `cuda-admitted/`, `ppu-source-admitted/`, `final-algebra.log`, `final-contracts.log`,
`final-boundaries.log`, `legacy-dialect.log`, `layout-*-o1-asan`,
`missing-launch.log`, `wrong-target.log`, `native-probe/target.log`.
The replay with `--reuse-device` verifies exact source/flags/dependency/object
hashes, rechecks the target and each assembled body, and relinks/imports the
real host binding. Changed source, target or object negatives reject reuse.
Both final `*-admitted` builds are fresh complete rebuilds after source whitespace
normalization, not reused objects; their final source hashes bind the committed bytes.

## Uncovered, not green

**Native PPU1.7 compile: SKIP — installed SDK lacks the required target.**
The SDK2.1.1 compiler lists vm_10/vm_15. Its CUDA wrapper fails the same-flags
SM90a receipt (legacy architecture and overridden ACOMPUTE selection). The
requested native build exits nonzero; that failure is preserved, not rewritten
as a successful source check. Kernel syntax/body failures are never automatically
classified as environment skips.

**Device/simulator numerical admission, repeat stability and performance:
NOT RUN.** CPU algebra and assembly cannot prove async races or actual BF16
rounding. Use the single-call harness with a matching target/toolchain, first
tail/GVA/nonzero state, then B1/T2048/Hk16/Hv32/D128. Retain the 2% oracle gate,
verify the trace's launch count, then profile resources/spills. No 1.5x claim,
automatic route promotion or production PPU1.0 replacement is made.
