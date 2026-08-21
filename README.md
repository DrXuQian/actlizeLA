# actlizeLA

`actlizeLA` is a forward-only, pure CUDA C++/CUTLASS implementation of
chunked Gated Delta Networks for PPU. It contains no Triton, CuTeDSL, Python
runtime dependency, autograd entry, or backward kernel.

The public ABI intentionally retains the proven
`quactlize_ppu_chunked_gdn_fwd_bf16_v1` symbol so extracting the code does not
silently break existing consumers.  The explicit-workspace v2 entry prepares
the common A/W/P seam once and then runs the same split-V64 recurrence. The
standalone shared library is named `libactlize_la_ppu.so`.

The canonical C++ header root is `actlize_extensions/`.  The retained
`quactlize_ppu_*` spelling is limited to the stable public C ABI; it is not an
alternate C++ include tree.

## Stable v1 scope

- forward inference only;
- BF16 Q/K/V/output and FP32 recurrent state;
- chunk size 64, key/value head dimensions 128;
- fixed-length sequences and grouped-value attention;
- one CTA owns one complete `(sequence, value_head)` recurrence chain;
- QK and KK products use the actlize PPU0010 BF16 AIU collective;
- generated-operand products use the independently executable scalar path in
  stable `main`.

The blocked unit-lower inverse under `dev/gates/reference` is a forward WY algebra
oracle, not backward propagation. See [the detailed contract](docs/PPU_CHUNKED_GDN.md).

An experimental all-AIU forward conversion is preserved on
`agent/all-aiu-forward`. It is deliberately not on stable `main` until an
actual hgcc build, PPU RAW-BIT verdict, and ACU opcode/spill check close it.

### Status of this experimental branch

This branch routes six generated-operand product kinds (11 logical product
instances) through the production PPU `TiledMma`, for 1,152 additional
`m16n16k16` operations per full chunk and 1,408 BF16 dense-forward MMA
operations including QK/KK. The inverse adds six blocked products implemented
as 40 `m16n16k8` TF32-input/FP32-accumulate operations, leaving only the four
16x16 triangular dependency chains scalar. Local algebra, exact device-type
instantiation, physical fragment delivery, TF32-aware inverse authority, and
negative controls pass. It is **not a release verdict**: the real hgcc body,
PPU output/state, the separate 1,408-BF16/40-TF32 opcode denominators,
registers, and spills remain device postconditions.

## Dependencies

- the pinned `third_party/actlize` submodule;
- PPU SDK with `hgcc` and the HGGC runtime for PPU builds;
- CUDA/NVCC only for the local independent device-reference gates.

Clone with the pinned dependency:

```bash
git clone --recurse-submodules git@github.com:DrXuQian/actlizeLA.git
cd actlizeLA
```

## Local verification

The host algebra oracle has no PPU SDK dependency:

```bash
mkdir -p /workspace/actlizeLA-cmake-local
cmake -S . -B /workspace/actlizeLA-cmake-local \
  -DACTLIZELA_ENABLE_PPU=OFF
cmake --build /workspace/actlizeLA-cmake-local -j16
ctest --test-dir /workspace/actlizeLA-cmake-local --output-on-failure
```

With NVCC available, run the complete local proof bundle:

```bash
OUT=/workspace/actlizeLA-local-gates bash tools/run_local_gates.sh
```

The gates cover recurrence/WY algebra, every C16/C32/C64 tail, public ABI,
the exact device type, and independently anchored PPU operand/destination
ownership and compact physical fragment delivery. They also distinguish the
blocked inverse's exact TF32 fixture from its bounded non-exact numerical
seam. They do not substitute for the PPU-only opcode, shared-memory, register,
or runtime verdicts.

## PPU build and correctness

```bash
git submodule update --init --recursive
BUILD_DIR=/workspace/actlizeLA-ppu-build \
PPU_SDK=/usr/local/PPU_SDK JOBS=16 bash build.sh

LD_LIBRARY_PATH=/workspace/actlizeLA-ppu-build:/usr/local/PPU_SDK/lib \
  /workspace/actlizeLA-ppu-build/test_ppu_chunked_gdn_abi
```

Or run the source-bound box gate, which builds and links the standalone
library before checking a 64+1 tail, GVA 1:2, zero/nonzero state, two WY
fixtures, workspace admission negatives, and raw-bit v1/v2/v3/v4 parity:

```bash
OUT=/workspace/actlizeLA-l205-box \
  bash dev/gates/run_l205_ppu_chunked_gdn_abi.sh --box
```

## Performance runner

```bash
OUT=/workspace/actlizeLA-gdn-perf \
  bash tools/run_ppu_chunked_gdn_perf_box.sh --box
```

Set `GDN_PIPELINE=triton-aligned`, `two-stage`, or `legacy` with a distinct
`OUT` to run v4, v2, or v1; the default remains the four-stage v3 control. All use
the exact
`B1,T2048,Hqk16,Hv32,K128,V128,C64` Qwen3.5-35B-A3B shape.

Artifacts stay under `/workspace`; no runner uses `/tmp` or `mktemp`.

The v1 PPU tactic partitions each V128 recurrence head into two independent
BV64 CTA owners.  v2 adds a 1,024-CTA common prepare stage for the published
Qwen3.5-35B-A3B `B1,T2048,Hqk16,Hv32,K128,V128` shape, then retains 64 BV64
recurrence CTAs.  Its fixed interpretation is recorded in
`dev/gates/QWEN35_TWO_STAGE_PERFORMANCE_PREREGISTRATION.md`.
v3 moves U and O onto independent 2,048-CTA grids, leaving only WH and the
state update in the 64-CTA ordered H chain.  Its fixed interpretation is in
`dev/gates/QWEN35_FOUR_STAGE_PERFORMANCE_PREREGISTRATION.md`.
The first v3 timing was slower than v2, so the follow-up keeps that DAG and
shrinks only the stage-local shared ledgers to
`57856/41472/98816/66048 B` for prepare/U/H/O.  Its before-the-run verdict is
in `dev/gates/QWEN35_FOUR_STAGE_STAGE_SMEM_PREREGISTRATION.md`.

v4 is a separate post-cumsum alignment subject rather than another v3
storage optimization. Its four launches match flash-linear-attention's
KKT/solve, W+U, recurrent-H, and O boundaries. For the Qwen shape their grids
are exactly `1024/1024/64/2048`; the global seam is A/W/U/H-start/Vnew and
does not materialize causal QK/P. The local CUDA oracle proves v4 raw-bit
equivalent to the independent token recurrence on exact fixtures. The PPU
performance target is the measured Triton post-cumsum sum (~230.04 us); it is
not claimed until a box report establishes it.

To capture the same mathematical shape under ACU with exactly one public-ABI
invocation (one kernel for v1, two for v2, four for v3/v4):

```bash
OUT=/workspace/actlizeLA-gdn-qwen35-four-stage-acu \
ACU=1 GDN_ACU_SHAPE=1,2048,16,32,qwen35-35b-a3b-t2048 \
  bash tools/run_ppu_chunked_gdn_perf_box.sh --box
```

For the aligned subject:

```bash
OUT=/workspace/actlizeLA-gdn-qwen35-triton-aligned-acu \
ACU=1 GDN_PIPELINE=triton-aligned \
GDN_ACU_SHAPE=1,2048,16,32,qwen35-35b-a3b-t2048 \
  bash tools/run_ppu_chunked_gdn_perf_box.sh --box
```

For a direct ACU comparison against FLA/Triton, use the same registered
`B1,T2048,H16,HV32,K128,V128,C64` fixture and the same post-cumsum API
boundary.  The FLA runner first autotunes outside ACU into its private
`/workspace` cache; a tuning-cache miss inside the profiled process voids the
report instead of adding candidate launches to it.

```bash
OUT=/workspace/actlizeLA-fla-gdn-post-cumsum-acu \
FLA_SCOPE=post-cumsum \
  bash tools/run_fla_chunked_gdn_acu.sh

OUT=/workspace/actlizeLA-cpp-gdn-post-cumsum-acu \
ACU=1 GDN_PIPELINE=triton-aligned \
GDN_ACU_SHAPE=1,2048,16,32,qwen35-35b-a3b-t2048 \
  bash tools/run_ppu_chunked_gdn_perf_box.sh --box
```

The resulting reports are respectively
`fla-post-cumsum.report.acurep` and
`qwen35-35b-a3b-t2048.report.acurep`. Each must contain four ordered stages:
KKT+solve, W+U, recurrent H, and O. To profile FLA's complete public forward
including its cumsum launch, rerun the first command with `FLA_SCOPE=full` and
a new `OUT`; that five-kernel report is not the primary four-stage timing
denominator.
The FLA source authority is resolved from the `fla` package imported by the
selected `PYTHON_BIN`; set `FLA_ROOT` only when that package is intentionally
used from a separate source checkout.

## Source authorities

- Forward recurrence and chunk/WY semantics: `fla-org/flash-linear-attention`
  commit `033b19e81239b13971a410e55dd6c178d430b9d4` (MIT).
- Fused-kernel boundary and one-CTA state ownership: `inclusionAI/cuLA`
  commit `4cc51c5eff79761744ae630536c808a3f8039a0f` (Apache-2.0).
- PPU transport and AIU atom: pinned actlize/CUTLASS submodule.

The external projects are design/semantic authorities; their source is not
vendored into this repository beyond the explicit actlize submodule.
