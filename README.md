# actlizeLA

`actlizeLA` is a forward-only, pure CUDA C++/CUTLASS implementation of
chunked Gated Delta Networks for PPU. It contains no Triton, CuTeDSL, Python
runtime dependency, autograd entry, or backward kernel.

The public ABI intentionally retains the proven
`quactlize_ppu_chunked_gdn_fwd_bf16_v1` symbol so extracting the code does not
silently break existing consumers. The standalone shared library is named
`libactlize_la_ppu.so`.

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
`m16n16k16` operations per full chunk and 1,408 dense-forward MMA operations
including QK/KK. Local algebra, type instantiation, and independently anchored
A/B/C coordinate ownership pass. It is **not a release verdict**: the real
hgcc body, PPU RAW-BIT output/state, the 1,408-opcode denominator, registers,
and spills remain device postconditions.

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
ownership. They do not substitute for the PPU-only opcode, shared-memory,
register, or runtime verdicts.

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
fixtures, and seven admission negatives:

```bash
OUT=/workspace/actlizeLA-l205-box \
  bash dev/gates/run_l205_ppu_chunked_gdn_abi.sh --box
```

## Performance runner

```bash
OUT=/workspace/actlizeLA-gdn-perf \
  bash tools/run_ppu_chunked_gdn_perf_box.sh --box
```

Artifacts stay under `/workspace`; no runner uses `/tmp` or `mktemp`.

## Source authorities

- Forward recurrence and chunk/WY semantics: `fla-org/flash-linear-attention`
  commit `033b19e81239b13971a410e55dd6c178d430b9d4` (MIT).
- Fused-kernel boundary and one-CTA state ownership: `inclusionAI/cuLA`
  commit `4cc51c5eff79761744ae630536c808a3f8039a0f` (Apache-2.0).
- PPU transport and AIU atom: pinned actlize/CUTLASS submodule.

The external projects are design/semantic authorities; their source is not
vendored into this repository beyond the explicit actlize submodule.
