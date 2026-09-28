# PPU port of the original optimized GDN

New, explicit candidate: [C64 WY weak-path alignment](PPU_WY_ALIGNMENT.md).
It has a separate API/binding and does not replace the original implementation
or the reset/scan admission described below. Device performance is pending.

This port compiles the **original** `csrc/gdn_chunk/` kernels and
`gdn_ops.cu` host dispatcher for PPU. The simplified backend from
`a6ac2de` has been removed: it reproduced the broad affine-scan algorithm,
but did not retain upstream's important optimizations. It is not the path
described below.

The source authority is upstream `aa04271`. This is forward-only GDN,
BF16, D=128, C=16, zero initial state, optional final state, including GVA
(q/k heads divide value heads). QSA and output-gate kernels are **not** PPU
ported by this change. The user reports PPU numerical admission PASS after the
retile-view repair below; corrected performance measurements remain pending.

The auto API retains upstream's Hv=32 restriction on its serial path; the
explicit two-level API supports the other positive divisible head counts.

Same-input comparison update: the user reports g=-0.1 ours 925.020 us vs
FLA 726.992 us, and g=-1 ours 457.340 us vs FLA 746.904 us. These are full
public-API event spans, not profiled kernel durations. For the next diagnostic,
use the [one-command ACU capture and tar handoff](PPU_GDN_ACU.md). It preserves
both implementations, their shared fixture and the original tolerance.

## What is retained

| Optimization | Actual shared source | PPU change |
|---|---|---|
| Strong-decay, B-only last-chunk reset preparation; no six-tensor chunk workspace on the accepted path | `gdn_scan_stage1_reset.cu` and original `gdn_ops.cu` | Native loads/MMA and shared layout |
| Eight-warp fused replay; `s_reg[kCPW][K_BLOCKS_TOTAL]` survives the entire group | `gdn_kernel.cu::gdn_recurrence_fused_kernel` | Native result-to-operand register permutation |
| `raw[2]` / `in[2]` asynchronous next-chunk prefetch, wait cadence, current cooked tiles | Same fused replay | Native PPU cp.async; 16-byte transfers remain contiguous |
| Six 16x16 MMA products forming L²/L⁴/L⁸ Neumann inverse | Original prepare and fused reset helpers | Native F16 MMA with explicit A/C register-order conversion |
| Full A/B transfer and packed `A2 @ [A1|B1]` scan | Original stage1/stage2 TUs | Target primitives only |
| Blelloch exclusive scan for power-of-two group counts; Hillis–Steele inclusive otherwise | Original `gdn_ops.cu` | No dispatch rewrite |
| Serial short/weak-decay path and existing sequence/decay thresholds | Original `gdn_ops.cu` | Same C++ host wrapper, compiled/linked for PPU |

Upstream's reset is an **approximation with numerical admission thresholds**,
not an order-independent/raw-bit guarantee. The gt metric and subsequent
max-|A| check are preserved; this port does not establish a universal error
bound for arbitrary q/k/g. Device tests use the existing repository's 2%
max-absolute-error / reference-max criterion. Repeat stability and GVA
expansion equality are separately checked bit-for-bit.

For S=2048, upstream auto dispatch uses GC=8 two-level/reset when
`mean(-g) >= 0.55`, otherwise serial. Those are original A800-tuned thresholds,
**not a PPU performance optimum**. The benchmark reports both decay domains
and includes host dispatch synchronization; it must not be presented as
kernel-only latency.

## Target seam

`csrc/gdn_chunk/gdn_target.cuh` selects NVIDIA or PPU primitives at compile
time. NVIDIA continues to use its original SM80 m16n8 operations and layouts.
PPU uses actlize's native m16n16 BF16/F16 operations, not scalar-gather GEMMs.

- `include/gdn_qsa/ppu/shared_copy.cuh`: independent swizzled 16x16 cubes,
  transpose views, hardware swizzled matrix loads and native accumulator stores.
- `include/gdn_qsa/ppu/fragment.cuh`: explicit native A/B/C coordinate maps
  and cross-lane register conversion, without a shared-memory state spill.
- All producer and consumer layouts change together. Global inputs and public
  outputs are unchanged; internal prepared workspace belongs to this backend.
- PPU's native result layout differs from its input layout. Treating the
  registers as interchangeable is not valid, even though both fragments have
  eight half elements per thread.

This is optimization-structure parity, **not identical ISA or a speed claim**.
PPU register shuffle cost and hardware load semantics still need device
measurement. An inherited SDK diagnostic says the optional `.LLC::128B`
cp.async hint is ignored; cp.async itself is emitted. No scalar fallback is
silently selected.

## Local evidence and limits

`scripts/verify_ppu_backend_local.sh` runs:

1. Independent affine-algebra tests: 55 cases with inverse-sign, scan-order
   and replay-prefix negatives. These are algebra tests, not device execution.
2. L006 actual CuTe native fragment/layout tests: every lane/slot in normal
   and transposed cubes across the selected 16/32/128/256 extents. Swizzle
   addresses are independently anchored to actlize's hardware-facing cube
   simulator, and 16-byte contiguity is exhaustive for those extents.
3. A six-product native-fragment Neumann chain versus independent forward
   substitution, on an exactly representable dyadic fixture. All 256 results
   match; leaving the result in C order corrupts 111 elements.
4. Original-source comparison: all 305 control expressions across the seven
   original TUs, launch geometries, wait/barrier cadence, and the complete
   host dispatch body remain upstream-identical. Removing a wait, a buffer,
   changing a loop bound, or moving state into the chunk loop must turn the gate red.
5. Actual CuTe register-view aliasing: 3072 cells (BF16 + FP16, all 32 lanes,
   all six copy roles). Writes through a returned `retile_D` view must reach
   its owning fragment; a saved `retile_S` view must observe subsequent owner
   writes. The exact legacy reference-return implementation fails all 1536
   BF16 cells in each direction. The CPU zero-result consequence reproduces
   the first box failure's exact output/state error pair, using the real
   device test's fixture and comparator.
6. hgcc PPU0010 compile **and link** of all six original device TUs plus the
   original PyTorch host wrapper, then resource/ISA admission.

Local SDK 2.1.1 emits 15 kernel symbols including both reset and full-scan
paths. After the retile-view repair, default C16 kernels have zero stack;
serial recurrence uses 176 vector registers/thread and fused register replay
uses 204 (the incorrect pre-fix binary used 148 and 182 respectively).
These are compiler resources, not a performance result. The original experimental C32
recurrence has a 144-byte stack and is explicitly **not PPU performance
admitted**; auto never chooses it. The resource report prints it, rather than
hiding it or treating it as a C16 regression.

The source gate is tied to `aa04271`; a shallow checkout must contain that
commit to run it. These checks do not prove PPU runtime load/execute behavior,
rounding, race freedom or speed. This local host cannot import the SDK runtime
(GLIBC_2.38 / GLIBCXX_3.4.32 are unavailable), and it has no PPU. Compile/link
is PASS; local runtime import and device execution are **unavailable**, not PASS.
Use the matching PPU SDK/PyTorch container on the box.

The SDK's executable named `nvcc` is its PPU compatibility driver (a trial
`-arch=sm_80` invocation still reported `compiling for ppu001`), not independent
NVIDIA codegen validation. NVIDIA execution/rebuild is not claimed here.

## First device failure: owning Tensor copied instead of viewed

At `fc8cbac`, the first box case (B2/S65/Hk1/Hv2, C16/GC2, g=0,
Hillis-Steele) reported output/state errors `[0.9999995827674866, 1.0]`.
This is a defect in this port's shared-copy adapter, not an upstream algorithm
change or a reason to weaken the existing 2% criterion.

The old `retile_D(T& t) -> T&` returned an **owning** CuTe register tensor.
Original call sites use `auto view = retile_D(fragment)`. Value deduction
copies the owning `ArrayEngine`, so the load fills a detached register array;
the subsequent transform/MMA reads the unfilled original. `retile_S` had the
same contract error: a retained `auto` source became a snapshot rather than a
live view. The repair returns `make_tensor(t.data(), t.layout())` by value,
whose non-owning engine aliases the actual operand.

The scope scan found retained destination views in serial/fused/C32/column-
split recurrence, full-transfer stage1, reset-B preparation and both scans.
Immediate `copy(..., retile_D(tmp))` calls do not make that owning copy; this
explains why a coordinate-only/inverse-chain test did not catch the failure.
Both APIs are fixed at the common seam. All original kernel bodies, shapes,
load/MMA/barrier cadence, reset policy and actlize pin are unchanged.

Before the fix the expanded L006 gate found `3072/3072` detached destination
cells and `3072/3072` stale source cells. After the fix both are `0/3072`;
the exact old implementation remains an expected-red control. The local
zero-result model yields precisely the reported error pair on fixture hash
`d25bbac598263f63`. This is corroboration, **not an elementwise device replay**:
the old log did not give nonzero counts or actual values. The device test now
prints those on failure, plus input/reference hashes before launch.

A fresh hgcc compile/link passed after this repair. Adapter headers now also
invalidate all six device objects; actlize's custom build rule did not infer
those header dependencies. `make -n -f CMakeFiles/gdn_qsa_ppu.dir/build.make
-W <absolute-shared_copy.cuh> CMakeFiles/gdn_qsa_ppu.dir/build` schedules all six
compiles in the tested Makefiles build. No old object is used as fix evidence.
On 2026-09-20 the user reported **PASS** for the `PERF=0` rerun requested
after handoff `fbdbf61` (repair `f003fad`). This is user-reported device
admission; the complete new log, binary hash and error measurements have not
yet been supplied. Corrected performance remains pending. The next run uses
`PERF=1` and reports both decay domains separately; no timing from the failing
arm is admissible.

## Build and device handoff

On the PPU box, from this repository's `ppu-backend` branch:

The logical actlize target remains `ppu0010`. HGGC releases disagree on its
CLI spelling: some accept `-arch=ppu_10`, others only `-arch=ppu001`. GDN's
CMake compiles a tiny kernel to select the accepted spelling before compiling
the real sources. It retries the legacy spelling **only** after the specific
unsupported-architecture diagnostic; unrelated SDK/header/codegen failures
remain failures. It never selects `all`, PPU1.5, or a different collective.
The final `GDN HGGC arch:` configure line and `gdn_hgcc_arch.txt` record the
selection. That compiler-hash/architecture contract is a dependency of all
six device objects, so reconfiguring for a different SDK cannot reuse an old
architecture selection. No actlize submodule edits/pin changes are needed.

```bash
git pull --ff-only
git submodule update --init --recursive
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 \
  bash tools/run_ppu_gdn_backend_box.sh
```

All artifacts go under `/workspace/gdn-qsa-ppu-<sha>-<UTC>`. The runner builds
against the box's installed PyTorch (no hard-coded torch 2.8 binary), preserves
source SHA/diff and binary hashes, then tests reset, Blelloch, Hillis–Steele,
serial, tails, GVA and S=2048. Any correctness/route failure stops timing.
The default final timing is B1/S2048/Hk16/Hv32/D128 at strong and weak decay.
Set `PERF=0` for correctness only.

### Same-input FLA comparison

With a working PPU-compatible FLA/Triton installation in the box's Python:

```bash
git pull --ff-only
WITH_FLA=1 PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 \
  bash tools/run_ppu_gdn_backend_box.sh
```

This opt-in mode uses `benchmarks/bench_ppu_gdn_fla.py`. It checks and times
the two target cases B1/S2048/Hk16/Hv32/D128 with g=-1 and g=-0.1, using the
**same CPU-generated BF16 tensors and recurrent oracle** as device admission.
Both calls start with zero state, return output **and final state**, use
scale=1/sqrt(128), and disable QK normalization. Both output/state pairs must
satisfy the existing 2% criterion, and each arm must repeat bit-for-bit. The
state dtypes are reported rather than silently cast for the comparison.
The optimized reset approximation remains in our path; passing these two
fixtures does not establish its accuracy on arbitrary model activations.

FLA is the installed public `chunk_gated_delta_rule` Triton route; backend
auto-dispatch to other libraries is disabled before import. Its version,
entry-source path/hash and Triton version are recorded. There is **no silent
ours-only fallback** if FLA cannot import or execute. The wrapper supports
both older explicit `head_first=False` APIs and newer APIs that removed that
keyword. `FLA_ROOT=/path/to/flash-linear-attention` is optional; by default the
installed package is used, with no clone or installation performed by the
runner. FLA compatibility code does not patch its math or kernels.

For the CUDA-13 SDK with an older Triton, the benchmark backports the
`13.0 -> PTX 9.0` version-parser entry from
[Triton v3.5.0](https://github.com/triton-lang/triton/blob/v3.5.0/third_party/nvidia/backend/compiler.py).
It applies **only inside this benchmark process**, only when the original
parser raises the exact known unsupported-version error for `13.0`.
Already-working vendor mappings, other CUDA versions, LLVM feature caps,
backend/assembler selection and unrelated errors are untouched. Nothing is
written to site-packages. The compatibility decision and original compiler
source hash are included in the baseline identity/log/JSON; this is not a
silent toolchain change or a claim that all CUDA-13 JIT behavior is verified.

The old parser failure was reproduced with the locally installed Triton.
After the shim, 13.0 returns 90, 12.9 still returns 88, and its LLVM feature
cap remains `+ptx86`. The local PPU SDK 2.1.1 assembler reports release 13.0
and accepts a minimal `.version 9.0` PTX kernel; `hgobjdump` recognizes its
output as PPU 1.0. This is compile-only evidence, **not an FLA device result**.

Default GVA is native: exactly the same device tensor objects reach both
APIs. If an older installed FLA lacks GVA, explicitly set `FLA_HEADS=expanded`;
that prepares Q/K head expansion once **outside timing**, prints the mode and
never hides the extra layout choice. Ours remains native GVA. FLA uses chunk
64 where its API accepts it (otherwise its installed default), ours uses
chunk 16; equal logical inputs do not require equal internal chunking.

First launch/JIT/autotune and five warmups are excluded. Seven samples of ten
launches per arm are taken in alternating ours/FLA order, with synchronization
between arms: the two implementations **never run concurrently**. These are
full-public-API event spans including allocations, launch gaps and our host
dispatch synchronization, not device-kernel-only durations. Memory reporting
is preflight peak allocation minus the immediately preceding live allocation.
Every timed result is checked against the oracle and its own preflight hash.

The runner saves `fla-comparison.log` and `fla-comparison.json`, including raw
samples, errors, fingerprints and `speedup = FLA median / ours median`, beside
its ordinary source SHA/diff and binary hashes. Overlapping observed sample
envelopes are marked `UNRESOLVED`, not a claimed winner. This is an empirical
spread rule, not a statistical confidence interval.

Local CPU contract tests cover native input identity, one-time head expansion,
old/new FLA argument conventions, missing final state, wrong output/state,
nonfinite output, missing FLA and both performance verdict directions. PPU FLA execution
and A/B speed remain device-only; no local benchmark result is claimed.

To use the built original host wrapper directly:

```python
# Set GDN_QSA_PPU_EXTENSION to the build's _gdn_chunk_ppu*.so before first use.
from actlize_la import gdn_chunk, gdn_chunk_twolevel
out, state = gdn_chunk(q, k, v, g, beta)
out, state, route_info = gdn_chunk_twolevel(q, k, v, g, beta, group_chunks=8)
```

The PPU extension is explicit opt-in. Without the environment variable, the
original NVIDIA extension remains the default. An invalid PPU path fails;
there is no silent fallback. All public inputs must be BF16 on one device.
