# Complete serving wheel (0.2.2)

Version 0.2.2 binds ordinary PPU1.0 `gdn_forward` and `gdn_chunk_residual` to the
complete wheel's retained full-chunk/guarded-tail library, without another
`_gdn_wy_ppu.so`. `PreparedGDNCore` exposes the same processed-input launch with
caller-owned output/state/scratch. No device kernel or numerical schedule changed.
Installed qualification adds the actual Flare QSA paged/catalog component,
36 public-default cases and four mutable-input core graphs. Performance remains
deferred by request, not admitted by correctness.

Version 0.2.1's symbolic process-local serving-handle fix is retained.
Flare's Torch 2.9 AOT workers represent a rebound handle as an unbacked SymInt;
it cannot be used as a weak-registry key. Fake dispatch leaves symbolic handles
unresolved, while concrete owners still validate tensor metadata and real
dispatch still requires a live owner. Native kernels and the C ABI are unchanged.
Rebuild/reseal the complete wheel; editing installed binding files invalidates
the package manifest. Device and complete-model qualification remain separate.

Distribution `actlize-la`, import `actlize_la`. A complete serving wheel includes
GDN/transactions/prefill, QSA, ShortConv/history, normalization, HC, PLE and MTP
pointwise helpers. The PPU1.0 allocating default and prepared API share one native
implementation; SM90, PPU1.5 and explicit legacy experiment routes are unchanged.

## Install and prepare

Use the **matching PPU target and SDK artifact channel**. Linux wheel tags do not
encode a PPU generation. The filename's build tag and the sealed manifest identify
the target/SDK; never publish ppu10 and ppu15 wheels in one unfiltered index.
The local PPU1.0 build uses **SDK 2.2.1-184ee6 (runtime v3)**. ppu15 and
cuda_sm120 are separate explicit build targets, not aliases or fallback binaries.

With the platform's Torch/runtime already installed:

```bash
python -m pip install --no-deps /path/to/actlize_la-0.2.2-*-py3-none-linux_x86_64.whl
```

The 0.2.2 PPU1.0 wheel is distributed through the repository's GitHub Release:

```bash
python -m pip install --no-deps \
  https://github.com/DrXuQian/actlizeLA/releases/download/v0.2.2/actlize_la-0.2.2-1ppu10version221184ee6-py3-none-linux_x86_64.whl
```

The wheel declares no dependency that would download/replace platform Torch,
CUDA, Triton or BLAS. It does not compile or download code at installation.
Platform Torch must support `torch.library.custom_op` and fake/export tracing;
the intended box stack is Python 3.12 / platform Torch 2.9.0. SDK 2.2.1's own
runtime requires a compatible OS C/C++ runtime (notably GLIBC 2.38 and
GLIBCXX 3.4.32 in its wrapper); use the matching platform container. The wheel
does **not** bundle those system libraries. Actual compiler, SDK library hashes,
symbol-version requirements and native ELF dependencies are in the manifest.

```python
import actlize_la

# All three lines are safe before choosing a device: file reads only.
info = actlize_la.get_library_info(target="ppu10")  # JSON-serializable dict
path = actlize_la.get_library_path(target="ppu10")  # absolute pathlib.Path
default_path = actlize_la.get_library_path()        # explicit wheel default only

# Explicit preparation: imports Torch/registers custom op, then loads the DSO.
from actlize_la.serving import NativeLibrary
library = NativeLibrary(target="ppu10")
debug_library = NativeLibrary("/explicit/libactlize_la_serving.so", target="ppu10")
```

For the allocating API, processed Q/K/V/beta are BF16, natural-log `g` is
BF16/FP32, and initial/final state is FP32 `[B,Hv,K,V]`. This is not the raw-logit
and VK-slot contract of `PreparedGDN`. Query scaling remains inside the kernel,
exactly once. `PreparedGDNCore` has the processed contract and does not allocate
or configure during replay. Prepare/warm up before capture, and keep its owner
alive with the graph. `gdn_core_workspace_specs(B,T,Hk,Hv)` returns only inverse,
snapshots, vnew and gates; it preserves PPU1.0's padded inverse pitch.

Source developers may explicitly set `ACTLIZE_LA_SERVING_LIBRARY` to their built
PPU1.0 serving DSO and use the same public default; no such variable is needed
for the complete wheel. `GDN_QSA_WY_EXTENSION` still selects a legacy diagnostic
extension. Setting both is rejected, and a missing/corrupt selected library
never falls back to another implementation.

Public signatures are `get_library_info(target=None) -> dict` and
`get_library_path(target=None) -> pathlib.Path`. They do not import Torch, dlopen,
run a subprocess, initialize/query a device, or search another source/system path.
Missing native resources, unsupported/ambiguous target, changed bindings or
metadata fail before loading. `NativeLibrary` checks every public export, ABI
version, numeric target and all eleven Python/C structure sizes on the real DSO.
The device-generation check remains an explicit preparation/admission step, not
part of discovery or captured execution.

Retain prepared owners and the library until all dependent graph work drains.
Discovery/loading happens once during preparation, never from graph replay.
FP32 `[slot,Hv,V,K]`, BF16 boundaries, enum errors, consumed-prefix counts,
zero-prefix raw immutability, inactive rows and single-use transactions are
unchanged. Workspace bounds and examples remain in
[Flare serving](FLARE_SERVING.md) and [auxiliary helpers](FLARE_FLASHNEXT_AUX.md).

## What the manifest covers

`actlize_la/_native/manifest.json` binds every packaged Python/resource byte,
the DSO, native graph checker and C ABI headers. `wheel.json` seals the manifest's
SHA-256. `manifest_sha256` therefore changes for a private resource **or Python
binding** change, not only for a top-level DSO change. The inventory also rejects
extra shadow files. Installation-generated `__pycache__/*.pyc` is not payload.
This detects artifact corruption/mixing; authenticity still belongs to the
artifact store's signature/trust policy.

The native payload has `$ORIGIN` RUNPATH only. Platform SDK/driver, libc,
libstdc++, libtorch, CUDA runtime and BLAS are never included. The native graph
checker is a qualification executable, not a production dependency loaded by
Python. No private secondary DSO is needed by this serving target.

## Build (maintainer, not the end user)

Use a clean checkout with the pinned `third_party/actlize` initialized, a matching
SDK/container, CMake, setuptools, wheel and platform Torch already installed:

```bash
python tools/build_serving_wheel.py --target ppu10 \
  --sdk /usr/local/PPU_SDK --jobs 8 \
  --out /workspace/actlize-la-wheel-ppu10-sdk221
```

The output directory must be new. This builds the **existing** complete
`csrc/serving/CMakeLists.txt` target and native graph executable, checks actual
ELF dependencies/relative RUNPATH, dlopens the actual DSO for its ABI/target,
then packages it. No device kernel runs during build. `--target ppu15` also
includes that target's existing native prefill translation units.
`--target cuda_sm120` instead uses nvcc and its own binary/ABI check.
`--allow-dirty` is development-only and is recorded as such; release artifacts
require clean source and the exact pinned dependency.

Outputs: `dist/*.whl`, `result.json` (hashes, revision, target/channel),
`qualification-kit.tar.gz` (only runners/tests/oracles), build logs. Keep binaries
in the artifact store, not ordinary source Git blobs. Publish each target/SDK
channel separately. A host-only source install has no `_native` receipt and its
discovery API reports that a complete precompiled wheel is absent.

## Installed-only box qualification

To build on the matching box and test without replacing its existing Python
installation, use the complete non-performance handoff:

```bash
TARGET=ppu10 PPU_SDK=/usr/local/PPU_SDK JOBS=8 \
  FLARE_ROOT=/path/to/spark-local-inference \
  bash tools/run_serving_release_box.sh
```

It creates a new `/workspace` artifact directory and a `--system-site-packages`
environment, installs only the newly built wheel there with `--no-deps --no-index`,
and runs installed qualification. Platform Torch, the existing actlizeLA install,
and framework sources are not changed. Set `FLARE_ROOT` to the actual checkout;
omitting it leaves component admission NOT_RUN. Upload its small `results.tar.gz`.
No timing, ACU or performance comparison runs in this command.

No box compiler is needed. Install the wheel as above and run the test kit (or
these same test tools from the checkout; production imports still come only from
the installed package). In the existing Flare environment, with QUactlize's
complete matching wheel, gguf and the framework dependencies already installed:

```bash
TARGET=ppu10 FLARE_ROOT=/path/to/spark-local-inference \
  bash tools/run_serving_wheel_box.sh
```

Set `FLARE_ROOT` to the actual `feat/qwen38-gguf` checkout. The runner sets
`FLARE_RUN_CUDA_TESTS=1`; actlizeLA and QUactlize paths are discovered automatically.
It uses the provided platform BLAS
`/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcublas.so` via **cublas** ABI,
never acblas or a copied BLAS. `BLAS_LIBRARY` can explicitly override that test path.
Use `PYTHON` to select the existing platform environment, not to install a new Torch.

The script uses `python -I`, clears PYTHONPATH, copies only tests/oracles to a new
`/workspace` directory, and verifies production-module locations. It runs:

1. Guarded installed discovery, complete native ABI and corruption/wrong-target negatives.
2. Existing host contracts, actual fake tracing/export and mutation declarations.
3. `serving_device.py`, `aux_device.py`, and the installed native graph executable.
4. PPU1.0 installed public default: 36 recurrent-oracle cases plus four zero/nonzero
   state, FP32-gate, full/tail graph cases; eight replays and output/state RAW-BIT
   agreement with the caller-owned core, without legacy extension variables.
5. The real Flare GDN sequence, HC/MTP mixing, PLE and QSA page/catalog test files.

It reports actual pytest pass/skip/fail counts and requires execution in **all
four** Flare files. A zero-test/skip/device-absent run is **NOT_RUN**, never
DEVICE_PASS. `HOST_ONLY=1` intentionally leaves device/Flare admission open and
returns exit 3. Missing Flare also returns exit 3 after operator gates; it is not
complete component admission. Failed tests return nonzero and remain failures.
Upload the small `results.tar.gz` printed at the end; it excludes binaries,
models and corruption-copy directories.

Operator/component graph PASS is not complete target/MTP export or Rust replay
admission. Actual PPU numerical/graph/performance evidence must come from the
new wheel's box run, not the earlier source checkout's user-reported PASS.
