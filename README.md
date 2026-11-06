# actlizeLA

Forward linear attention in CUDA C++/CuTe/CUTLASS, with independent execution
backends for NVIDIA SM80, Hopper SM90a, PPU1.0 and ZW890/PPU1.5. Python supplies the tensor
API, not the compute kernels. Triton/FLA are optional comparison baselines.

## Install on H100 / H800 (SM90)

Use an environment with CUDA-enabled PyTorch, a C++17 compiler, and CUDA 12.8
`nvcc` (the compiler used for the recorded H800 results). No PPU SDK or actlize
submodule is needed for this CUDA backend; pinned CUTLASS headers are vendored.

```bash
git clone git@github.com:DrXuQian/actlizeLA.git
cd actlizeLA
CUDA_HOME=/usr/local/cuda bash tools/install_sm90.sh
```

This installs the `actlize_la` frontend and builds/registers all three retained
SM90a candidates for automatic shape selection. It does not launch a GPU
kernel. Plain `pip install .` installs
only the frontend; it never silently builds the old SM80 implementation.

```python
from actlize_la import gdn_forward

output, final_state = gdn_forward(q, k, v, g, beta,
                                 initial_state=initial_state,
                                 output_final_state=True)
```

Inputs must be contiguous on one CUDA device:

| Tensor | Shape | Type / semantics |
|---|---|---|
| `q`, `k` | `[B, T, Hk, 128]` | BF16, caller-normalized if desired |
| `v` | `[B, T, Hv, 128]` | BF16, `Hv % Hk == 0` |
| `g` | `[B, T, Hv]` | BF16 or FP32, natural-log decay, not raw gate logits |
| `beta` | `[B, T, Hv]` | BF16 |
| `initial_state` | `[B, Hv, 128, 128]` | Optional FP32, K-by-V order |
| `output`, `final_state` | V shape / state shape | BF16 / FP32 |

Chunk size is 64. Tail chunks and grouped-value heads are supported. Output
includes the `1/sqrt(128)` query scale; do not apply it twice. No backward,
implicit Q/K normalization, gate activation, or automatic precision change.

### Automatic selection

The normal SM90 call selects from tensor shape and cached device metadata;
users do not select a configuration. There is no runtime compilation,
autotuning, gate-value readback, or extra GPU kernel. The H800 table has 12
distinct shapes covering the 14 measured workloads, with one selection for
both gate regimes. Unknown shapes or other SM90 devices use `value64` as an
**unmeasured default**, not a promised optimum. PPU does not inherit this table.

Configuration names remain available for diagnostics:

| Build configuration | Recorded H800 use |
|---|---|
| `value64` | Main low-head baseline, including B1/T2048/Hk16/Hv32 |
| `value64-local-inverse` | Selected T8192 and Hv16 cases |
| `value128-paired` | Selected higher-head / larger-batch cases |
| `control` | Original fused SM90 comparison control, not the optimized default |

The loader verifies all three receipts, binary hashes and compiled labels
once. Missing or mismatched candidates fail instead of silently switching to
another implementation. See [dispatch policy](docs/SM90_AUTO_DISPATCH.md) for
the exact shape table and diagnostic overrides.

For a device correctness check after installing, see [SM90 usage](docs/SM90_INSTALL.md).

For simulator input, `mode="perfmodel"` with an explicit backend and
`sm_count=20` uses configured metadata instead of querying device properties.
This is opt-in (PPU alone does not enable it); see [perfmodel usage](docs/PERFMODEL.md).

## PPU and SM80

ZW890 has a separate native PPU1.5 backend using the pinned actlize dependency.
Build and run its admission suite inside the **matching PPU SDK/PyTorch container**:

```bash
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=8 bash tools/run_ppu15_gdn_box.sh
```

Then set `GDN_QSA_PPU15_EXTENSION` to the printed `_gdn_residual_ppu15*.so` and
call `gdn_forward(..., backend="ppu15")`. There is no PPU1.0 binary fallback.
Native bodies and host layouts are locally checked; **ZW890 numerical and
performance admission is pending device execution**. The runner tests native
primitive maps first, then 40 forward cases, then complete-forward timing.
See [ZW890 support and scope](docs/PPU15_ZW890.md).

PPU1.0 uses native actlize AIU and paired shared-memory readers, not emulated
Hopper instructions. The retained V32/8 tuning is on `main`: static solve,
gate reuse and full-C64 state with guarded-tail fallback. Use the unified
entry with `backend="ppu10"`; the default is the retained residual/full-chunk
implementation, including its guarded-tail branch. No caller-selected delivery
or warp geometry is needed. Original, scalar residual and materialized-WY
remain explicit alternatives. Default final state is FP32, not the old
original algorithm's BF16. Default-route numerical PASS is user-reported; the
historical speed measurements belong to the explicit retained kernel. See
[PPU build/API](docs/PPU_RETAINED_PATH.md),
[original implementation](docs/PPU_BACKEND.md) and
[residual formulation](docs/PPU_GDN_RESIDUAL.md).

```bash
git submodule update --init --recursive third_party/actlize
GDN_QSA_TARGET=python python -m pip install --no-build-isolation --no-deps -e .
# Then build the PPU backend following docs/PPU_RETAINED_PATH.md.
```

The optional original CUDA SM80 operators can be compiled explicitly with
`GDN_QSA_TARGET=cuda_sm80 GDN_QSA_BUILD_GDN_ONLY=1 python -m pip install
--no-build-isolation -e .`. PPU1.7 has a separate PPU CUTLASS3.6/10700 source
graph; CUDA source checks are **not** native PPU1.7 execution evidence.

## Prepared serving interfaces

The complete **0.2.2 serving wheel** packages its AOT library and Python bindings.
Use `actlize_la.get_library_info(target="ppu10")` for file-only discovery and
`NativeLibrary(target="ppu10")` during preparation; no manual `.so` path is needed.
Install from the matching target/SDK artifact channel, with platform Torch already
provided. See [serving wheel installation and qualification](docs/SERVING_WHEEL.md).
Plain source/host-only installs remain distinct and do not claim native support.
The PPU1.0 public default now uses this same packaged retained library; a separate
legacy `_gdn_wy_ppu.so` is not required. For caller-owned processed QKV/log-g/beta,
use `PreparedGDNCore`; raw logits and VK slot pools use `PreparedGDN` instead.

`actlize_la.serving` adds caller-owned GDN decode/verify/prefix commit, native
PPU1.0/PPU1.5 residual prefill with explicit state-layout conversion, and paged incremental
BF16 QSA. Flare preparation bindings, workspace bounds and a box test script
are documented in [Flare serving](docs/FLARE_SERVING.md). These operators require
their own device admission; model routing is not changed by installing the library.
PPU1.0 prefill uses the retained full-chunk/guarded-tail implementation, with
configuration outside capture and caller-owned padded workspace. The new serving
adapter's device gate is separate from the public-default forward's numerical PASS.
The [PPU1.0 prefill ACU comparison](docs/PPU10_PREFILL_PERF.md) reuses a passed
serving library and separates full serving transport from matched retained/FLA
cores; it does not rebuild kernels or infer a speedup from correctness.

The serving branch also supplies [FlashNext auxiliary helpers](docs/FLARE_FLASHNEXT_AUX.md):
rounded Grouped/Gated RMSNorm, HC combine/norm, speculative dilation1/3 ShortConv,
packed PLE hash/gate and single-use GDN records. C/C++ launch ABIs and a
Torch-free native graph example accompany the caller-owned Python bindings.

## Layout

- `actlize_la/`: Python contracts, shape policy and verified native loading.
- `csrc/backends/`: target-specific CUDA/PPU implementation families.
- `csrc/gdn_chunk/`, `include/gdn_qsa/`: original and PPU GDN kernels/layouts.
- `third_party/`: pinned dependencies and their original licenses.
- `tests/`, `dev/`: numerical, ownership, native-code and negative checks;
  not imported by the installed frontend.
- `tools/`, `benchmarks/`: builds and identity-bound measurements.

The numerical schedules of original, WY/residual and SM90 are deliberately
distinct; see [backend boundaries](docs/BACKEND_ARCHITECTURE.md). Performance
comparisons use all kernels in a complete forward, not selected stages or API
dispatch latency. Retained H800 evidence is in
[the closure report](docs/SM90_H800_CLOSURE_20260927.md); migration is not a new
device performance measurement.

## Provenance

`main` is the product line; the serving integration is being qualified on
`feat/flare-gdn-qsa-mtp`. Unselected tuning arms and their
evidence remain available through [archive tags](docs/BRANCHES.md).

This is a new actlizeLA source history, migrated from the GDN-QSA-sm80 work.
The old Python package name is not provided. Original code, cuLA derivatives,
and CUTLASS/actlize keep their licenses and attribution; see [NOTICE](NOTICE).
