# actlizeLA

Forward linear attention in CUDA C++/CuTe/CUTLASS, with independent execution
backends for NVIDIA SM80, Hopper SM90a, and PPU1.0. Python supplies the tensor
API, not the compute kernels. Triton/FLA are optional comparison baselines.

## Install on H100 / H800 (SM90)

Use an environment with CUDA-enabled PyTorch, a C++17 compiler, and CUDA 12.8
`nvcc` (the compiler used for the recorded H800 results). No PPU SDK or actlize
submodule is needed for this CUDA backend; pinned CUTLASS headers are vendored.

```bash
git clone git@github.com:DrXuQian/actlizeLA.git
cd actlizeLA
CUDA_HOME=/usr/local/cuda CONFIGURATION=value64 bash tools/install_sm90.sh
```

This installs the `actlize_la` frontend and builds one explicitly selected
SM90a binary. It does not launch a GPU kernel. Plain `pip install .` installs
only the frontend; it never silently builds the old SM80 implementation.

```python
from actlize_la import load_sm90

forward = load_sm90("build/sm90-value64")
output, final_state = forward(q, k, v, g, beta,
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

### Named configurations

| Build configuration | Recorded H800 use |
|---|---|
| `value64` | Main low-head baseline, including B1/T2048/Hk16/Hv32 |
| `value64-local-inverse` | Selected T8192 and Hv16 cases |
| `value128-paired` | Selected higher-head / larger-batch cases |
| `control` | Original fused SM90 comparison control, not the optimized default |

Build a different configuration by changing `CONFIGURATION`; each gets a
separate directory. The loader verifies its receipt, binary hash and compiled
configuration. There is no automatic winner selection or fallback to SM80.
Do not extrapolate the H800 winner to every shape, GPU, or PPU1.7.

For a device correctness check after installing, see [SM90 usage](docs/SM90_INSTALL.md).

## PPU and SM80

PPU1.0 uses native actlize AIU and paired shared-memory readers, not emulated
Hopper instructions. It retains independently selectable original, WY and
residual algorithms; installing the frontend does not promote an experimental
path. See [PPU build/API](docs/PPU_BACKEND.md) and
[residual formulation](docs/PPU_GDN_RESIDUAL.md).

```bash
git submodule update --init --recursive third_party/actlize
GDN_QSA_TARGET=python python -m pip install --no-build-isolation --no-deps -e .
# Then build the explicitly selected PPU backend following docs/PPU_BACKEND.md.
```

The optional original CUDA SM80 operators can be compiled explicitly with
`GDN_QSA_TARGET=cuda_sm80 GDN_QSA_BUILD_GDN_ONLY=1 python -m pip install
--no-build-isolation -e .`. PPU1.7 has a separate PPU CUTLASS3.6/10700 source
graph; CUDA source checks are **not** native PPU1.7 execution evidence.

## Layout

- `actlize_la/`: Python contracts and explicit native loading.
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

This is a new actlizeLA source history, migrated from the GDN-QSA-sm80 work.
The old Python package name is not provided. Original code, cuLA derivatives,
and CUTLASS/actlize keep their licenses and attribution; see [NOTICE](NOTICE).
