# Retained PPU1.0 forward

`main` contains the retained V32/8 residual implementation: native AIU/SWZL
delivery, static inverse solve, reused gate coefficients and a full-C64 state
specialization. Incomplete chunks use the same static solve with the guarded
gate-cache state implementation. Inputs, BF16 rounding boundaries, FP32 state,
and public output layouts are unchanged.

This does not add the experimental V16/4 shape selector. NVIDIA SM80/SM90
backends and the existing SM90 automatic shape policy remain independent.
The original, materialized-WY and residual numerical schedules are distinct;
the consolidation does not silently change the default algorithm.

## Build

Use the SDK and PyTorch from the same PPU container. All commands below run
from a `main` checkout; an old extension must be rebuilt for the added symbols.

```bash
git submodule update --init --recursive third_party/actlize
GDN_QSA_TARGET=python python -m pip install --no-build-isolation --no-deps -e .
cmake -S . -B /workspace/actlizeLA-main-build \
  -DGDN_QSA_TARGET=ppu10 \
  -DPPU_SDK_ROOT="${PPU_SDK:-/usr/local/PPU_SDK}" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /workspace/actlizeLA-main-build --target _gdn_wy_ppu -j 8
```

Set `GDN_QSA_WY_EXTENSION` to the generated `_gdn_wy_ppu` Python extension
in that build directory, including its Python ABI suffix. Do not point it at
`libgdn_wy_ppu.so`, which is the device library rather than the Python binding.
Missing entry points fail; the loader does not silently select an older path.

## Unified API

```python
from actlize_la import gdn_forward

output, final_state = gdn_forward(
    q, k, v, g, beta,
    initial_state=initial_state,
    output_final_state=True,
    backend="ppu10",
    algorithm="residual",
    delivery="full-chunk",
)
```

There is no caller-selected warp geometry. The native entry checks sequence
length and selects full-chunk or guarded-tail state; it does not inspect gate
values, reset/truncate the recurrence, repack inputs, or autotune on a call.

- Q/K: `[B,T,Hk,128]`; V: `[B,T,Hv,128]`; BF16.
- Gate/beta: `[B,T,Hv]`; gate BF16 or FP32 in natural-log units, beta BF16.
- `Hv` is a positive multiple of `Hk`; no explicit GVA expansion is needed.
- Initial/final state: `[B,Hv,128,128]`, FP32; initial state may be `None`.
- Output has V's shape. `output_final_state=False` returns `None` for state.
- C64 is internal. Tails are supported. Scale is `1/sqrt(128)`;
  callers perform any required Q/K normalization or gate transformation.

## Validation boundary

The retained runtime files are taken unchanged from `41ade14`. Its device
admission covered 36 cases per delivery, eight repetitions, full/tail chunks,
GVA, nonzero carried state, output-only and input immutability: the unchanged
2% independent oracle and RAW-BIT equality to scalar residual both passed.
The full-C64 state was also retained as the control in later geometry tests.

Prior paired ACU captures at B1/T2048/Hk16/Hv32 measured complete forward
times of 168.924/169.362 us for gates -1/-0.1, versus same-run FLA
225.333/227.875 us. These are single-capture kernel sums, not API latency,
universal speed guarantees or fresh measurements of this main consolidation.
Source/runtime identity and historical receipts are retained in the
[archived evidence](https://github.com/DrXuQian/actlizeLA/blob/archive/ppu10-tuning-closed-20260929/docs/PPU10_FULL_CHUNK_ACU_20260928.md).

Local integration checks cover the public API, missing-symbol rejection,
full/tail selection and the existing host/native-code checks. Device tests
remain explicit; a host build is not device numerical or performance proof.

The 2026-09-29 consolidation passed 104 available Python host tests and 25
C++ ownership/policy tests. The changed gate-cache and full-chunk TUs compile
with local HGCC 2.1.1: 124/120 vector registers, zero stack, unchanged native
matrix/traffic/synchronization contracts; four missing-instruction negatives
are rejected. Full binding rebuild and Torch-dependent checks are unavailable
in that local environment (missing SDK runtime libraries and PyTorch), not
counted as PASS. No device rerun or new speed admission was performed.

To repeat device admission after building in a PPU environment:

```bash
python tests/test_ppu_residual_backend.py \
  --extension "$GDN_QSA_WY_EXTENSION" --deliveries full-chunk
```
