# PPU1.0 native BF16 default

The product dispatch now selects the validated native scalar BF16 conversion
for these processed-input geometries:

| B | T | Hk | Hv | K, V | g dtype | solve threads |
|---|---|---|---|---|---|---|
| 1 | 2048 | 16 | 32 | 128 | FP32 | 256 |
| 1 | 2048 | 16 | 48 | 128 | FP32 | 128 |
| 1 | 8192 | 16 | 32 or 48 | 128 | FP32 | 128 |

Q/K/V/beta and output remain BF16; initial/final state remain FP32 in public
`[B,Hv,K,V]` order. The gate is processed natural-log decay, not raw logits.
There is no gate-value inspection or strong/weak decay routing. Every original
rounding boundary, the three-product TF32 inverse, query scale and workspace
layout are preserved. `ppu.cvt.rtte.bf16.f32` replaces the software constructor
sequence; no global compiler-version macro or dependency guard is changed.

The following all reach `actlize_la_ppu10_forward_prepared`:

- `gdn_forward(..., backend="ppu10")` with the product serving library;
- `PreparedGDNCore` on caller-owned processed tensors;
- `PreparedGDN(mode="prefill")`, after its prepare/pack/gather transport.

For serving, the policy uses the prepared request capacity and token width,
not a CPU readback of the live-row count. The original native-BF16 promotion
left helpers, state-slot transposes and short/inactive-row masking unchanged.
The subsequent [transport integration](PPU10_PREFILL_TRANSPORT.md) adds
token-parallel/H1 helpers without changing this core policy; it has a separate
combined product device gate. Decode/verify/commit do not use
this prefill core policy. Other shapes, tails and BF16 gates use the retained
implementation; PPU1.5/1.7 and NVIDIA backends are unchanged.

## Inspection and installation

No delivery flag, environment tuning switch or caller-selected warp geometry
is required. Before capture, inspect:

```python
from actlize_la import get_gdn_route_info
info = get_gdn_route_info(q, k, v, g, beta, backend="ppu10")
print(info["core_route"])  # id=3/name=native-bf16, or id=0/name=retained
# PreparedGDNCore exposes the same value as plan.core_route.
```

Source users must rebuild `libactlize_la_serving.so` and select it through
`ACTLIZE_LA_SERVING_LIBRARY`; installed users need a wheel built from this
source. The existing 0.2.2 wheel does not acquire new kernels from a Python-only
update. An explicit `GDN_QSA_WY_EXTENSION` selects the old diagnostic extension,
not this product library. Clear that override when testing the product default.
Old libraries without the new route query correctly report the retained route.

## Evidence boundary

The eight paired ZW810 measurements at source `687828e` (two gate values per
geometry above) passed output/final-state RAW-BIT checks and were faster than
the old default by 22.38–30.76%. FLA speedup was 1.233–1.308x; six of eight
met the separate 20%-lower-latency goal. These are complete-call ACU kernel
sums, not API latency, serving-helper performance or other-SKU predictions.

Local HGCC2.2.1 integration checks found the four native bodies byte-for-byte
equal to that candidate's same-SDK build, including encoded branches and full
resource/argument records. The 32 retained main bodies are likewise unchanged.
The native bodies use respectively 84/84/120/70 vector registers and zero
stack. This is compile evidence, not a fresh product-entry device result.

### Product-entry device admission

Source `dd13c95a03e39654df3796ecd02a6204ac2926e5` passed the product-entry
regression on 2026-11-10: PPU-ZW810, 72 CUs, HGCC 2.2.1-184ee6,
PyTorch 2.9.0, runtime 13.0. The complete receipt contains:

- Eight native-route cells and five retained-route controls, with no missing
  or duplicate cells. Each passes eight public/prepared output and FP32-state
  RAW-BIT comparisons, plus zero-state and output-only checks.
- 104 prepared graph replays with changed inputs and no replay allocations.
- Four serving geometries, each covering changed state slots, shortened live
  lengths and inactive requests in one graph. State-layout round trips,
  untouched slots, workspace canaries and allocation checks all pass.

The device log and structured results agree exactly. Archive SHA-256:
`e9c995f0baa4026a74a275433b2b8696ad40c009757046f16e9486a901c905c8`.
Both identity and results record binary SHA-256
`4480e661587afa1a09534a34f9d4b30c716bd11f9570499c0faad5053705618e`.
The binary itself remains on the box, so this is a mutually consistent receipt,
not an independent local rehash of that binary. Tracked sources are clean;
only an ACU report and result archive are listed as untracked.

This closes the product dispatch/numerical/graph gate. It does not add a new
performance measurement, validate a full Flare model, change other backends,
or update the already-published 0.2.2 wheel.

Run the product-entry regression on the box:

```bash
JOBS=16 bash tools/run_ppu_native_default_box.sh
```

It builds for PPU1.0 using `PPU_SDK` (default `/usr/local/PPU_SDK`). To execute
an already built library, set `LIB=/absolute/path/libactlize_la_serving.so`;
the script then never rebuilds. The test covers eight native and five fallback
cells, public/prepared RAW-BIT equality, zero/nonzero state, changed-input graph
replay, and four raw-serving geometries against the retained C entry. It emits
a small `results.tar.gz`; no device means SKIP, never PASS. This gate does not
rerun or replace the independent numerical oracle or performance campaign.
