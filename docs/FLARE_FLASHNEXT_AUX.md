# FlashNext non-GEMM serving helpers

Additive interfaces on `feat/flare-gdn-qsa-mtp`, checked against Flare
`c37b36b199f858399fa5cbeceba15161f84564ed` and b12x
`560fef16648e4953bf39b701a226c322a0ab6c7c`. No b12x, cuTile or Triton runtime
dependency. Legacy `gdn_forward`, serving ABI v1 and QSA calls are unchanged.
These bindings are **not automatically installed into Flare's model graph**.

## Operators and arithmetic

`PreparedPointwise(library, operation, *, x, output, errors, ...)` accepts
BF16 logical `[T,G,D]` or an explicitly grouped `[T,G*D]` view. All outputs
are caller-owned, disjoint from inputs and other outputs. Regular positive
element strides, including column views in a common pool, are supported.

| `operation` | Inputs / explicit outputs | Arithmetic boundary |
|---|---|---|
| `grouped_norm` | weight BF16/FP32 `[G,D]`; output BF16 | FP32 logical-width RMS, FP32 gain, final BF16 |
| `gated_norm` | BF16 gate; weight BF16/FP32 `[D]` or `[G,D]` | optional normalized BF16, extra BF16-weight product round, then FP32 gate activation |
| `hc_scaled_silu` | `[T,1,320/160/80]` | BF16(x/4), then BF16(SiLU) |
| `hc_gate_mean` | x/gate `[T,4,2560]`; output `[T,1,2560]` | BF16 sigmoid, BF16 product, FP32 sum/4, BF16 output |
| `hc_combine` | residual=x, value `[T,1,D]`, logits=gate `[T,4]` | BF16(residual + 2 sigmoid(logit/4) value) |
| `hc_combine_norm` | same plus weight; **combined and output** | norm reads the rounded combined BF16, never the unrounded residual |
| `ple_gate` | key=x, query=gate `[T,4,2560]`, value `[T,1,2560]` | BF16 product/sum/divide/clamp/sqrt/sign/sigmoid/final multiply |
| `ple_contribution` | x=gated, value=convolved, both `[T,4,2560]` | BF16(gated+convolved) |

Norm epsilon defaults to `1e-6`; reduction denominator is the logical D,
not its power-of-two padding. Main grouped width is four groups of 2560.
Whole-10240 MTP feedback norm is **not** that operation: use platform norm
or explicitly `groups=1`. HC's prepared FP32 gain uses `zero_centered=False`;
PLE's FP32 delta weight uses `True`. Gated norm rejects zero-centering;
`activation='sigmoid'/'silu'`, both `norm_before_gate` and `round_normalized`
modes implement Flare's oracle and are covered separately.

`ServingRows(T,num_tokens=...)` masks flat rows. Packed mode additionally
binds starts `[R+1]` INT32, slots `[R]` INT64 and num_seqs `[1]` INT32.
All counts are device-resident. **Raw convolution slot0 is valid, -1 inactive**;
this is deliberately different from recurrent-state slot0-null. Inactive
pointwise outputs/traces are zero. Errors are per-token INT32 `[T]`; an optional
per-request `upstream_errors[R]` propagates without overwriting its source.

Optional FP32 `trace` exposes actual intermediates: gated norm `[T,G,2D]`
(normalized, weighted), HC mean `[T,4,2D]` (gate, product), scaled-SiLU
`[T,1,D]`, PLE gate `[T,4,6]` (sum, divided, clamped, sqrt, signed, sigmoid).
Traces are diagnostic outputs; omit them in production. No external scratch
is needed. Each pointwise call launches one kernel (32 B static shared/CTA).

## ShortConv and acceptance

`PreparedShortConv` takes BF16 x/output `[T,C]`, weight `[C,4]`, state
`[slots,C,3*dilation+drafts]`, packed rows, BOOL/U8 fresh `[R]`, INT32
previous_accepted `[R]`, errors `[R]`. All state strides are explicit;
position-major physical pools are supported. Dilation is 1 (GDN, H3) or 3
(PLE, H9), drafts 0..7. Decode is `mode='verify'` with one input;
verification accepts up to drafts+1 inputs. `mode='prefill'` consumes all
inputs and clears the speculative tail. Taps are oldest to current, at
`t-9,t-6,t-3,t` for dilation3. FP32 accumulation rounds to BF16 before SiLU.
History contains **inputs**, not convolution outputs. Empty/inactive requests
write no state; inactive output is zero. Conflicting active state slots fail.

Verification encodes the history after input0 followed by later inputs.
The next call reads H inputs starting at `previous_accepted-1`; nonfresh
counts must be 1..drafts+1. Positive commit must **not shift again**.
`PreparedConvHistoryV1` snapshots before convolution and restores only when
`consumed_prefix=0`. Retain the old previous_accepted in that case. Snapshot
uses BF16 `[R,C,H+drafts]`, INT64 slot stamps `[R]` and independent errors.
The snapshot/restore and convolution each launch two kernels, with no hidden
scratch or allocation. Snapshot retention is `2*R*C*(H+drafts)+8*R` bytes.

`consumed_prefix` counts target verification **inputs**, not accepted drafts
or emitted user tokens. The framework computes acceptance/EOS/budget and
passes its count; the library makes no acceptance decisions.

## PLE hash and geometry

`ple_geometry(...)` returns only immutable host data: 16 primes, 16 cumulative
offsets, 3 odd multipliers (280 bytes), table/padded extent and vocabulary.
Prefer checkpoint integers. Reconstruction follows the pinned SplitMix64,
seed1234, dense-layer ordinal and prime order; no embedding table or b12x Plan
is created. Multipliers satisfy `m <= INT64_MAX/vocab`; cumulative and padded
extents are checked. Geometry weights are immutable during serving.

`PreparedPLEHash` consumes INT64 tokens `[T]`, committed history `[R,2]`,
geometry tensors and packed rows, and writes INT64 hashes `[T,16]`, request IDs
`[T]`, errors `[T]`. Missing context is EOS. A previous EOS cuts earlier
history; the current EOS still hashes its preceding context. Integer products,
XOR and nonnegative remainder do not pass through floating point. Inactive
hash/request IDs are -1. Committed history is **read-only**. One kernel,
zero external scratch. Preparation verifies the immutable device geometry's
280 bytes against the host geometry once; execution does no readback.

## GDN/QSA additions and failure dependencies

`PreparedGDNVerifyV2` / `PreparedGDNCommitV2` add caller-owned INT64 `[R]`
`epochs`, `ready_epochs`, `consumed_epochs` (24R bytes). Initialize ready/used
to zero, assign a positive monotonically increasing round before verify,
and use the same epoch at commit. Outstanding records cannot be overwritten;
wrong-round, wrong-slot and duplicate consumption fail, including prefix0.
Temporarily inactive rows retain outstanding records. Resolve/discard them by
the normal zero-prefix commit **before** reassigning their request slots.
Each V2 verify/commit adds one small stamp kernel. Compact delta/key/decay
records and the FP32 `[slot,Hv,V,K]` state layout are unchanged.

`PreparedGDNCheckedV2` adds a read-only upstream error vector to decode/prefill.
V2 verify normally reads ShortConv errors; V2 commit reads verify errors.
Downstream norm can read the same request errors and zero its rows.
Keep each error output separate; values 0=OK, 1=Bounds, 2=Slot, 3=Conflict,
4=Prefix, 5=History, 6=Page, 7=Unsupported are **enums, not bit masks**.
Native launches return -1 for static descriptor rejection or runtime launch
status; device metadata faults are in the explicit device error vectors.
Failing GDN requests do not update state; legacy GDN outputs on failed/empty
ranges remain invalid and must not bypass the downstream live/error mask.

QSA computation is unchanged. `qsa_cache_specs(pages,slots,page_size=64/128,
ring_size=16)` and `qsa_workspace_specs` provide allocation shapes/dtypes
without a b12x object. Initialize raw tags/rope/anchors to -1. BF16 caches,
ratio4, selected capacity2051, causal visibility and compact incremental
compression remain authoritative. Ratio0 MTP attention stays outside QSA.
See [existing bounds and numerical contracts](FLARE_SERVING.md).

## Prepare, execute, native replay

The thin `actlize_la.serving.flare` helpers bind real DeltaMetadata and
PLEMetadata fields. They do not import or monkey-patch Flare. Prepare/load,
check descriptors, set PPU1.5 kernel attributes, and perform genuine warmup
on the target **before** capture. All run calls enqueue on the current stream;
none allocate device storage, tune, JIT, synchronize or read back live metadata.

The custom op lists all writable buffers (including state, records, traces and
errors), registers fake/meta behavior and preserves mutations through export.
Read-only views may overlap; writable regions must be disjoint, not merely
distinct Tensor objects. Disjoint strided subregions in one allocation work.
Each concurrently executing stream/graph needs its own writable storage.
Sequential layers may share transient scratch, never outstanding records.

Native headers: `csrc/serving/aux_v1.h` (C-compatible),
`gdn_transaction_v2.h` (C++/extern-C, legacy structs in `abi.h`), `qsa.h`.
New descriptors carry version, struct size, dtype, positive element strides,
alignment and accessible byte span from data. Rust can mirror `repr(C)` and
check sizes before launch. No Torch dependency in the library. Native callers
must honor legacy base descriptor capacities/alias constraints in addition
to the V2 guard's checked descriptors. Keep the module/context, all weights,
buffers and graph resources alive until the last launch completes.

`native_graph_example.cpp` builds `actlize_la_native_graph`: prepare, warmup,
capture and eight native replays with changing live counts, independent CPU
checks, and **no Python or Torch callback**. `binary.json` records target,
source SHA, binary hash, symbols and ABI sizes. It is not a serialized model
graph or a cross-platform binary. Flare's model exporter/Rust loader still
owns full-graph packaging and resource retention.

### Framework replacement points

| Flare location at the pinned commit | Replacement / retained owner |
|---|---|
| `components/recurrent.py` ShortConv prepare/run | `bind_shortconv`, dilation1; retain metadata and acceptance scheduler |
| `components/actlize_gdn.py::prepare_commit` | `bind_shortconv_history` V1 for raw conv slots; do **not** pass them to legacy slot0-null `PreparedConvHistory` |
| actlize GDN binding | optional `transaction={epochs,ready_epochs,consumed_epochs}` + upstream errors; use the same fields in commit |
| `components/gated_norm.py` BF16 output path | `bind_gated_norm` with explicit model activation/rounding; no unused MXFP8 branch |
| `recipes/qwen38/hyperconnection.py` | pointwise HC primitives; cuBLAS owns down/up/injection GEMMs |
| `recipes/qwen38/ple.py` | `bind_ple_hash`, grouped norms, PLE gate, dilation3 ShortConv, contribution; keep table/I/O/gather and history commit in framework |
| QSA allocation/binding | BF16 cache specs and existing `bind_qsa`; remove duplicate old cache writer |

The pinned Flare adapter currently passes raw slot0-valid convolution IDs to
legacy slot0-null history transport. The new binding fixes that seam only
when the framework adopts it; old ABI semantics are intentionally preserved.

## Validation and box command

Local (2026-11-05): **147 Python tests +29 subtests and27 C++ tests pass**,
covering independent CPU references/negative witnesses, native descriptor
checks and export metadata. SM120 AOT library and native
example compile/link; PPU1.0/1.5 changed bodies compile with real SDK2.1.1.
Local PPU full link lacks `hg_wrapper`; **NVIDIA/PPU numerical execution and
full Flare model export are not yet admitted**. No device result is fabricated.

```bash
git switch feat/flare-gdn-qsa-mtp
git pull --ff-only
git submodule update --init third_party/actlize
TARGET=ppu15 PPU_SDK=/usr/local/PPU_SDK JOBS=8 \
  bash tools/run_flare_serving_box.sh
```

Use `TARGET=ppu10` for ZW810; for NVIDIA use `TARGET=cuda_sm120 CUDACXX=nvcc`.
Install the repository's Python package/dependencies in the matching platform
Torch environment beforehand; do not overwrite its Torch/FlashInfer artifacts.
The runner creates a new `/workspace` directory, rebuilds the correct target,
runs legacy GDN/QSA plus auxiliary/device/graph tests and the Torch-free example.
Upload its small `results.tar.gz`, not the build tree. Missing device is
SKIP/NOT_ADMITTED (exit3), never PASS.

The device suite covers rounding intermediates, tails/strides/canaries,
dilation1/3 and every prefix0..W, immutable hash geometry/EOS/live mapping,
epoch/error negatives, HC/PLE and GDN graph chains with eight replays,
concurrent independent streams, QSA legacy graph gates, and allocation growth.
It reports operator/graph event spans, actual captured node types/kernel counts
and retained storage. These times include launch gaps, are not ACU device-only
measurements, and do **not** prove absence of GPU idle or full-model speed.
