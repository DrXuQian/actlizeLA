# Flare serving operators

The `actlize_la.serving` API adds prepared, caller-owned operations without
changing `gdn_forward`, the allocating QSA entry points, or Flare's sampling
and graph families. It is **not an automatic replacement of Flare's operators**.
Bind these calls at model preparation using `actlize_la.serving.flare`, retain
the prepared objects for the lifetime of captured graphs, and route error
buffers through the framework's existing device-side failure handling.

Source contracts checked against Flare `feat/qwen38-gguf` at `51a4f2e`:
`components/gdn_recovery.py`, `gated_delta.py`, `recurrent.py`, `speculation.py`,
and Qwen38 `attention.py`, `state.py`, `speculative_state.py`, `speculative_model.py`.
The QSA compression/rounding contract follows b12x's `qsa/reference.py` and
`qsa/_kernels.py`. Existing actlizeLA online softmax, score ordering and native
PPU1.0/PPU1.5 residual prefill are retained. The additive
[FlashNext auxiliary interfaces](FLARE_FLASHNEXT_AUX.md) now provide optional
ShortConv, gated/grouped norm, HC/PLE helpers and checked record transactions.
They do not automatically replace framework calls. Main Q/K norm/RoPE,
projections, model scheduling and sampling remain in their original owners.

## Target and validation boundary

| Target | Incremental GDN / QSA | Prefill | Device status |
|---|---|---|---|
| `ppu15` (ZW890) | native CUDA-compatible C++ | existing native AIU residual | requires box admission |
| `cuda_sm120` | independent SM120 binary, same ABI | sequential semantic validation | requires NVIDIA device admission |
| `ppu10` | native CUDA-compatible C++ | retained AIU full-chunk/guarded-tail residual | operator suite user-reported PASS at226355a; performance separate |

The SM120 semantic prefill path is not a tensor-core performance claim. PPU1.0
now selects the retained residual implementation by default, not the semantic
recurrence. An old serving binary lacking the prepared PPU1.0 entry is rejected
at preparation rather than silently falling back. Neither an SM90a binary nor
an SM90 performance result is reused. Both PPU prefill implementations' BF16
intermediate rounding differs from sequential
FP32 recurrence; the original 2% relative-L2 gate is retained. Incremental
decode/commit use FP32 live state and compact records, not BF16 partial state.

The PPU1.0 adapter configures both full/tail state kernels and the static solve
and output kernels during preparation. Replay launches the same retained device
bodies, with external scratch and the caller's stream; no allocation, configure,
device-profile query, synchronization or CPU readback occurs in that adapter.
PPU1.0 prefill transport now distributes independent tokens across CTAs and,
for six registered capacity geometries, fuses prepare and packing (H1).
The core selector and incremental modes are unchanged. See
[transport scope and product gate](PPU10_PREFILL_TRANSPORT.md); historical
core/serving receipts do not replace this combined path's device gate.
PPU1.0's public-default 36-case NUMERIC/RAW-BIT PASS is user-reported; it is not
admission of this new serving pack/slot/graph path or a new speed measurement.
The subsequent complete serving/auxiliary/native-graph suite at226355a was also
reported PASS by the user, after correcting a PPU15-only test predicate. Its raw
bundle has not yet been independently inspected. The release retains this as
source-suite evidence, not installed-wheel or performance admission.

Local compilation, host algebra, device execution, and full-model Flare
capture/export are distinct gates. The box script tests **operator** graph
capture/replay; it does not claim model-package export, MTP acceptance policy,
or all 48 layers have run on PPU. No production routing changes automatically.

The PPU1.0 serving library and native graph checker compile and link with
SDK2.2.1. The installed 0.2.2 wheel has independent host/ABI/relocation checks.
PPU1.5 and SM120 compile checks remain separate from device execution; their
device status is not inherited from PPU1.0. See [release notes](releases/v0.2.2.md)
for the current validation boundary.
Fake tracing and export functionalization preserve the custom op's declared
mutations; this is metadata validation, not a fake numerical implementation.

## GDN contract

- Q/K/V and raw `a`/`b`: BF16, convolved `mixed_qkv=[tokens,q|k|v]`.
  Column views of a fused projection may have a larger row pitch.
- `A_log`, `dt_bias`: FP32 `[Hv]` (convert weights once during preparation).
- Normalize Q/K in FP32 with `rsqrt(sum(x*x)+1e-6)`.
- Apply `1/sqrt(128)` to Q exactly once. The native residual prefill owns
  that scale in its output kernel; the incremental path owns it in prepare.
- `g=-exp(A_log)*softplus(a+dt_bias,threshold=20)` is natural-log decay;
  recurrence uses `exp(g)`. `beta=BF16(sigmoid(float(b)))`, widened for math.
- Grouped mapping: `key_head=value_head/(Hv/Hk)`; primary geometry 16:48.
- Serving state is explicitly FP32 **[slot,Hv,V,K]**. Legacy `gdn_forward`
  remains **[B,Hv,K,V]**. Prefill gathers/transposes and scatters explicitly;
  equal 128x128 shapes never authorize byte reinterpretation.
- Slot zero is null. Decode/verify null rows write zero output and no state.
  Fresh prefill reads slot zero, writes a positive final slot. Inactive rows
  do not modify a live state. Empty ranges and output outside live packed
  ranges are not valid output; callers may clear those separately if needed.
- Slots, packed starts and live request counts are device tensors. Invalid
  ranges, slot bounds, conflicting writable state rows, and invalid accepted
  counts produce nonzero **device `errors`** and suppress that request's writes.

Modes: `decode` consumes one token in-place; `verify` leaves state untouched
and records corrections; `prefill` consumes packed intervals through explicit
initial/final slot indices. The GDN record window may be a maximum capacity8
while device starts select current widths1/2/4/8. Intermediate prefill
checkpoint exports are not supported by this binding; prepare rejects an
explicit `checkpoint_export=True`. Nonzero device checkpoint fields passed by
the Flare binding produce `errors=7` without state writes. Final chunk
continuation is supported; intermediate checkpoint exports remain unsupported.

### Verify / commit / ShortConv counts

Records are FP32 `delta[R,Hv,W,V]`, `keys[R,Hk,W,K]`, `decays[R,Hv,W]`,
plus per-request recorded slot and valid length. There is no full H per token.
`PreparedGDNCommit` replays:

```
H = decay[t] * H
H = H + delta[t][:,None] * key[t][None,:]
```

`consumed_prefix` is the number of **verification INPUTS** actually committed.
It is not the number of accepted drafts. Flare's current graph verifies the
previous pending token first, followed by drafts. Its clipped
`commit.emitted_count` also advances `consumed_tokens` by that count; the last
emitted token remains pending and was not one of those inputs. Pass that
already-computed tensor; do not recompute sampling/EOS/budget decisions here.
Records may be consumed only once per verification round, before reuse.

Preserve Flare's speculative ShortConv contract (or use the new compatible
native helper). It stores the first consumed token plus
the speculative tail; the next call selects history at `previous_accepted-1`.
For a positive consumed prefix do not also shift that history now, or the
next call will apply the prefix twice. For a **zero target-consumed prefix**,
history transport snapshots the small convolution row before ShortConv and
restores it after acceptance. Use new `PreparedConvHistoryV1` for raw ShortConv
slots (0 valid, -1 inactive), not legacy `PreparedConvHistory` (0 null).
Flare must retain
the prior `previous_accepted` in that zero-count case (its current
`_commit_hidden` does). This is state transport, not another convolution.
Zero accepted drafts usually means one target-consumed input, not this case.

### Prepared use

```python
from actlize_la.serving import NativeLibrary, PreparedGDN, gdn_workspace_specs

library = NativeLibrary("/path/libactlize_la_serving.so", target="ppu15")
specs = gdn_workspace_specs(requests, 8, mode="verify", target="ppu15")
workspace = {n: torch.empty(shape, dtype=dtype, device="cuda")
             for n, (shape, dtype) in specs.items()}  # PREPARATION ONLY
op = PreparedGDN(library, mixed=mixed_qkv, a=a, b=b, a_log=A_log, dt_bias=bias,
    state=state_vk, starts=starts, live=num_seqs, initial_slots=slots,
    final_slots=slots, output=output, workspace=workspace, width=8, mode="verify")
op()  # warm up before capture; state stays unchanged
# Bind PreparedGDNCommit with the same records, slots and request ordering.
# During capture/replay: op(); existing acceptance(); commit(); conv_rollback().
```

The complete runnable examples/fixtures are in `tests/serving_device.py`.
`plan.transport_route` reports whether its loaded library uses direct packing;
`tests/prefill_transport_device.py` checks that report against actual graphs.
`actlize_la.serving.flare.bind_gdn` maps actual Flare metadata field names;
`bind_gdn_commit` binds one layer's retained records. Sequential layers may
share transient query/beta/native prefill scratch, but not outstanding
delta/key/decay records. Those stay live until acceptance and commit.

## Paged QSA contract

Ratio4 only; query heads24, KV heads2, D256, four D128 index heads; GQA divisibility
is checked. Selected capacity is **2051**, not 2048 or 2052. Complete groups are
eligible only when their fourth token is visible. The incomplete tail starts
at `floor((position+1)/4)*4`; a just-completed block is not appended twice.
Scores are `sum_h relu(dot(index_q[h],compressed_k))/sqrt(128)`, ordered by
descending FP32 score, ties by ascending logical block ID. An exact bounded
512+512 merge retains the top512 without a query x context global score matrix.

Main q/key arrive already normalized and partial-RoPE applied. Index q/raw key
arrive before the indexer's encoder: four raw keys are averaged and rounded
to BF16, Gemma RMSNorm multiplies by `1+weight` and rounds to BF16, then partial
RoPE64 at the group's **first** position. Flare's FP32 cosine/sine tables keep
FP32 rotary products; BF16 tables use the explicit BF16 product/sum boundaries.
Compressed keys are stored BF16. mRoPE sections are (11,11,10), interleaved.

Supported caches: BF16 K/V `[pages,page_size,Hkv,256]`, BF16 compressed keys
`[pages,page_size/4,128]`; page sizes64/128. Indices and physical offsets use
64-bit arithmetic. Main and compressed tables can map differently. The caller
owns allocation/COW and writable page exclusivity; invalid pages and conflicting
writable boundary pages fail on device. **Flare currently allocates FP8 KV by
default: switch its cache allocation to BF16 explicitly before binding.**
The prepared API rejects FP8 buffers rather than reinterpreting them.

The raw ring has at least12 entries and divides page size (normally16 for
width8). Its logical tags are authoritative. Newly completed groups read all
accepted prior keys before ring writes. A subsequent interval starts at
`previous_anchor + accepted_previous`; rejected future K/V and representatives
remain physically present but are hidden by the causal position/length.
On prefill, anchor is the last valid input position and the next accepted count
is1; on verification it is the interval's first position. Zero consumption
retries from that first position with count0. No history-wide recompression.

`bind_qsa` owns append+compression+selection+attention, so do not also run the
old paged writer. Preserve the existing output gate/projection. Its index-only
RMSNorm/RoPE is the necessary state-processing portion of the existing indexer,
not a new general auxiliary-operator library. The portable online-softmax
path is a correctness implementation, not a claimed TC performance replacement.
The shared-Q8 MTP layer with compression_ratio0 is rejected; ordinary attention
belongs to FlashAttention.

## Memory and alias contract

All execution operands are preallocated tensors. No execution allocation,
JIT, synchronize or CPU readback occurs. Kernel attributes and library loading
are prepared once. Launches use the caller's current stream. The custom op
declares every output, scratch and mutable state tensor as mutated; fake
registration performs no device work. Read/read views may alias. Mutable
regions must be disjoint, except a state read/write is passed once as a mutable
operand. Disjoint strided regions within one pool row are supported.

Workspace upper bounds are returned by `gdn_workspace_specs` /
`qsa_workspace_specs`, including dtypes. For R requests, record width W, Hk16/Hv48:

- GDN persistent H: **3 MiB per slot per layer**, 108 MiB for36 layers.
- Retained verify records: **32,960 R W bytes per layer**, plus12R bytes of
  slot/length stamps. At W8, all36 layers retain about9.05 MiB per request.
- Transient GDN queries+beta: **8,288 R W bytes** plus error flags.
- Prefill: let `J=R*Hv*ceil(W/64)`. Native inverse/snapshots/vnew/gates total
  `82,176 J` bytes for PPU1.0 or `57,600 J` for PPU1.5. PPU1.0 inverse planes
  have a **128x128 padded group pitch**, despite writing only64x64 logical values;
  never substitute PPU1.5's compact inverse buffer. Add packed q/k/v/beta/log_g/output and two FP32 KV states
  exactly as the returned specs declare. These can be shared between layers.
- QSA K/V+compressed keys: **2,112 bytes per cached token per layer** for
  Hkv2/D256; 24.75 KiB/token across12 layers. BF16 main KV is twice Flare's
  default FP8 main-KV storage. Add page rounding, tables and raw ring separately.
- QSA returned workspace: **26,640 bytes per query +4R bytes** including
  BF16 output. It does not scale with context length.
- Optional convolution rollback snapshot: `2 R channels (3+drafts)` bytes,
  plus stamps/error flags, much smaller than per-token recurrent H snapshots.

## Run on box

```bash
git submodule update --init third_party/actlize
TARGET=ppu10 PPU_SDK=/usr/local/PPU_SDK JOBS=8 \
  bash tools/run_flare_serving_box.sh
```

Use `TARGET=ppu10` for ZW810, `TARGET=ppu15` for ZW890. The runner checks the
measured device generation before building. For NVIDIA validation use
`TARGET=cuda_sm120 CUDACXX=/path/to/nvcc` instead.
No PPU probe utility or profiler is needed. This builds against the box's
matching SDK (local PPU runtime libraries may be unavailable), runs host tests,
then actual device/graph tests. Upload the printed `results.tar.gz`; it contains
logs/results, not the potentially large build directory. A missing device is
SKIP / NOT_ADMITTED, never PASS. Default artifacts are below `/workspace`.

Required device gates include non-symmetric state/3:1 head mapping, prefix0..W,
widths1/2/4/8, prefill63/64/65/127/128/129/2048, continuation into decode,
invalid slot/count/history/page negatives, no-state-change during verify,
raw-bit replay stability, causal compression tails, rejection replacement,
and changing device metadata without recapture or memory growth.
