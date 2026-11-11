# PPU1.0 prepared prefill transport

The PPU1.0 `PreparedGDN(mode="prefill")` path now includes token-parallel
prepare/pack/scatter and bounded direct packing (H1). These are native
transport changes around the existing core, not a new GDN algorithm.
Decode, verify, accepted-prefix commit, PPU1.5 and SM120 keep their schedules.

## Default route and contracts

| Prepared capacity | PPU1.0 transport |
|---|---|
| requests=1, width=2048 or 8192, Hk=16, Hv=32/48/64 | H1 prepare directly to packed operands |
| Other valid prefill capacities | Token-parallel prepare, then separate pack |
| PPU1.5 / SM120 | Existing target implementation, no H1 |

One warp still normalizes one head/token with four FP32 values per lane and
the original XOR reduction tree and `1e-6` epsilon. Independent tokens are
distributed across four warps per prepare CTA and a capped grid.z. Pack and
scatter likewise partition their elements. Final-state elements have exactly
one owner; increasing grid.z does not repeat the state write in every block.

H1 retains natural-log decay, the beta BF16 rounding boundary, original Q/K
normalization, and the single query scale owned by the core. It writes final
BF16 q/k/v/beta and FP32 log_g without the intermediate queries/keys/decays/
betas writes. Those four scratch buffers remain allocated for ABI compatibility
but are **unspecified intermediate contents**, not outputs. All other public
layouts and workspace upper bounds are unchanged, including FP32 VK state slots
and explicit KV/VK conversion. Offset workspace views retain their original
alignment contract; H1 has vector and unaligned scalar stores.

The existing request/transaction validation still precedes all work. Live rows,
starts, slots and errors remain device-resident; unsupported checkpoints and
invalid slots suppress state/output writes. Empty and inactive requests cannot
pollute live slots. Shape selection is based on immutable prepared capacity,
not actual device-side lengths or a readback. Execution adds no allocation,
JIT, device query, synchronization or host callback.

After preparing the normal object, inspect `plan.transport_route`:

```python
assert plan.transport_route["entry"] == "ala_prefill_run"
print(plan.transport_route["direct_pack"])
```

The optional C queries `ala_prefill_token_parallel()` and
`ala_prefill_direct_pack_route(requests, width, hk, hv)` run only on the host
and report the implementation in the loaded library. Older libraries without
these queries retain their existing staged path; a Python update alone does
not add the optimized native kernels. There is no public tuning switch or
new core selector. The existing native BF16 core policy is unchanged.

## Evidence and regression

Source provenance: token-parallel `9286a2a`, direct-pack `e291857`. Their
experimental build switches are not part of the product interface. Local
same-SDK compilation compares the promoted native instruction bytes and full
resource/argument records to those reference helper bodies, and the unaffected
core/decode/commit/QSA bodies to main. A changed encoded instruction is an
expected-red control. This is compilation evidence, not device execution.

Integration checks: 319 host tests and 36 subtests pass, with native ABI checks
against the locally compiled SM120 library (no GPU execution). PPU1.0 SDK2.2.1
compile/link passes; all 38 product device bodies are accounted for by native
byte/resource comparisons. All nine bodies in the affected PPU1.5 translation
units retain their mainline encodings/resources. The no-device runner returns
3/SKIP and writes that status, not PASS.

The reference branch records twelve paired ZW610 H1 cells: helpers 39.28–44.69%
lower latency, complete serving 17.14–19.88% lower, against the token-parallel
staged baseline. It also records six selected and three fallback graph cases,
mutable metadata checks, and full serving/auxiliary gates. Earlier token-parallel
ZW810 evidence is separate. These historical records do not constitute a new
measurement of H1 combined with the current native BF16 core, a ZW890 claim,
or full-model speedup. No new performance result is claimed by this merge.

Run the complete product gate on PPU1.0 in the matching SDK/Torch environment:

```bash
git pull --ff-only origin main
git submodule update --init third_party/actlize
TARGET=ppu10 JOBS=16 bash tools/run_flare_serving_box.sh
```

This includes the unchanged independent serving/auxiliary numerical gates and
native graph consumer, plus `tests/prefill_transport_device.py`. The new gate
checks all six H1 capacities and three fallbacks. Its staged control is the
same library with an additional inactive request, identical first-request
operands and state. It checks every meaningful packed input, output and state
as raw bits; capacity/core-route differences make this a correctness control,
**not** a timing comparison. Independent recurrence remains in the full suite.

Eight replays change slots, length, live count, empty/null requests and invalid
checkpoint/slot metadata. Captured graphs must contain eight kernels for H1
or nine for staged transport, no allocation or host nodes; replay allocations
must remain stable. Workspace canaries and unused H1 scratch remain unchanged.
Removing a field or changing one packed-input bit is red. Host tests separately
exhaust the actual production schedule and route predicate and plant missing
owners, duplicated owners, widened routes and a missing admitted geometry.

`transport-results.json` and its log join the existing small `results.tar.gz`.
No local device is available during integration: the new combined product
device gate remains **NOT_RUN** until this command succeeds. Missing hardware
is SKIP/NOT_ADMITTED, never PASS. Existing released wheels are unchanged.
