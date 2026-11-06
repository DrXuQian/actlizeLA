# PPU1.0 prepared prefill: same-input ACU comparison

This measurement reuses the exact `.so` from a successful serving/auxiliary box
run. It does not compile kernels or change default routing. The numerical PASS
received for226355a does not establish a performance win or main/wheel release.

## Boundaries

| Role | Input → output contract | Included work |
|---|---|---|
| `serving` | raw BF16 qkv/a/b + FP32 VK slot → output + final VK slot | validation, QK norm, gate/beta preparation, pack, state gather/transpose, residual, scatter/transpose |
| `retained` | processed BF16 QKV/beta + FP32 log-g/KV state → output + KV state | exact prepared retained residual, all four kernels |
| `fla` | exactly the same processed operands and FP32 KV state | FLA Triton forward, including its fills and final state |

Only `retained` versus `fla` receives a speed verdict. Comparing full serving to
bare FLA would conflate different work. An end-to-end Flare FLA adapter is not
implemented here by inventing a slow chain of PyTorch helper operations.
Serving helper cost is the sum of its own captured helper kernels, not a
subtraction of unrelated call medians.

Default cells are B1,T2048/8192,Hk16/Hv32,K=V128. They test the stated token
length interpretation of the Qwen3.5 case, not a captured model trace. Hv48
(FlashNext) and natural-log decay near-1 are optional. The default decay is
near-0.1; the BF16 raw logit's resulting actual FP32 decay range is recorded.
The initial state is deliberately nonzero/asymmetric. Thus T2048 is a historical
geometry control, not a claim to reproduce the old zero-state fixture/time.

The untimed admission executes real serving prepare once and saves its exact
processed operands. Both core arms consume those same bytes. Scale is applied
once; FLA's internal QK normalization, sigmoid/gate activation and V-first state
interpretation are disabled. GVA stays native; no silent head expansion.

## Gates and measurement

- Serving: unchanged2% relative-L2 versus raw-input sequential reference.
- Core: unchanged2% max/max versus processed-input sequential reference.
- Serving/retained canonical output and FP32 state: RAW-BIT equal.
- Each role repeats8 times; inputs and initial committed slots remain unchanged.
- Each capture process makes exactly one subject call, with CPU-only checking.
  There is no PPUProfiler, CUDA profiler start/stop, or warmup under ACU.
- Default3 captures per role, rotating order so each role occupies each position.
  Captures and roles are sequential, never concurrent on the same device.
- Primary metric is the sum of `ppu__time_duration.sum` from ACU raw CSV
  (native base nanoseconds converted to microseconds). API event spans are not
  substituted. This remains a profiler-replay measurement, not normal latency.
- All captured kernels remain in the report. Known native inventory must appear
  exactly once; the five FLA math phases must appear once, with all FLA fills
  included. Extra/unknown work or repeated autotuning yields UNADJUDICATED,
  never a filtered fastest-subset speedup. Unknown FLA versions need inventory
  review, not automatic admission.
- Compare full observed sample envelopes. Overlap means UNRESOLVED. Both
  RETAINED_CORE_WINS and FLA_CORE_WINS are valid outcomes, not script failures.

## Run on an otherwise idle PPU1.0

```bash
bash tools/run_ppu_serving_prefill_acu_box.sh
```

The script selects only if exactly one completed `/workspace/actlize-la-flare-ppu10-*`
run has matching library hashes, the full retained-pair denominator and the
corrected checked/unchecked gate. If multiple valid runs exist, it lists them:

```bash
SERVING_RUN=/workspace/<the-passed-serving-run> \
  bash tools/run_ppu_serving_prefill_acu_box.sh
```

Optional wider geometry/decay coverage, still no tactic sweep:

```bash
bash tools/run_ppu_serving_prefill_acu_box.sh \
  --value-heads 32 48 --gates -0.1 -1.0
```

Uses `/sim/eec/shared/junfu.qx/asight/bin/acu`, the existing working FLA/Triton
installation, `DEVICE` (default0), and the matching `PPU_SDK`. Set `FLA_ROOT`
only if using a checkout instead of the installed FLA. No package installation,
clock change, remote compilation or model data is performed by this script.

Upload the small `results.tar.gz`: summaries, exact commands, prior passed-run
receipts and their binary identity, current numerical receipts,
raw counter CSV and FLA source snapshots. The large synthetic `operands.pt` and
original `.acurep` files stay in the printed box directory for later re-import.
No CSV copy/paste is needed. Partial/failing captures also produce a diagnostic
tar; missing FLA is never relabeled as an ours-only comparison PASS.

Local host gates and metadata tests are not PPU execution. The host stand-ins
test pointer order, protocol and negative controls only; new timing remains
NOT_RUN until this command completes on the device.
