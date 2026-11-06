# PPU1.0 public-default prefill: same-input ACU comparison

This measurement reuses the exact `.so` from a successful serving/auxiliary box
run. It does not compile kernels or change default routing. The Python/runtime
source is main/v0.2.2 (`6505987`); this diagnostic branch changes only measurement
helpers/tests/docs. The reused binary's own source SHA is reported separately.
This is a source-default comparison, not an installed-wheel or full-model claim.

## Boundaries

| Role | Input → output contract | Included work |
|---|---|---|
| `serving` | raw BF16 qkv/a/b + FP32 VK slot → output + final VK slot | validation, QK norm, gate/beta preparation, pack, state gather/transpose, residual, scatter/transpose |
| `default` | processed BF16 QKV/beta + FP32 log-g/KV state → output + KV state | public `gdn_forward(backend="ppu10")`, algorithm/delivery omitted, all four retained core kernels |
| `fla` | exactly the same processed operands and FP32 KV state | FLA Triton forward, including its fills and final state |

Only `default` versus `fla` receives a speed verdict. Comparing full serving to
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
The stated 5120 output tokens are not a GDN matrix dimension; decode/MTP and
complete model timing are outside this prefill-only measurement.

The untimed admission executes real serving prepare once and saves its exact
processed operands. Both core arms consume those same bytes. Scale is applied
once; FLA's internal QK normalization, sigmoid/gate activation and V-first state
interpretation are disabled. GVA stays native; no silent head expansion.

## Gates and measurement

- Serving: unchanged2% relative-L2 versus raw-input sequential reference.
- Core: unchanged2% max/max versus processed-input sequential reference.
- Serving/public-default canonical output and FP32 state: RAW-BIT equal.
- The real default's route and mapped image are checked before profiling.
  A scalar/legacy fallback or different library is an error, not an alternative
  measurement. The benchmark does not force a fast variant to hide default drift.
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
  DEFAULT_CORE_WINS and FLA_CORE_WINS are valid outcomes, not script failures.

## Run on an otherwise idle PPU1.0

```bash
bash tools/run_ppu_serving_prefill_acu_box.sh
```

Run from the `perf/ppu10-fla-v022` worktree, leaving the release checkout untouched.
The runner does not install packages, compile kernels, or replace the published
wheel. It prints one summary per shape, with all three totals and each kernel's
duration in `summary.json`.

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
Do not select a legacy `GDN_QSA_WY_EXTENSION` for this comparison; if present it
is rejected rather than silently replaced. `ACTLIZE_LA_SERVING_LIBRARY`, if set,
must name the same library as the chosen passed serving run.

The collector binds `ASIGHT_HOME` and `--section-folder` to that ACU installation
for capture AND import, and records executable/section hashes. It validates the
unchanged `full` set before numerical preparation. This prevents an inherited
SDK's `ASIGHT_HOME` from silently mixing another section schema with the selected
binary. An incompatible companion installation still fails; no section/metric
is deleted or silently dropped. On failure the last60 log lines are printed.

If `MemoryMetricTable` is rejected even with the explicit companion folder,
inspect `acu-section-check.log` and the selected installation's libraries/version;
do not modify GDN or infer a numerical failure from a profiler schema error.

Upload the small `results.tar.gz`: summaries, exact commands, prior passed-run
receipts and their binary identity, current numerical receipts,
raw counter CSV and FLA source snapshots. The large synthetic `operands.pt` and
original `.acurep` files stay in the printed box directory for later re-import.
No CSV copy/paste is needed. Partial/failing captures also produce a diagnostic
tar; missing FLA is never relabeled as an ours-only comparison PASS.

Local host gates and metadata tests are not PPU execution. The host stand-ins
test actual Python default routing, protocol and negative controls only; new timing remains
NOT_RUN until this command completes on the device.

Host gates: `python -m pytest tests/test_serving_prefill_perf.py
tests/test_ppu_fla_benchmark_contract.py tests/test_gdn_route.py -q`.
