# GDN versus FLA: one-command counter bundle

## Profiler selection (2026-09-24)

All new box captures default to the operator-requested
`/sim/eec/shared/junfu.qx/asight/bin/acu`, independent of the compiler SDK
and PATH. If it is missing or not executable, capture fails; there is no
automatic fallback or retry with a different ACU. An explicit `ACU` override
is retained for deliberate tool comparisons. The actual path, binary hash,
version and loaded profiler libraries remain part of every bundle.

This selection supersedes the earlier SDK-first discovery rule below. It
does not rewrite the identity or numerical provenance of existing reports.

## Current state-pipeline result

[The verified latest capture](PPU_WY_STATE_PIPELINE_ACU_20260924.md) compares
split-prepare / state-pipeline / FLA, all5/5/7 kernels. The handoff is
[the state-pipeline wrapper](PPU_WY_STATE_PIPELINE.md); do not select an
older AIU/shared-row arm as the current incumbent.

## Historical: same-binary shared-row / AIU-both / FLA capture

Capture completed and verified: [AIU phase/counter result](PPU_WY_AIU_ACU_20260924.md).
The command below is retained for reproduction, not a request to repeat it.

User-reported API medians: shared-row354.364 us, AIU-both318.564 us,
FLA492.188 us. These are not per-kernel ACU durations; see
[the result scope](PPU_WY_AIU_PAIR.md). Keep that measured binary immutable.
Use the just-completed `AIU_AB=1` run, not a freshly compiled library or the
older shared-only run. Paste its `artifacts=` directory when prompted:

```bash
git pull --ff-only origin ppu-backend &&
read -r -p 'Completed AIU comparison artifacts directory: ' WY_RUN &&
env -u OUT -u EXTENSION \
  DEVICE=0 PPU_SDK=/usr/local/PPU_SDK \
  ACU=/sim/eec/shared/junfu.qx/asight/bin/acu \
  bash tools/run_ppu_gdn_fla_acu_box.sh \
    --wy-run "$WY_RUN" \
    --wy-control prepare-rows-shared \
    --wy-delivery aiu-state-output --gate -1.0
```

Select the same physical `DEVICE` as the preceding run. This first capture
explicitly chooses strong decay -1.0 for continuity with the previous ACU
baseline; the user's unlabeled median table is not asserted to be that case.
The copied comparison retains both gates. `--gate -0.1` selects a separate
weak-decay capture when needed. Do not overlap profiling with timing workloads.

No compile, installation or timing sweep. Three independent preflights run
first, then direct `acu -f -o ... --set full` in this fixed order:

| Capture | WY mask | Native reports inside the bundle |
|---|---:|---|
| old best: prepare-rows-shared |1520| `wy-control/wy-g-1.0.report[.acurep]` |
| candidate: aiu-state-output |13808| `wy-g-1.0.report[.acurep]` |
| FLA reference |not applicable| `fla-g-1.0.report[.acurep]` |

FLA is captured once. The two WY arms must use the same binding/device DSO,
input, physical-device receipt and raw output/state, differing only in the
requested delivery. A missing arm, mask mismatch, changed loaded library,
failed profiler or missing report keeps the bundle INCOMPLETE. Both preflight
and subject must retain the selected delivery. Existing two-arm mode is
unchanged when `--wy-control` is absent.

Early AIU benchmark JSON has `delivery_ab=false` because that summary field
omitted the new family. Its concrete arms and `delivery_mask` values remain
valid evidence. The collector now checks those exact records and numerical
fingerprints instead of trusting the summary boolean. **Do not edit the old
comparison JSON or rerun the sweep to repair this metadata.** New runs record
the family flag correctly. Non-scalar arms without an exact recorded mask
are rejected, not inferred from identical output bits.

Upload the single file printed as `UPLOAD=/workspace/...tar.gz`. It contains
all three native reports, automatic text exports, receipts, loaded hashes,
ISA/resources and the unchanged preceding comparison. No CSV copy/paste is
required. All artifacts use a new directory, without overwriting old runs.

Read prepare/state/output separately. Prepare is unchanged and serves as a
control; state/output share the same math but use paired AIU/SWZL delivery.
Inspect actual frequency, MMA work, native instructions, traffic and stalls
before attributing a duration change. FLA preparation can span several
kernels; retain fills and transforms separately. ACU replay times are
diagnostic and cannot be subtracted from full-API medians as host overhead.
Local mock capture/negative tests pass. The user capture is now verified;
no PPU kernel was run locally during the analysis.

## Previous: prepare shared-row candidate, mask1520

Capture completed: [verified stage/counter analysis](PPU_WY_SHARED_ACU_20260922.md).
The following command is retained for reproduction, not a request to rerun.

The latest user-reported strong-gate full-API median is352.192 us, versus
356.590 us for prepare-address and494.490 us for FLA. Capture the **same
measured binary**, not a new build or the older mask48. The exact run
directory was not included in that excerpt; paste the directory printed by
its final `artifacts=` line when the following command prompts:

```bash
read -r -p 'Previous comparison artifacts directory: ' WY_RUN
if [ -n "$WY_RUN" ] && [ -f "$WY_RUN/comparison.json" ]; then
  env -u OUT -u EXTENSION \
    PPU_SDK=/usr/local/PPU_SDK ACU=/sim/eec/shared/junfu.qx/asight/bin/acu DEVICE=0 \
    bash tools/run_ppu_gdn_fla_acu_box.sh \
      --wy-run "$WY_RUN" --wy-delivery prepare-rows-shared --gate -1.0
else
  printf '%s\n' 'Missing comparison.json; no capture started. Use the preceding artifacts directory.'
fi
```

Use the same physical DEVICE as the comparison. There is no new binary,
kernel edit, timing sweep or automatic latest-directory selection. The
existing collector supports this exact role and refuses a different role
even if it has the same numerical fingerprint. It validates both DSOs,
the prior fixture/FLA contract and new separate preflight before capture.
Current helper sources and old binary origin are recorded separately.

Source/compile expectations for B1/S2048/Hk16/Hv32/K128/V128/C64:

| Stage | Expected native body | Grid / threads | Shared B / local registers |
|---|---|---|---|
| Prepare | `gdn_wy_rows_prepare<1>` | 1024 /128 | 70,144 /86 |
| State | `gdn_wy_state_ab<true,true>` | 128 /128 | 49,408 /242 |
| Output | `gdn_wy_tiled_output` | 1024 /256 | 49,408 /98 |

These are **expected mappings and local compilation resources**, not the
new box's observed occupancy. Check report symbols and actual resources
before attribution. FLA's prefix/solve/WU together correspond to prepare;
state and output compare separately. Keep additional fills/transforms and
any fresh-process internal autotuning visible rather than discarding them.

The next change is selected from current measured stage duration, actual
work/traffic, issue/dependency stalls, active warps, spills and frequency.
The shared cache's4.398 us API improvement does not identify any of those
causes by itself. Do not reuse old scalar/tiled stage times to rank this
new pipeline or subtract profiled sums from352.192 us. No weak-gate claim
or routing promotion follows from this strong-gate capture.

The collector uses direct `acu -f -o ... --set full`, not PPUProfiler;
captures WY and FLA sequentially; retains all stages; and prints
`UPLOAD=/workspace/...tar.gz`. Upload that one archive, not screenshots or
manually copied CSV. It includes the complete preceding comparison, so the
weak-gate samples and admission log can be reviewed if present. All24 host
capture-contract tests pass; actual capture is still pending the box run.

## Earlier: state/output combination from 90eafeb

The reported strong-decay state/output result is 439.486 us. Reuse that exact
comparison binary and select mask48, not scalar WY, old packed-all or tiled-all:

```bash
git pull --ff-only &&
env -u OUT -u EXTENSION \
  PPU_SDK=/usr/local/PPU_SDK ACU=/sim/eec/shared/junfu.qx/asight/bin/acu DEVICE=0 \
  bash tools/run_ppu_gdn_fla_acu_box.sh \
  --wy-run /workspace/gdn-wy-fla-90eafeb-20260922T013928Z \
  --wy-delivery tiled-state-output --gate -1.0
```

Use the same physical DEVICE as that comparison. This captures WY and FLA
sequentially, all stages, with no rebuild. Upload the tar named by `UPLOAD=`;
the previous full comparison JSON joins it, including the weak case if present.
Read [the result and per-stage interpretation](PPU_WY_STATE_OUTPUT_RESULT_20260922.md).
The pair wins versus same-run scalar/state/FLA, but pair-versus-all and
pair-versus-original remain UNRESOLVED. No routing change follows.

## Earlier scalar-only WY follow-up: reused binary, no rebuild

For the `dd70e5d` WY comparison already run on the box:

```bash
git pull --ff-only
DEVICE=0 bash tools/run_ppu_gdn_fla_acu_box.sh \
  --wy-run /workspace/gdn-wy-fla-dd70e5d-20260921T041641Z
```

Use the physical `DEVICE` of that comparison. This selects **WY**, not the
original API denoted by `ours` in older captures. Both roles print their
implementation; the native reports are named `wy-g-0.1.report.acurep` and
`fla-g-0.1.report.acurep` (some ACU versions omit the appended extension).
This mode **never builds or installs anything**. It finds exactly one
`build/_gdn_wy_ppu*.so`, checks it and `libgdn_wy_ppu.so` against that run's
`binaries.sha256`, and checks the binding against `comparison.json` too.
A replaced/missing/ambiguous binary fails, not a silent rebuild or fallback.
Do not also set `EXTENSION` in this mode.

The complete old samples, manifest and source SHA/diff join the bundle.
Preflight verifies identical input, output fingerprints, numerical contract,
FLA identity and device properties before ACU starts. The original comparison
did not record a device UUID: matching properties alone **cannot** prove the
same physical card across runs. Current capture receipts do record UUID when
available. Updated checkout sources describe the capture helper, **not** the
origin of the reused kernel binary; the old SHA/diff remains its provenance.

Upload the single tar printed after `UPLOAD=`. No CSV copy-paste or screenshot
is needed. `--gate -1.0` explicitly selects the separate strong-decay control.

### Profiler exception reported on 2026-09-21

The user capture at `/workspace/gdn-qsa-acu-20260921T070716Z-252514` failed in
WY's first profiled call with `IndexError: map::at` and no profiled kernels.
The collector reaches this point only after both independent preflights pass.
This is a capture failure, not a new output/state mismatch. Its Python stack
does **not** identify the C++ map that threw. In particular, it does not prove
bad tensor indexing, a failed shared-memory attribute call or bad replay.

The old discovery order preferred the shared site's ACU over the chosen SDK.
An earlier saved capture of that same site path reports ACU
`v2.0.0_20251231-4f7cd70` / data 12006, while HGGC is 2.1.1. The local SDK's
own ACU reports `v2.1.1_20260725-15d8b9d` / data 15000. The failed run's tool
version has not yet been supplied, so the old path alone is **not proof** of
which binary version it used or of the exception's root cause.

At that historical checkpoint discovery was changed to SDK-first. The
2026-09-24 operator choice above supersedes it. This earlier explicit command
is retained only to reproduce that historical tool comparison:

```bash
PPU_SDK=/usr/local/PPU_SDK ACU=/usr/local/PPU_SDK/asight/bin/acu DEVICE=0 \
  bash tools/run_ppu_gdn_fla_acu_box.sh \
  --wy-run /workspace/gdn-wy-fla-dd70e5d-20260921T041641Z
```

The selected path/version is printed before capture; failure to run that tool
is a failure, never an automatic retry with an older profiler. Native failures
now retain a FAIL receipt with actual runtime, profiler and binary-analysis
library paths/hashes, then rethrow the original exception without retrying the
call. The bundle stays INCOMPLETE. No extra warmup, rebuilt kernel, tolerance
change or exception suppression is used. If the matched-tool capture still
throws, those identities and a native throw backtrace are the next diagnostic;
this selection repair is **not** claimed to have reproduced or fixed the
vendor exception locally. There is no local PPU execution.

### The next decision is per phase, not a new winner claim

At B1/S2048/Hk16/Hv32/K128/V128, the WY source and local compilation predict:

| WY kernel | Corresponding FLA work | WY grid / threads | Shared bytes / vector registers |
|---|---|---|---|
| `gdn_wy_prepare` | gate prefix, KKT/solve, W/U | 1024 / 128 | 70,144 / 84 |
| `gdn_wy_state` | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 128 / 64 | 37,120 / 244 |
| `gdn_wy_output` | `chunk_fwd_kernel_o` | 1024 / 128 | 73,984 / 80 |

These resources are **local compiler observations**, not box occupancy or
performance evidence. All three locally have zero stack. Inspect actual
resources from the reused box binary and all kernels in the FLA report;
FLA's installed version may fuse or split preparation differently. Include
clears and transformations separately; do not mistake extra tuning launches
for a longer steady-state algorithm. No stage is filtered out of capture.

For each stage compare duration, launched work, MMA count, active warps,
register/stack/shared limits, achieved frequency, memory dependency, sync,
fetch stalls and traffic. Optimize the largest **measured** gap first:
preparation fusion may cost occupancy, state may cost register delivery, and
output may cost fragment/shared delivery. They remain hypotheses until the
new capture, not three changes to make together.

The ~715 us WY versus ~500 us FLA engineering target comes from unprofiled
complete-API samples. **Do not subtract the sum of ACU replay durations from
715 us and call the residual CPU overhead.** Cache, replay and process state
are different. The API also includes allocation and submission gaps; WY calls
`hggcFuncSetAttribute` three times per API invocation, whose cost is currently
unmeasured. Host-gap attribution needs a separate timeline if it remains the
unexplained part. The prior samples and their UNRESOLVED verdict are preserved
in [PPU_WY_FIRST_DEVICE_RESULTS.md](PPU_WY_FIRST_DEVICE_RESULTS.md).

## Original path (retained control)

From the `ppu-backend` checkout on the PPU box:

```bash
git pull --ff-only
DEVICE=0 JOBS=16 bash tools/run_ppu_gdn_fla_acu_box.sh
```

Use the same physical `DEVICE` as the preceding latency comparison. The
default SDK is `/usr/local/PPU_SDK`; override `PPU_SDK` only if needed. ACU is
fixed by default to `/sim/eec/shared/junfu.qx/asight/bin/acu`;
`ACU=/absolute/path/to/acu` explicitly overrides that choice. Installed FLA is
used by default; optional `FLA_ROOT` selects an existing checkout. No clone,
pip install, SSH, clock setting, or mandatory identity form is involved.

The script builds locally **on the box**, against its installed PyTorch,
then captures ours and FLA sequentially. It does not modify kernels or run a
performance sweep. To reuse a particular build, set `EXTENSION` to its exact
`_gdn_chunk_ppu*.so` filename, with `libgdn_qsa_ppu.so` beside it. That mode is
explicitly recorded as operator-supplied: the current checkout is **not**
asserted to be the origin of an old binary. The actually loaded binary
hashes must match the archived files.

Artifacts use a new `/workspace/gdn-qsa-acu-<UTC>-<pid>` directory (`mkdir`,
not `mktemp`). `OUT` may specify another **new** directory. Existing output
directories are refused, not deleted. Upload the **single `.tar.gz` file**
printed after `UPLOAD=`. The tar contains native `.report[.acurep]` reports;
there is no GUI screenshot or CSV-copying requirement.

## Fixed subject and capture scope

- Weak decay g=-0.1, BF16, B1/S2048/Hk16/Hv32/K128/V128, native GVA.
- The same CPU fixture/seed, recurrent oracle and fixed 2% output **and**
  final-state criterion as `bench_ppu_gdn_fla.py`. Inputs/reference hashes
  must match across arms. Start with zero state; return final state; no QK
  normalization. FLA's Triton path is explicit, not FlashQLA dispatch.
- Follow `quactlize/tools/run_dense_marlin_m8_acu_box.sh`: first run a
  **separate unprofiled preflight process** (first call + five warmups with
  correctness/fingerprint checks), then run a **subject-only process** under
  `acu -f -o ... --set full python ... --phase subject`.
- There is no profiler API, dynamic profiler-library lookup, start/stop
  range, launch-skip/count guess, or kernel-name filter. The subject process
  calls forward exactly once, with zero warmups. Verification copies results
  to CPU **before** dtype casts/comparisons, so it adds no verification GPU
  kernels. Its output/state must match the separate preflight bit-for-bit.
- ACU profiles the **whole subject process**: runtime setup if any, gate
  reduction, head expansion, clears, transpose, preparation, recurrence,
  and FLA's complete pipeline. Preflight and subject inherit the same Triton
  disk-cache environment, but in-memory autotuner choices are not carried
  between processes. If the installed FLA repeats library-internal tuning
  in the fresh process, those launches remain in the report. Do not claim
  the report excludes all JIT/autotuning or contains only steady-state work.
  One public API call is not necessarily one kernel launch/replay.
- ACU's default profiling/cache policy is used. Profiled durations are
  **diagnostic**, not new winner timings. The profiler flushes caches and
  replays launches. Neither sum of profiled kernel times nor wall time of
  capture is the preceding unprofiled full-API event span. Host dispatch
  synchronization/gaps require separate timeline measurement if needed.

Use `bash tools/run_ppu_gdn_fla_acu_box.sh --gate -1.0` for the strong-decay
control in a separate bundle; it is not mixed into the default weak capture.

## What is archived

Both native reports, exact commands and complete ACU logs; automatic plain
text `details`/`raw` metric exports; per-arm correctness/identity JSON;
SHA/diff/submodule pins and relevant C++/Python source snapshots; actual
extension and device library, their hashes and PPU ISA/resource dumps;
imported FLA source files and its Triton compiler source/hash; torch/Python
versions/build information; SDK/ACU versions/help; read-only `ppu-smi -q`
before and after, if available. ACU's in-kernel frequency counters, not an
idle pre/post clock sample, determine the achieved frequency.

`STATUS.json` separates optional probe/export unavailability from capture
success. `SHA256SUMS` covers every other archived file. The archive's own
checksum is saved beside it. No full build directory, Triton cache, model
weights, tensors, credentials or environment dump is archived. A failed
build/import/capture still produces an **INCOMPLETE** diagnostic tar, never
a one-arm comparison PASS. The original numerical benchmark is unchanged.

## Questions this capture must answer (not preselected conclusions)

The user's unprofiled same-input results are 925.020 vs 726.992 us at g=-0.1
(FLA wins), and 457.340 vs 746.904 us at g=-1 (ours wins). Their complete
timing logs/binary identities are not yet in this repository.

On the current weak-path source, auto dispatch expands Q/K from 16 to 32
heads, builds per-chunk workspace, then runs serial recurrence with 32 CTAs,
128 C16 chunks per CTA. Only 128 of its 256 threads execute the C16 MMA
body. Those are **source facts**, not measured causal attributions. Determine:

1. Which kernels dominate: preparation/clears/expansion, or recurrence?
2. For recurrence: grid, resources/occupancy, active warps, tensor issue,
   memory dependency, synchronization, instruction fetch, traffic and spills.
3. For FLA: its actual kernel split, launch grids, resources and the same
   counters on the identical input/device. Compare work as well as duration;
   the two implementations need not use the same chunk size/rounding points.
4. Whether the weak-path gap is primarily insufficient independent work,
   state/fragment delivery, or public API preprocessing. Low occupancy alone
   does not establish which optimization will win.

Local tests validate separate preflight/subject call counts, CPU-only
verification, absence of profiler APIs, identity binding, report
selection, failure propagation and tar/checksum contracts. No local PPU is
available: actual ACU interception/counters remain **box-unverified** until
this command completes successfully. No new device speedup is claimed.
