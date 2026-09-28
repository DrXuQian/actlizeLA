# PPU1.0: omit first-chunk zero-history products

Parent: `8bbd3dc220367933add1cc5d606e0fa9e54231a0`.
Branch: `agent/ppu10-first-chunk-20260928`.
Plan registered locally before edits: 2026-09-28 08:19:31 UTC.
Goal: transfer the retained SM90 architecture-independent mechanism, **not**
the suspended 1.5x FLA target. No default routing or SM90 changes.

## Contract and implementation

The explicit `first-chunk` residual delivery admits only positive `S % 64 == 0`
and **absent** initial state (`None`, not a tensor containing zero). Every
explicit initial state and every tail falls back to the unchanged
`full-chunk` entry, which retains its own generic tail fallback.

For finite operands, `K @ H0` and `Q @ H0` are zero when `H0 = 0`.
Only those two matrix products and their shared operand reads are omitted in
chunk zero. The accumulators start at positive zero; all subsequent scalar
operations, FP32 association, BF16 rounding boundaries, state decay and
updates remain in their original order. This is not an arithmetic fast-math
mode, reset approximation, or a gate-dependent truncation.

| Stage | Control | Candidate |
|---|---|---|
| Prefix | existing prefix symbol | same symbol |
| Inverse | static solve, three-product TF32 | same symbol |
| State | full-C64 state | no-initial specialization; skip KH only at ct=0 |
| Output | HV-layout output | skip QH only at ct=0 |

There are still four launches. Grids, 256-thread state/output CTAs, AIU.swzl
writers, matching ld.swzl readers, workspace extents and barriers are unchanged.
The first H snapshot **is still initialized and published**. Output still
stages that defined snapshot using AIU; it merely omits its shared-to-register
reads/MMA. There is no uninitialized allocation read, suppressed publication,
or new claim of reduced global-to-shared matrix traffic.

Two isolated device TUs retain immutable old bodies. A source gate removes
just the history guards/restores initial-state loading, then requires exact
parent-body equality. The host composition has a separate cross-TU/link gate.
This bounded A/B duplication is intentional; do not template-refactor the
controls while trying to measure one mechanism.

## Expected work, not measured speed

Priority shape: B1/S2048/Hk16/Hv32/K128/V128/C64, BF16 input, FP32 state,
natural-log gates -1 and -0.1, initial=None, final state enabled.

| Counter | Parent logical MMA count | Candidate prediction | Removed |
|---|---:|---:|---:|
| State BF16 m16n16k16 | 655,360 | 647,168 | 8,192 (1.25%) |
| Output BF16 m16n16k16 | 524,288 | 516,096 | 8,192 (1.5625%) |

These are independently dimension-derived **warp MMA counts**, not ACU
results. The state native guard covers 8 static MMA / 16 shared matrix-load
sites; output covers 16 MMA / 24 loads per panel. There are 32 chunks, so this
is a small saving at S2048. No substantial whole-call speedup is promised.
Short sequences have a larger fraction of zero-history work, but their
performance is not inferred from the S2048 capture.

Local HGCC 2.1.1 native results:

| Body | Static instruction sites, parent -> new | Registers | Stack | Shared bytes |
|---|---:|---:|---:|---:|
| State | 1085 -> 1030 | 120 -> 122 | 0 | 46,080 |
| Output | 1405 -> 1433 | 70 -> 70 | 0 | 49,408 |

State pays two more registers; output grows statically despite fewer dynamic
first-chunk operations. Both costs stay visible. This is precisely why a
correct port is not automatically a performance admission. The uploaded box
previously used HGCC 2.2.0-dev; its actual native resources are authoritative.

## Verification and device verdict

Local evidence:

- Real SDK compilation and linked Python extension, no device execution.
- All 40 old native instruction/operand sequences and resource/ABI records
  identical to the same-SDK parent build.
- CPU algebra: 58 finite/signed-zero cases, RAW-BIT against the admitted
  residual algebra plus independent recurrent oracle; three semantic plants.
- Actual CuTe ownership: 131,074 extent/initial-state combinations,
  524,800 full chunks, exactly 1,024 first chunks; 16,384 H and 8,192 output
  elements owned exactly once. Four negative controls.
- Source/native/linked ABI plants include ignoring supplied state, skipping
  the second chunk, uninitialized output, missing snapshot publication,
  omitting only one guard, wrong binding, guard inversion and renaming the
  old unguarded body. All must reject.

Local runtime import is separately **SKIP**: this host lacks GLIBC_2.38
required by the installed SDK runtime. Compilation/host checks are PASS;
device numerics and performance are **NOT_RUN**, not implied PASS.

The box runner admits both deliveries with the existing 36-case x8 device
gate: independent 2% output/state oracle and residual RAW-BIT equality,
including nonzero initial state, tails, GVA, variable gates/beta, output-only,
and immutable inputs. It additionally checks 16 signed-zero/None-vs-explicit-
zero cases x8. A failed gate stops before profiling; no tolerance changes.

Then capture fresh control/candidate/FLA at **both** gates. Score the sum of
**all** kernel durations per complete call (currently 4/4/7 including FLA
fills), not API event timing, not just the changed state. Verify first-chunk
symbols, the predicted matrix counts, resource changes and both signs.

- Correct and lower complete time at both gates: observed gain, pending
  repeat/non-regression admission before any routing change.
- Correct but unchanged/slower: migration evaluated; keep the control.
- A regression at either gate remains reported; do not hide it in averages.
- Unexpected numerical differences, missing kernels, spills or compatibility
  lowering invalidate the experiment. A different valid codegen layout needs
  explicit inspection, not a relaxed checker.

## Run and upload

On an idle PPU, from this branch:

```bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_first_chunk_box.sh
```

It uses `/sim/eec/shared/junfu.qx/asight/bin/acu`, builds once, admits, captures
two paired reports, and prints **one** upload path:
`/workspace/actlizeLA-ppu10-first-chunk-.../first-chunk.tar.gz`.
Upload that tar. It includes both nested ACU bundles, numerical/native/host
receipts, the measured SHA and binary hashes. The retained collector includes
sources, tool/runtime/device identity, actual native code and per-PC metrics.

SDK precedence: PPU_SDK, PPU_SDK_ROOT, then /usr/local/PPU_SDK. Fresh output
directories use mkdir under /workspace. No remote compilation is performed
by the agent. The packer can package an existing complete run without reruns:

```bash
bash tools/pack_ppu10_first_chunk.sh /workspace/ACTUAL_RUN_DIRECTORY
```

Mixed full/tail optimization, inverse operand retention, paired conversion
applicability and public PPU shape selection remain separate migration items.
