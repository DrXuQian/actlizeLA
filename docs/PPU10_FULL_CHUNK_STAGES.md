# PPU1.0 full-C64 solve/output: registered experiment

Parent: e379fea12adb07a3d8917b4b730bcfee10e0649d.
Branch: agent/ppu10-full-chunk-stages-20260928.
Control: residual-full-chunk, the already measured full-C64 **state** path.
No default routing, numerical tolerance or SM90 implementation changes.

## What changes

The retained control already specializes state, but solve/output still receive
runtime valid-row counts. This extends the same exact full-chunk optimization
to those two stages, separately and together:

| Explicit delivery | Solve | State | Output |
|---|---|---|---|
| full-chunk (control) | retained static solve | retained full-C64 | retained HV layout |
| full-chunk-solve | new full-C64 | same symbol | retained HV layout |
| full-chunk-output | retained static solve | same symbol | new full-C64 |
| full-chunk-both | new full-C64 | same symbol | new full-C64 |

Prefix is the same symbol in all four arms. Each complete call still launches
four kernels. The stage selection is compiled into the host launcher, not a
new device branch. All arms require positive sequence length divisible by64;
other lengths delegate to the unchanged full-chunk control and its guarded
tail path. Interior/tail partitioning for a length such as2051 is deferred.

Only these source seams may differ:

- Solve: valid=64; remove beta's redundant clamped row and in-range predicate.
- Output: valid=64; remove row<valid from the attention predicate.
  The causal row>=col mask is retained.

Source comparison restores just those seams and requires exact equality with
the old kernel bodies. Three-product TF32 inverse, FP32 accumulation order,
expf, BF16 rounding, AIU.swzl/ld.swzl pairing, shared layouts, grids, barriers
and workspace sizes are unchanged. In particular this is not the still-pending
warp-local off-diagonal inverse rewrite.

Two separate CUs preserve the old generated code. A host-only composition CU
reuses the exact full-state symbol; the old public entry is untouched.
This bounded duplication is intentional for the experiment, with a source
equality gate preventing drift. Do not refactor controls into a template during
the A/B: a previous such factoring changed their generated instructions.

## Local evidence, not a performance result

Real HGCC2.1.1-a5c56e, native PPU1.0, pinned actlize423253c:

| Stage | Control -> candidate static sites | Registers/thread | Stack | Shared |
|---|---:|---:|---:|---:|
| Solve |1472 -> 1451|84 -> 84|0|49,664 B|
| Output |1405 -> 1378|70 -> 70|0|49,408 B|
| State, reused |1085 -> 1085|120 -> 120|0|46,080 B|

No new register-residency advantage is claimed. Useful MMA/FP32 arithmetic,
exp2, AIU/SWZL transfers, global stores and synchronization opcode counts
are unchanged. Both new bodies lose the runtime s.min. Static counts cannot
predict dynamic speed; the prior box used HGCC2.2.0-dev and its native body
sizes differ from this local SDK.

All38 old native instruction+operand sequences and all38 resource/ABI records
are identical to the same-SDK parent. Actual compiled Python specializations
resolve to four distinct, correct C ABI entries; all three new entries resolve
to the intended stage launches and tail fallback.

The native CuTe C-layout host proof exhausts65,537 lengths,524,800 full chunks,
4,096 inverse cells,4,096 attention cells and8,192 output cells, with the strict
lower/causal denominators fixed independently. Missing lane, lost causal
diagonal and shortened denominator are negative controls. Source/native/link
plants include removing MMA, exponent or barrier, renaming the old body,
wrong compiled Python selection, and modifying an old resource record.

## Device admission and verdict, fixed before results

Priority shape: B1/S2048/Hk16/Hv32/K128/V128/C64. BF16 inputs, FP32 state,
natural-log gates g=-1 and -0.1 separately; zero initial and enabled final state
for profiling. Admission additionally covers tails, GVA, nonzero carried state,
variable gates/beta, output-only calls and immutable inputs.

Each delivery must pass the existing36-case,8-repeat device gate: independent
2% numerical oracle plus exact RAW-BIT equality to scalar residual. All four
arms are admitted before any ACU capture. A failed numerical/native gate stops
the runner; no tolerance relaxation or fallback masquerading as a candidate.

For each of solve/output/both, capture a fresh same-binary control, candidate
and FLA at each decay. Use **all kernel durations summed per complete call**:
4/4/7 for the current implementations, including FLA fills. Verify actual
stage symbols against the table above, not just the selected variant label.
If FLA's kernel inventory changes, report the difference and keep every kernel;
do not force a seven-row denominator by omitting helpers.

Report each stage, the whole call, unchanged-stage variation, dynamic opcodes,
registers/spills, occupancy and shared traffic. API-event spans are not this
experiment's score. No candidate's favorable stage counter can replace its
whole-call result.

- Lower complete time at both gates: observed candidate win, pending repeats.
- Fewer instructions but unchanged/slower time: no performance admission;
  retain control and report the unfavorable result.
- A regression at either gate remains visible; no automatic routing change.
- New spill/register increase, numerical change or extra matrix traffic:
  invalidate this bounds-only experiment.
- The overall1.5x target requires candidate sum <= same-capture FLA sum/1.5
  at both gates; one capture per cell is not broad selector admission.

The historical full-state results168.92412/169.36236 us are not fresh controls.
Relative to their own FLA arms, the remaining budget was18.70216/17.44550 us.
This smaller bounds-only edit does not promise to close that entire gap.

Local gates: 222 Python unit/contract tests, 26 host CTests and 61 CPU algebra
cases pass. All 68 whole-library and 28 new source/native/link/resource
negative controls reject their planted defects. The actual runtime import is
SKIP: this host lacks GLIBC_2.38 required by the installed SDK runtime.
See the [hash-bound local evidence](../dev/ppu/results/full_chunk_stages_local_20260928.json).

## Run

On an idle PPU, in this experiment's actlizeLA branch:

~~~bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_full_chunk_stages_box.sh
~~~

It builds once, runs admission, then captures six paired bundles using
/sim/eec/shared/junfu.qx/asight/bin/acu directly. SDK precedence is PPU_SDK,
PPU_SDK_ROOT, then /usr/local/PPU_SDK. Outputs use fresh /workspace directories;
no manually specified binary or profiler API is needed. The script prints
each upload tar path. SHA, source snapshot, submodule, native code/resources,
binary hashes, actual runtime/device identity and input/output fingerprints
are collected by the retained runner.

Device correctness and performance are NOT_RUN locally. Successful local
compilation/linking and CPU proofs are not a substitute for box admission.
