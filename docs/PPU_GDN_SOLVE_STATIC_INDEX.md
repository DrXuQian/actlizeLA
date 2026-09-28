# Static register indexing in the FP32 diagonal solve

Opt-in candidate `residual-solve-static`, control `residual-warps8-hvlayout`
(HV). Parent `a96bf0d`; no auto-routing or precision change. Metadata lookahead
was rejected as a speed optimization and is **not** part of this candidate.

## Measured problem, bounded change

Workload B1/S2048/Hk16/Hv32/K128/V128/C64, g=-1, PPU1.0. The preceding
site-ACU capture has HV prefix/solve/state/output times
2.40000/52.90647/89.88176/41.81941 us, total187.00764 us. FLA's complete
seven-kernel total is223.44057 us. These are historical **control** results,
not new candidate timings. See `PPU_GDN_RESIDUAL_METADATA_ACU_20260925.md`.

Actual old solve PCs execute196608 `v.mov ... ivreg` reads. Its diagonal
region has612static sites/5,013,504executed instructions, including466944
`pipe_flush` waits. Whole solve has14,767,104instructions and704512pipe_flush.
Source `CUTE_UNROLL` did not eliminate the inner register-array loops.
These are counts, not wall-time shares or a claim that every wait costs a cycle.

The new header `wy_solve_static.cuh` instantiates row and inner-term indices
as template constants. It retains the same ascending row/k order and
`x -= lower * column` expression. No algebraic reduction/reassociation and no
precision shortcut: the off-diagonal solve still evaluates all three TF32
high/residual products; lower/inverse remain FP32; BF16 boundaries stay fixed.

The separate `gdn_wy_solve_static_ppu.cu` contains one new kernel. A source
gate substitutes only this diagonal seam and compares the rest of the body
against the original solve. Prefix and the measured HV state/output are calls
to their **existing exact device symbols**, not new copies. The small additions
to their files are host-only reuse entrypoints; original device bodies and
launchers remain unchanged. Grid,128threads,49664Bshared, all retirement
barriers, layouts, inverse allocation/lifetime and public output/state stay fixed.

## Local proof and native screen

The host test runs the actual production template helper against an independent
rolled scalar oracle. All64tails,8distinct signed coefficient fixtures,4K-blocks,
32lanes and16rows are enumerated:65536contexts,1048576values,
7864320orderedproducts and524288published entries. Guard values check that
inactive lanes and other matrix entries stay untouched. This does not exhaust
floating-point inputs or replace the device numerical gate.

Wrong-column, reverse-term-order, omitted-term and missing-final-row plants
all change the oracle; an actual run omitting the final context must fail the
fixed denominator. Ten source/binding negatives cover stale dispatch,
wrong warp base, wrong owner, aliasing and changed arithmetic. Native negatives
include the **old indirect-index body under the new symbol**, incorrect FMA
sign, missing TF32, missing vector publication and missing retirement barrier.

SDK2.1.1, PPU1.0, same native compile flags:

| Property | Original solve | Static-index solve |
|---|---:|---:|
| Vector registers/thread |84|84|
| Stack bytes/thread |0|0|
| Whole-kernel static sites |1678|1472|
| Diagonal static sites |612|405|
| Diagonal indirect-register sites |12|0|
| Diagonal pipe-flush sites |42|0|
| Diagonal register-index backedges |present|none|
| BF16/TF32 MMA static sites |8/12|8/12|
| CTA barrier static sites |5|5|

The new diagonal has120round-to-nearest negative-product FMAs,120FP32
coefficient words read and16FP32row stores. Static addressing also lets the
compiler combine coefficient reads into8scalar,8two-word and24four-word
shared loads:120words, not fewer required coefficients. Forward branches
remain for lane ownership; "no backedges" is not "no control instructions".
The unchanged three-product TF32 scheme is not FLA's single-TF32 scheme.

Native checks keep all useful non-diagonal math/rounding, matrix transfers and
barriers fixed. Linked-library validation requires35/35previous instruction+
operand sequences identical to the same-SDK parent, with exactly one new
kernel (36total). Counts/zero-spill are compile facts, **not performance**.

Evidence directory: `/workspace/gdn-wy-solve-static-evidence-20260925`.
Reproducible commands are `configure.sh` and `verify_local.sh` there.
Final `replay-local.log` exited0:23/23CTest,95interface/capture contracts,
7compiler-dialect checks,45WY+61residual algebra cases,305original source
controls,36WY+15original native images and all negative controls PASS.
35/35old native instruction/operand sequences are identical. The earlier
full replay caught an ambiguous old negative-test symbol prefix after the
new solve was added; it now selects the exact old symbol and the entire suite
was rerun. No negative was removed or reclassified as a skip.

Local DSO SHA256:
`19c4545491e44ce63b94d89900997e47ddf6e3564e3c8f379740b21ae02798ca`.
Local Python binding SHA256:
`62e1ce3ad93d7afe30f59da077101442682f1f10f6442ea5a25b6b848db3e691`.
The runner rebuilds on box and records its actual binary/compiler identity;
these hashes identify local compile evidence only. The archived originalC32
experiment still has144Bstack; the zero-spill statement is for this solve pair,
not every archived kernel.
Device correctness and speed: **NOT_RUN locally**.

## Box command and predeclared interpretation

On updated GDN-QSA-sm80 `ppu-backend`:

```bash
DEVICE=0 JOBS=16 CANDIDATE=residual-solve-static \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

`PPU_SDK` is optional if not installed at `/usr/local/PPU_SDK`. `OUT` defaults
to a fresh `/workspace` directory. ACU defaults to
`/sim/eec/shared/junfu.qx/asight/bin/acu`. The script prints the tar.gz to upload.
No PPUProfiler; no request to paste CSV.

Before profiling, run30scalar+30HV+30candidate cases, eight repeats, RAW-BIT
equality and the unchanged independent2%recurrence gate. Include tails, GVA,
variable metadata, nonzero initial state and output-only. Any failure stops
timing; close numerical error alone cannot admit a changed rounding scheme.

Profile the same binary/fixture/device sequentially: HV versus static-index
versus FLA. Expected subject kernels: old prefix, **new static solve**, oldHV
state and oldHVoutput. Compare **all4/4/7kernel durations**, not API dispatch
spans. Bind SHA/compiler/binaries/device and exact captured symbols.

If solve and complete time improve, retain for confirmation, not automatic
routing. If indirect access disappears but time does not improve, record **no
observed speed gain**; a slowdown is a rejection, not a hidden win. Report
unchanged-stage variation separately. Inspect dynamic instructions, shared
requests, waits and resources without treating them as latency shares.
The1.5xFLA goal would require148.96038us on the previous capture; even closing
the entire old solve gap cannot supply all38.04726us still needed. No claim
that this isolated change will reach1.5x.
