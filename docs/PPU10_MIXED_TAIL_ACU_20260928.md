# PPU1.0 mixed-tail state: complete device evidence

Measured source `d8b8b8536ce9c957a15b312fcbce11d19a7c7ff0`. The contract
was registered before implementation in [PPU10_MIXED_TAIL.md](PPU10_MIXED_TAIL.md).
All four registered cells are present. No kernel or routing was edited while
adjudicating this upload.

Verdict: **device correctness validated; lower complete-call ACU time in all
four captures; retain opt-in pending repeated performance/non-regression
admission**. This closes the state interior/tail migration experiment, not
automatic default selection. S2048 still uses its exact old symbol. The old
1.5x FLA target remains suspended.

## Complete-call results

BF16 inputs, FP32 final state, B1/Hk16/Hv32/K128/V128/C64, initial=None.
All times below are sums of the complete captured call's kernels, in us.
There is one capture per arm/cell, not a distribution of repeated timings.
Each control/candidate has four kernels. Each FLA call has seven, including
both required fills. No API-event time or inter-launch gap is substituted.

| S | Gate | Control | Mixed-tail | FLA | Latency reduction vs control |
|---:|---:|---:|---:|---:|---:|
| 2049 | -1.0 | 181.90000 | 177.30707 | 227.70882 | 2.525% |
| 2049 | -0.1 | 182.26295 | 175.94001 | 228.87293 | 3.469% |
| 2111 | -1.0 | 180.73941 | 178.21471 | 231.32176 | 1.397% |
| 2111 | -0.1 | 182.64117 | 178.03883 | 232.08706 | 2.520% |

Observed speed versus FLA is 1.284x–1.304x. This is the measured tail-shape
scope, not extrapolation to other lengths or a new speed target.

| S / gate | State control -> candidate | State delta | Unchanged stages' combined delta |
|---|---:|---:|---:|
| 2049 / -1 | 91.08176 -> 85.91765 | -5.16411 | +0.57118 |
| 2049 / -0.1 | 92.18353 -> 85.90412 | -6.27941 | -0.04353 |
| 2111 / -1 | 90.03294 -> 86.61765 | -3.41529 | +0.89059 |
| 2111 / -0.1 | 90.97647 -> 85.61353 | -5.36294 | +0.76060 |

The three unchanged stages have identical executed opcode dictionaries in
their paired captures. Their timing variation is reported separately, not
credited to the state source edit.

## Mechanism, including costs that did not disappear

The old `full-chunk` entry delegates the entire mixed-length call to
`gdn_wy_residual_gate_cache_state`. The captured candidate is the actual
`gdn_wy_residual_mixed_tail_state`, not that fallback relabeled.

Per-PC execution binds to the uploaded native code: each of the 20 prefix
MMA sites executes 32,768 times; each of the 20 tail MMA sites executes 1,024
times. At 128 CTAs x8 warps this is exactly 32 full iterations plus one tail.
Both arms execute **675,840 BF16 MMAs**. FP32 state stays within the same CTA;
no extra launch or rounding handoff has been introduced.

| State counter / resource | Control | Candidate |
|---|---:|---:|
| `s.min.i32` executions | 33,792 | 1,024 |
| `v.csel.b32` executions | 270,336 | 8,192 |
| `v.mov.v2s` executions | 1,155,072 | 1,024,000 |
| Total per-opcode warp executions, S2049 | 14,012,928 | 12,692,992 (-9.419%) |
| Total per-opcode warp executions, S2111 | 14,056,576 | 12,841,984 (-8.641%) |
| Registers/thread, actual box | 124 | 120 |
| Stack bytes | 0 | 0 |
| Shared allocation bytes | 46,080 | 46,080 |
| Static native sites, including padding | 1,181 | 1,918 |

Executed work is smaller despite a larger static body. The constant prefix
component contains 410 sites on this box; its 32 iterations no longer pay
the generic tail predicates. The larger final-tail body executes once.
Box HGCC is 2.2.0-dev: local 2.1.1's 122 registers/1,909 sites are not the
box resources and are not substituted here.

AIU copies, matrix loads, MMA, FP32 arithmetic, exponents and CTA barrier
counts are unchanged. Same-S bank conflicts and measured KVD global-load,
global-store and global-to-shared transaction bytes are also unchanged:

| S | State shared BC, either arm | Global-to-shared bytes, either arm |
|---:|---:|---:|
| 2049 | 4,551,680 | 118,530,048 |
| 2111 | 4,575,232 | 121,069,568 |

This is a control/address simplification, **not** a bank-conflict or byte-
volume reduction. Active warps/CU remain about14.1; fewer registers do not
lift the fixed-grid ceiling. Memory-dependency/TFU-WAR per-issue ratios rise
while latency falls; those changing-denominator ratios are not wall-time
shares and cannot be subtracted from saved microseconds.

Do not claim every scalar operation is identical: generic tail publication
uses separate masked zero/value stores. `tsm.st.b16` warp executions fall
by512 despite unchanged transaction bytes. BF16 conversion counts change
by+2,560 at S2049 and-1,024 at S2111 due to the generated masking/zero path;
source normalization and RAW-BIT device tests independently retain the
rounding contract. Scalar moves and some address instructions grow too.
All opcode deltas are retained in the local evidence.

The per-PC opcode sum above reconciles with `sass__inst_executed_per_opcode`.
It is **not** the distinct `pu__inst_executed.sum` metric: that metric is
14,585,088 -> 12,882,688 (S2049) and14,629,504 -> 13,010,944 (S2111).
Do not mix their denominators or explain their difference as missing work.

## Evidence admission and provenance

- Retained36 cases x8 plus261 mixed-tail cases x8 passed the unchanged2%
  independent recurrence oracle and RAW-BIT residual comparison. Includes
  all63 tail extents, two gate strengths, absent/nonzero initial state,
  GVA expansion, output-only, input immutability and variable-gate stress.
- Four cells /60 kernels;11 outer manifest files and4x812 inner manifest
  files verified. Each capture has337 source files byte-bound to measured SHA.
- Same binary, input, device UUID and preflight/subject identity within every
  paired cell. FLA source hashes checked. All60 kernels report1.700GHz CE.
- All42 old native instruction/operand sequences and resource/ABI records
  match the previous same-box-compiler upload. New source/native/linked-ABI
  gate rerun locally against uploaded native and ELF data; no device run.
- Eighteen analysis negatives reject missing/duplicate cells, a missing
  kernel, old state body, changed device and wrong prefix/tail execution
  counts. The17 original source/native/linked negatives also reject.

Device PPU-ZW810,72 CU, UUID `019ee024-8860-091c-0000-0000007aff6d`.
Driver `2.1.2-r7b50d071022`; compiler HGCC2.2.0-dev (Jun3 2026).
`ppu-smi` separately reports SDK `2.0.2-d2568e060312`; preserve both identities
rather than conflating that label with the actual compiler.
Torch2.9.0/runtime12.9; FLA0.6.0/Triton3.4.0, native GVA head mapping,
backend override disabled, previously recorded process-local CUDA13 parser
backport. Capture tool is `/sim/eec/shared/junfu.qx/asight/bin/acu`.

Archive SHA256:
`a53790f37730ce5eedc47860a0d3ebca828da340d1e2e5f18e4a0de9f7c16968`.
Compact machine-readable record:
[`mixed_tail_acu_20260928.json`](../dev/ppu/results/mixed_tail_acu_20260928.json).
Full local raw imports, per-PC bindings, counter deltas and audit script:
`/workspace/actlizeLA-migration-20260928/ppu10-tuning/mixed-tail/acu-analysis-20260928`.
No uploaded code or device binary was executed during analysis.

Next migration item: actual inverse-intermediate register retention and
paired-conversion applicability. Mixed-tail prepare/output remain generic;
this result closes state only. No H800 restart or automatic PPU selection.
