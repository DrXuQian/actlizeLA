# SM90 GDN: multi-workload admission

Status: **56 attempts complete; the expanded performance target is not met**. The H800
remains on. A primary-shape win is not permission to declare all shapes faster.
No production routing, legacy SM80 code, clock or power settings were changed.
Native PPU1.7 remains **SKIP: SDK/model unavailable**; these are H800 controls.

## Registered coverage

The inventory was registered before its measurements, following the user's
request not to rely on a single test case. It is fourteen explicit workloads,
not a full Cartesian sweep. Every row has both natural-log gate regimes
`g=-0.1` and `g=-1.0`, and both pinned reference libraries, for **56 captures**.
K/V dimensions128, chunk64, BF16 Q/K/V/beta and FP32 final state stay fixed.

| Workload axis | Cases |
|---|---|
| Sequence length, B1/Hq16/Hv32 | 512,1024,2048,4096,8192 |
| Incomplete final chunk | T2051 |
| Independent batch sequences, T2048/Hq16/Hv32 | B2,B4 |
| Value/query head mapping, B1/T2048 | 16/8,32/32,64/16,64/32 |
| Gate format/pattern | token/head-distinct FP32 gate, zero initial state |
| Initial state | same varying gate with nonzero, nonsymmetric FP32 state |

The last two use B1/T2048/Hq16/Hv32. The nominal fixture is preserved byte for
byte. Batch sequences are never concatenated into one recurrent history.
FlashInfer's packed-token/VK-state API gets explicit CPU-side adapters;
FlashQLA uses fixed-batch/KV-state inputs. Cross-library input, initial-state
and independent CPU-reference hashes must match per scenario.

Authority: experiment branch `sm90-multishape-20260926`,
`tools/sm90_workloads.py`. Every capture uses12 interleaved/reversed complete
forward calls and sums all GPU kernels from nsys, including reference helper
kernels. Both auto and no-CP routes are measured; each scenario uses its own
fastest reference. The original2% numerical criterion,8 repeated bit-stable
launches, candidate/parent raw equality, per-capture output checks and disjoint
observed timing envelopes are unchanged. Overlap remains UNRESOLVED.

## Counterexamples already established

Final frozen inventory: **55 valid captures / 1 reference numerical failure /
0 pending**. All3,636 complete forwards were re-extracted exactly from SQLite;
local and remote Python3.12 results are byte-identical. Against each library's
fastest path, S38 has **16 wins / 10 losses / 2 unresolved** versus FlashInfer,
and **25 wins / 2 losses / 1 invalid numerical reference** versus FlashQLA.
No aggregate speedup is used to hide losses.

All values below are paired medians in microseconds. Each entry is
S38/reference; W/L/U means win/loss/overlapping ranges. Weak/strong correspond
to the registered -.1/-1 gate parameters, including the varying-gate fixtures.

| Workload | S38 / FI, weak | S38 / FI, strong | S38 / QLA, weak | S38 / QLA, strong |
|---|---:|---:|---:|---:|
| seq2048 | 101.50 / 112.77 W | 101.39 / 112.61 W | 99.54 / 165.01 W | 100.15 / 164.56 W |
| batch2 | 194.39 / 114.29 L | 196.72 / 116.43 L | 194.69 / 218.67 W | 196.88 / 221.09 W |
| seq512 | 30.29 / 31.81 W | 30.27 / 31.87 W | 30.29 / 54.74 W | 30.42 / 54.86 W |
| seq1024 | 53.92 / 58.70 W | 53.55 / 58.50 W | 53.30 / 96.61 W | 53.52 / 96.98 W |
| seq4096 | 187.02 / 209.44 W | 186.99 / 207.81 W | 185.81 / 241.18 W | 186.54 / 238.85 W |
| seq8192 | 358.75 / 353.43 L | 361.09 / 353.11 L | 362.37 / 383.38 W | 366.24 / 377.48 W |
| tail2051 | 101.55 / 113.06 W | 101.65 / 112.83 W | 101.62 / 181.04 W | 101.38 / 174.47 W |
| batch4 | 288.79 / 227.33 L | 289.43 / 228.29 L | 289.04 / 325.52 W | 290.71 / 327.97 W |
| heads64-gva4 | 195.03 / 114.69 L | 197.22 / 116.14 L | 196.55 / 227.04 W | 196.06 / 225.71 W |
| heads64-gva2 | 196.80 / 115.23 L | 194.93 / 114.96 L | 197.71 / 228.70 W | 196.42 / 228.32 W |
| heads32-gva1 | 102.82 / 114.26 W | 102.40 / 113.65 W | NUMERIC FAIL | 99.86 / 172.69 W |
| heads16 | 93.97 / 94.46 U | 93.79 / 94.02 U | 94.11 / 91.82 L | 94.31 / 92.32 L |
| vary-fp32 | 102.02 / 113.01 W | 102.58 / 113.86 W | 99.19 / 168.98 W | 99.25 / 164.59 W |
| initial-vary | 105.25 / 114.37 W | 105.20 / 114.80 W | 101.04 / 169.86 W | 101.25 / 165.25 W |

The tested candidate is S38 (V128 split into two independent V64 CTAs,
aux232/state192). Control is the immutable unsplit S24. Representative
same-capture weak-gate medians in microseconds:

| Workload | S24 | S38 | Fastest FlashInfer | Verdict for S38 |
|---|---:|---:|---:|---|
| B1/T2048/Hq16/Hv32 | 120.513 | 101.505 | 112.769 | wins |
| B1/T8192/Hq16/Hv32 | 447.938 | 358.753 | 353.425 | loses |
| B2/T2048/Hq16/Hv32 | 125.089 | 194.386 | 114.289 | loses |
| B4/T2048/Hq16/Hv32 | 249.409 | 288.786 | 227.329 | loses |
| B1/T2048/Hq16/Hv64 | 125.185 | 195.026 | 114.689 | loses |
| B1/T2048/Hq32/Hv64 | 126.304 | 196.801 | 115.232 | loses |

The same losses also occur at the strong gate in the completed cells. Do not
infer a universal selector from the first row. V64 doubles the physical grid
and duplicates QK/KK/inverse work. At64 batch-heads it creates128 CTAs on114
SMs while resources still allow only one such resident CTA per SM. The
unsplit control is substantially faster there, but still loses FlashInfer.

T8192 is a different gap: the fastest FlashInfer route is CP. Its pinned
`gdn_prefill.py`/`delta_rule_dsl/varlen_helper.py` chooses CP for insufficient
parallel batch-heads, then bounds preprocessing chunks against the SM count.
Calling these two gaps the same scheduler bottleneck would miss that change
of algorithm. No averaging across rows may hide either counterexample.

## Evidence repair, not a timing-rule change

A later, independent failure remains visible: on B1/T2048/Hq32/Hv32,
g=-0.1, FlashQLA auto has output/state relative errors
0.021621605/0.009716575. It fails the registered2% gate before timing. Our
same-input S24/S38 pass. This is a reference admission failure, not evidence
of a regression in our kernel and not permission to loosen the criterion.
The paired strong-gate capture passed. Isolated auto/no-CP diagnosis has now
reproduced identical maximum errors and worst output location in both
reference paths, with8 stable repeats each. The no-CP result also fails:
CP is not necessary for this maximum-error violation. Our two arms are
raw-identical and have errors0.005405401/0.003333338 on those exact inputs.
No performance is claimed for the failing reference, no tolerance is changed,
and this reference-only failure does not stop independent candidate checks.

At the34th original capture, first-JIT FlashQLA passed all five arms' numerics
but failed before timing: the old identity collector required a mapped
`executable.so`. Fresh TVM executables can instead run live LLVM/CUDA modules.
Three previous partially fresh-JIT captures also had incomplete image sets.
All four attempts and the original matrix are preserved, not overwritten.

The repaired collector serializes each actual live CUDA module, validates
its complete CUDA ELF extent and hashes it. The pinned TVM JSON exporter
cannot describe TMA descriptor dtype30, so binary serialization is used; no
reference recompilation or disk-cache guess substitutes for the live image.
Real fresh-JIT and cache-hit runs of the failing workload each bind all four
CUDA images; their four hashes are identical. Both processes also pass all
five arms' CPU oracles and8 repeats. No timings from that proof are ranked.

Two source epochs are stored. An AST proof permits changes only inside
`reference_binaries`; input, numeric, capture and timing changes fail the
negative test. Retried cells use new `-identity-r2` directories. Negative
controls reject a missing module, wrong ELF, truncated extent, omitted
workload, failed capture or a performance loss being counted as goal closure.

`tools/record_sm90_workloads.py` re-extracts every completed SQLite, validates
images/epochs/device/input hashes and reports unfinished cells explicitly.
Run it with Python3.12 like the capture host: Python3.10's naive float `sum`
changes last bits of descriptive per-symbol totals. The integer-nanosecond
forward sums and verdicts are unaffected; no tolerance was added to disguise
that reanalysis mismatch. Python3.12 local re-extraction is exact.

Artifacts: `/workspace/gdn-sm90-multishape-20260926/{captures,jit-fresh-cached-proof-r2}`.
Full-inventory machine-readable result: final-matrix.json in that task root.
All279 recorded reference image entries (53 unique images) were hash-checked
and copied to reference-images/, including the older mapped-cache images.
The full archive is being sealed. S39/S40/S41 are separately registered
followups, not changes to the frozen S24/S38 comparison. They now pass14 CPU
cases with parent fingerprints and two raw stress pairs each; paired graph
screening is underway, and is not nsys performance admission.
