# Metadata lookahead: real overlap, no observed speed gain

**Keep HV as the preferred experimental control.** This candidate passed
numerics and emitted the intended prefetch, but the complete ACU kernel sum
rose **187.00764 -> 189.57529us (+1.37%)**. Changed state rose
**89.88176 -> 91.94353us (+2.29%)**. FLA was223.44057us including both fills.
This is one capture per arm, not a repeated estimate of a universal slowdown.
Default routing is unchanged; the candidate remains an opt-in counterfactual.

## Admission and identity

- Workload B1/S2048/Hk16/Hv32/K128/V128/C64, g=-1;72CU PPU-ZW810,
  UUID`019ee024-8860-091c-0000-0000007aff6d`. Every captured kernel1.700GHz.
  Source`6a102fd11e0ff656bd0128123e8762d27b88a99b`, implementation`b33686a`.
- Archive SHA256`762112d7ce83fd0bb4de02e41751204215eb3521dae54e6cfe2636bced94e1f9`;
  **632regular files/631manifest hashes/156source snapshots** all verified.
  Source diffs are empty. Two untracked loose ACU reports are explicitly
  recorded; this is tracked-source-clean, not an empty worktree.
- DSO SHA256`d1e36023a89744b4cd5406b2e7f7db99effeab17ff11c32daa73af52d14171ea`;
  binding SHA256`c424997d7dc21bd6b814bd3d2d5bacee0e152afa1c3a99c74b15e0bedded7eb2`.
  Comparison manifest, loaded-library hashes and subject/preflight receipts
  agree. Same fixture/device/binary across current control and candidate.
- 30scalar+30HV+30metadata device cases, each8repeats, RAW-BIT against scalar
  residual and the independent2% recurrence gate pass. Tail, GVA, nonzero
  state, varying metadata and output-only are covered. Weak/strong numerical
  admission is not a weak-gate performance result: **only g=-1 was profiled**.
  The old WY16-case/32-delivery-admission regression also passes.
- HGGC2.1.1-a5c56e, driver2.1.2-r7b50d071022/runtime13.0 (before/after agree);
  actual capture tool
  `/sim/eec/shared/junfu.qx/asight/bin/acu`, version
  `v2.0.0_20251231-4f7cd70/data12006`, binary SHA256
  `7d750971b45b0bb1f4367df6b8b0f37656d1be89894d40b39ceb2010f865b2b6`.
  FLA0.6.0/Triton3.4.0 sources and process-local CUDA13 parser patch are bound
  in the JSON; alternate FLA backend dispatch is disabled.

No uploaded source/binary was executed locally. Actual reports were reimported
with the compatible local SDK host parser; this does not replace the recorded
site-ACU capture identity. No device run or new capture request was made.

## All-kernel timing, not public-API event spans

| Stage, us | HV control | Metadata | FLA |
|---|---:|---:|---:|
| Prefix/cumsum | 2.40000 | 2.42000 | 2.72882 |
| BF16 fill | — | — | 4.33176 |
| Solve | 52.90647 | 52.51588 | 44.17176 |
| W/U | in residual state | in residual state | 35.97412 |
| FP32 fill | — | — | 2.25647 |
| State | 89.88176 | 91.94353 | 89.26588 |
| Output | 41.81941 | 42.69588 | 44.71176 |
| **Complete call** | **187.00764 (4)** | **189.57529 (4)** | **223.44057 (7)** |

Only state changed. Prefix/solve/output are the same kernel bodies and have
identical dynamic opcode histograms; their combined+0.50588us is observed
unchanged-stage variation, not a source regression attributed to metadata.
Indirect cache/scheduling effects are not isolated by unchanged source alone.
Residual state incorporates the W/U algebra, so its FLA work-grouping
counterpart is W/U+state=125.24000us, not FLA state alone.

Retained HV is **1.19482xFLA**, candidate1.17864x. The1.5x target is148.96038us;
HV still needs **38.04726us** removed. Do not combine this capture's timing
with earlier API/event or historical ACU numbers to claim a larger gain.

## Prefetch actually happened

The uploaded native CFG passes the same compile-time schedule gate: each
recurring gate/beta load precedes eight UPDATE MMAs and zero KH/PR MMAs before
its next vector-load wait. The final-chunk guard remains source/host-proved;
the CFG alone is not a proof of native predicate equivalence.

Nine source/binding and seven native negatives still fail, including the old
late-load body and serialized-wait plants preserving the complete opcode/
operand multiset **and** CFG. All15per-kernel opcode sums match ACU's built-in
total; dropping one executed PC fails the sum. State/output report PCs match
the uploaded ISA, including MMA, matrix loads, metadata loads, AIU and waits.

| State native work/resource | HV | Metadata |
|---|---:|---:|
| Static instruction sites | 1194 | 1309 |
| Dynamic per-PC instruction sum | 15,350,912 | 16,791,168 |
| Scalar `s.*` executions | 3,438,592 | 4,430,080 |
| BF16 MMA | 655,360 | same |
| BF16 conversions | 1,311,744 | same |
| AIU copies | 16,384 | same |
| Matrix loads, normal/transposed | 917,504/262,144 | same |
| Registers/thread; stack | 122;0 | 120;0 |
| Shared bytes; grid; block | 45568;128;256 | same |

The **+1,440,256instructions (+9.38%)** are not extra matrix work. Largest
positive deltas: `v.mov.b32`+209920, `s.wait`+184064, `s.add.i32`+183040,
`s.mull.i32`+142592, `v.add.co/ci.u32`+129024each, `s.lop.emsk`+98304,
`s.mov.b32`+73728 and `s.mulw.u64.u32`+71936. The generic future-address and
validity calculation has a real recurring control/address cost. These counts
do not isolate how many microseconds each opcode added. The separate
`pu__inst_executed.sum` has a different counting scope and is retained, not
substituted for the verified per-PC denominator.

## The intended wait shrank; the critical path did not

These are **PC sample counts, not elapsed cycles or additive time buckets**:

| State waiting evidence | HV | Metadata |
|---|---:|---:|
| Metadata/global `vldcnt` memory-dependency samples | 1007 | 38 |
| AIU `commit_group(0)` memory-dependency samples | 196 | 625 |
| Sync samples just after INPUTS_READY | 3735 | 4094 |
| All memory-dependency samples | 6378 | 5693 |
| All sync samples | 4264 | 4650 |
| All instruction-fetch samples | 809 | 1196 |

Control gate/beta waits are PCs`0x5480dcc8`/`0x5480dd00` (540/467samples).
Candidate's metadata wait is`0x5480daf0` (38samples); its AIU completion wait
is now`0x5480dcc8` (625samples). **The same absolute PC denotes a different
instruction after recompilation**; cross-arm address equality is not semantic
site identity. Native/opcode binding is required.

The old metadata loads and waits lay between current AIU issue and its
completion wait. Moving metadata earlier changes that overlap window too.
The rise at AIU completion and INPUTS_READY is consistent with exposing
other waiting, alongside added address/control cost. This does **not** prove
an exact cycle-for-cycle wait transfer or that one counter explains the whole
2.06us. Fewer wait samples and fewer registers are not performance admission.

BC is identical: total4,440,064, read3,014,656, write1,425,408. KVD-to-TSM
remains112MiB, ordinary global-load transactions17MiB, KVD/L2 global stores
50MiB and L2-to-LLC stores50MiB. This was not another bank-layout or traffic
reduction. Output's corresponding work and layouts are unchanged too.
Scalar/KSD global-load transactions do increase91,136->95,232B; "unchanged
traffic" is scoped to the matrix/metadata vector path, not every interface.
DRAM reads33,957,760->33,958,784B and writes10,140,160->10,012,288B are
retained as replay/cache observations, not removed logical data.

Resource capacity is unchanged (register/shared/warp limits4/5/8blocks),
and grid supply stays128*8/72=14.222warps/CU. Achieved warps14.16->14.15;
eligible warps/WE0.41->0.43. The register reduction did not increase residency.
Per-issue memory-dependency1.48->1.20 coexists with sync0.93->0.98 and
commit-dependency0.05->0.12. These ratios are not wall-time percentages.

## Decision and next isolated axis

The preregistered complete-call criterion says **no observed speed gain**.
Retain HV; do not select this metadata implementation by its smaller global
wait counter. Keep the candidate for attribution, with default routing unchanged.
No universal rejection of prefetch follows from one implementation/capture.

Next planned experiment is solve register indexing on HV, not another combined
state/layout change: inspect/replace the native indirect register-slot loop
with compile-time indexing. Preserve the three-product TF32 precision scheme,
MMA/order and original numerical gate. Solve is52.91us versus FLA44.17us;
closing just that gap cannot by itself establish1.5x. A cheaper metadata
pointer-advance variant is a separate possible follow-up, not part of this
verdict and not implemented here.

Evidence: `dev/ppu/results/residual_metadata_acu_20260925.json`.
Local reproducible parsing: `/workspace/gdn-metadata-acu-analysis-20260925/`
contains the preserved archive, `analyze.py`, `summarize.py`, native negative
log and imported PC/stall tables. Single capture/arm; no claim of continuous
device isolation or repeat-distribution speed admission.
