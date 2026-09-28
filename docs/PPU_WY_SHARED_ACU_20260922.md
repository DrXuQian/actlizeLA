# Shared-row WY: verified ACU result and next optimization boundary

## Evidence and admission

Archive: `gdn-qsa-acu-20260922T225857Z-797428.tar.gz`.
SHA256: `d4909bd256324de69af1112d76628f72296d8cde42896acc37f4785845fbb692`.
All 553 regular files and the complete 552-entry checksum denominator verify.
No archived program or library was executed. The native reports were imported
locally with SDK ACU2.1.1/data15000 and a private compatible host loader; no
system runtime change or device run. The capture used ACU2.0.2/data15000.

The loaded WY binding/device library match the archived files and the preceding
comparison from clean `eed5ba29d48d38df43d79ee62757b7d11aaae749`:

- Binding: `8dbfd293eb9a6dd1b36b86d5d9b52d75b64519f65ab447b855387656f5f7c721`.
- Device library: `aa0060d49890771a7d1df9815e9600b14ab3061d6c76008e32da70acfdddf7bd`.
- Original run: `/workspace/gdn-wy-fla-eed5ba2-20260922T132811Z`.

Selected delivery is **prepare-rows-shared, mask1520**, not scalar or mask48.
Workload: B=1, S=2048, Hk=16, Hv=32, K=V=128, chunk64, native GVA,
BF16 inputs/output, FP32 final state, zero initial state, final-state output
enabled, no QK normalization, and scale `1/sqrt(128)`.
WY and FLA match on input/reference fingerprints, physical device UUID,
runtime, shape and gate. All reported CE frequencies are1.700 GHz. Although
the preceding comparison has no separate UUID field, its properties string
contains the same UUID as both capture receipts; identity is established for
this bundle, not merely inferred from an equal device model/CU count.

Sixteen preceding numerical cases,80 candidate admissions, raw equality to
scalar WY and8 repeats per admission verify. Both gates have all8 roles and
16 finite samples; medians and every saved candidate verdict recompute.
Preflight/subject output fingerprints match; actual subject calls=1, warmup=0.
This also resolves the pasted `row-warp` spelling: raw JSON has the correct
`wy-prepare-rows-warp` role.

## Full public-API result, kept separate from profiling

| g | Prepare-address496 us | Shared1520 us | Warp2544 us | FLA us | Shared vs496 |
|---|---:|---:|---:|---:|---|
| -0.1 | 358.918 | 354.666 | 357.136 | 494.544 | UNRESOLVED |
| -1.0 | 356.590 | 352.192 | 354.350 | 494.490 | CANDIDATE-WINS |

Weak shared range `[353.712,357.808]` overlaps496's
`[357.548,361.524]`; the lower median is not an admitted win under the fixed
criterion. Strong saves4.398 us/1.233%, with disjoint ranges. Both shared/FLA
comparisons have disjoint ranges (median speedups1.394x/1.404x). Shared/warp
remains UNRESOLVED at both gates. No global selector/default promotion.

Original final state remains BF16 versus WY/FLA FP32. Scalar WY remains a
retained older control, not the subject named shared above.

## Same-capture stage comparison: g=-1.0 only

| Matched mathematical phase | WY us | FLA us | WY / FLA | Excess us |
|---|---:|---:|---:|---:|
| Prepare: prefix + KKT/solve + W/U | 135.350 | 83.107 | 1.629x | 52.243 |
| State recurrence | 145.466 | 89.490 | 1.626x | 55.976 |
| Output | 66.476 | 43.385 | 1.532x | 23.091 |

FLA prepare comprises2.633 +44.285 +36.189 us. Its two independent fills
cost4.295 +2.205 us and are retained separately, not hidden in matched math.
Profiled sums are WY347.292 us and FLA222.482 us including fills. Those are
**ACU diagnostic kernel durations**, not replacements for352.192/494.490 us
complete-API event spans. API victory does not establish faster individual
GPU kernels. Conversely, the profiled sums do not overturn the API result.
Do not subtract these unlike protocols to claim a measured host overhead;
dispatch/idle/allocation attribution would require a separate timeline.

The matched-math excess is131.310 us: state42.6%, prepare39.8%, output17.6%.
There is no longer one stage explaining nearly all the difference.

## What the counters exclude

| Metric | WY state | FLA state | WY output | FLA output |
|---|---:|---:|---:|---:|
| CTAs / threads | 128 /128 | 128 /128 | 1024 /256 | 1024 /256 |
| Registers/thread | 242 | 256 | 98 | 170 |
| Dynamic shared B | 49,408 | 50,432 | 49,408 | 49,152 |
| Achieved active warps/CU | 7.09 | 7.09 | 37.60 | 22.70 |
| Achieved occupancy | 11.08% | 11.08% | 58.75% | 35.48% |
| BF16 MMA instructions | 524,288 | 524,288 | 524,288 | 524,288 |
| KVD global-store traffic | 98 MiB | 98 MiB | 16 MiB | 16 MiB |
| DRAM read bytes | 92,562,176 | 92,562,816 | 67,385,472 | 67,401,600 |

Thus insufficient grid/residency does not explain the **difference** between
these state kernels. Output has more active warps yet remains slower. Global
write amplification is already closed for state/output; their math work and
HBM reads agree. This does not prove occupancy or memory latency irrelevant
in absolute terms; it rejects them as sufficient explanations of this gap.

## Executed instructions, not static site counts

Native per-PC counts were exported using HgRules, independently checked
against the original raw counters and SDK `ExportMetricsByKernel`, and their
sums close against `sass__inst_executed_per_opcode`. Removing one executed
PC fails closure for every one of the 3 WY and 7 FLA kernels. The PU instruction
counter is a different counter; it is not forced to equal the per-PC sum.
The existing 24 host ACU contract tests also pass in the recorded task Python
environment; the default system Python has no Torch and cannot run that suite.
No new kernel build or device run was performed for this documentation change.

| Phase | WY executed instructions | FLA matched instructions | Ratio |
|---|---:|---:|---:|
| Prepare | 30,292,992 | 17,228,800 | 1.758x |
| State | 21,741,184 | 10,040,320 | 2.165x |
| Output | 21,560,320 | 9,166,848 | 2.352x |

Representative **state** excesses: `v.madl.i32` +3,092,480;
`v.shrl.b32` +1,057,280; `s.mov.b32` +1,001,984;
`v.shll.b32` +857,600; `v.lop3.b32` +785,920;
`v.mov.v2s` +756,736; `v.and.b32` +726,016. This is primarily extra
address/operand/control work, not extra BF16 matrix products. A causal
latency reduction from removing a particular subset still needs its own
single-change device comparison.

In particular, WY matrix reads are `tsm.ld.swzl` with scalar descriptors,
and the native trace contains vector-to-scalar base transfers around them.
FLA uses `tsm.ld.ncom[.mt1616]` with lane-address operands and no `v.mov.v2s`
in these stages. This is an observed lowering difference, **not permission
to substitute one opcode for the other without a layout/fragment proof**.

`gdn_wy_state_ab` still uses generic signed-coordinate `swizzle()` for H,
U and scaled-V fragment stores and `load()` for shared-to-MMA delivery.
`wy_mma.cuh::load` applies a general CuTe RowLayout and passes a generic
pointer to the swizzle matrix-load atom each time. The earlier address
ablation optimized global/shared staging and vector publication, not all
these per-fragment consumers/producers. These are concrete next inspection
sites. The whole11.7M instruction excess is **not yet attributed to one
helper**, and instruction count alone is not a latency prediction.

Stall-per-issue ratios also are not time shares. For example, state memory
dependency is0.854 (WY) versus1.252 (FLA), despite WY taking longer: a smaller
ratio per issued instruction does not compensate for twice as many issued
instructions. Output memory/commit dependency is2.139/0.962 versus1.860/0;
this supports inspecting its delivery cadence, not blindly increasing blocks.

## Prepare has two additional, separate costs

1. **W/U publication remains scalar.** The trace contains524,288
   `vmem.st.b16` executions. Prepare's KVD global-store traffic is512.25 MiB:
   W/U512 MiB plus row-prefix0.25 MiB. The useful W/U output is32 MiB, and
   FLA's W/U kernel reports32 MiB at that interface. This is a16x **KVD
   interface** amplification, not16x HBM bytes. The smaller vector publication
   should be tested on the current prepare winner, not assumed equivalent to
   the older slower tiled-prepare variant.
2. **Inverse precision is intentionally different.** BF16 MMA work matches
   (344,064 versus344,064), but WY executes98,304 TF32 MMAs versus32,768:
   high/high plus two residual products. Keep this accuracy contract; silently
   deleting residual terms would be a precision change, not address tuning.

Prepare already executes fewer exponent instructions than the matched FLA
phase (75,776 versus94,208); output likewise90,112 versus163,840. Neither
count supports another blanket claim that exponent count is the dominant gap.

## Next bounded change, not implemented by this analysis

1. **State shared-address/operand delivery first:** retain1520 and all old
   native controls. Precompute chunk-invariant addresses and use proven
   nonnegative/cube-aligned offsets for fragment stores and matrix-load bases.
   For16-aligned cube bases, the existing layout reduces to
   `row*16 + col*Rows` half-elements; arbitrary element addresses do not.
   Check the full coordinate set against actual RowLayout and MMA traits,
   then check generated code to establish which overhead really disappears.
2. **Prepare W/U vector publication next**, isolated from TF32 solve or
   arithmetic changes. Reuse dead scratch only after proving lifetimes.
3. **Output operand delivery** after the state mapping is proved. Keep the
   old output: the prior output-address variant did not establish a speed win.

Each candidate must keep BF16 boundaries, FP32 state, reduction order and
raw equality/repeated launches. Both gates are timed against their same-run
incumbents. No promised latency follows from converting an instruction count
to microseconds. If an alternate native matrix-load family is needed, it gets
its own layout proof and ablation, not an undocumented change to all helpers.

Reproducible host analysis and native imports:
`/workspace/gdn-wy-shared-acu-analysis-20260922/` (`analyze_bundle.py`,
`reimport_native.sh`, `analysis-verified.log`, `parsed.json`, both native
imports and per-PC tables). `bash reimport_native.sh` repeats only host
report imports and analysis; `python3 analyze_bundle.py` rechecks the saved
imports without running ACU. Neither command executes a device kernel.
Only reports/checkpoints changed in the repository; no kernel or route edits.
