# UPDATE lookahead ACU: real schedule change, no latency gain

## Verdict

**Numerics PASS; no observed speed gain; do not promote this candidate.**
The preregistered UPDATE-only operand experiment changed the actual native
schedule but not state latency or whole-kernel readiness. Keep it as a
counterfactual; default routing and the retained non-B eight-warp incumbent
remain unchanged. This single capture establishes neither a repeatable win
nor a statistically stable regression. The 1.5x FLA target is not met.

Source `0755b9399ee18413e66fcfa6857648851144580d`; same binary, fixture and
physical PPU for control `residual-warps8-blayout`, candidate
`residual-warps8-operands`, and FLA. Shape B1/S2048/Hk16/Hv32/K=V128/C64,
V32, g=-1, zero initial state, BF16 inputs and FP32 final state. All15
captured kernels report1.700GHz on PPU-ZW810/72CU. Weak-gate performance
was not captured; both gates were numerically admitted.

## Complete ACU accounting

Microseconds; one full site-ACU capture per arm, **not API event latency**.

| Phase | B-layout control | UPDATE lookahead |
|---|---:|---:|
| Prefix | 2.38941 | 2.59529 |
| Solve | 52.38412 | 52.49294 |
| Residual state | 104.52765 | 104.36118 |
| Output | 45.53412 | 46.72647 |
| **All4 kernels** | **204.83530** | **206.17588** |

State changes by-0.16647us (-0.159%). Total changes by+1.34058us (+0.654%);
the **unchanged** prefix/solve/output contribute+1.50705us, including output
+1.19235us. Their native opcode histograms are identical. Do not attribute
that variation to the edited state loop or describe it as proved regression.

FLA all7 kernels total224.98530us: cumsum2.57765, BF16 fill4.32647,
KKT solve44.45765, W/U35.65706, FP32 fill2.24647, state91.25706,
output44.46294. Including both fills, the candidate is1.09123x FLA.
The 1.5x goal requires149.99020us: another56.18568us reduction is needed.
Our residual state includes P@R and replaces a separate W/U kernel; compare
104.36118us with FLA **W/U+state126.91412us** when grouping equivalent
mathematical stages. State91.25706 alone is not the same work.

## The intended edit reached the device binary

The uploaded native image independently passes the existing source/native
gates, including the negative which substitutes the exact old body with
unchanged useful work. That negative fails specifically for no improved
lookahead, not a missing MMA or an arbitrary instruction count.

| Native/static property | Control | Candidate |
|---|---:|---:|
| Registers/thread; stack bytes | 124;0 | 124;0 |
| Grid CTAs; threads; shared bytes | 128;256;45,568 | same |
| Whole-body instruction sites | 1,313 | 1,315 |
| Recurrence-backedge sites | 516/530 | 510/524 |
| UPDATE immediate MMA-input reuse by next load | 3 | 0 |
| Other immediate reuse sites (KH/PR) | 7/3 | 7/3 |
| UPDATE B-load lead, intervening MMAs | 0,1,0,1,0,1,0,1 | 0,1,2,3,2,3,2,3 |

The improved lead is a native scheduling fact, not a guarantee that all
MMA reads have retired before later loads. Removing immediately adjacent
reuse does not eliminate later reuse or hazards elsewhere. Native dynamic
opcode sum falls15,885,440->15,697,024 (-1.186%); useful MMA is unchanged.
Do not conflate this sum with `pu__inst_executed.sum`, which reports
16,442,496->16,254,080 with a different counter scope.

## Resource, traffic and dependency account

| State metric | Control | Candidate |
|---|---:|---:|
| Active warps/CU | 14.14 | 14.12 |
| Eligible warps/WE | 0.35 | 0.34 |
| Read BC | 2,883,584 | 2,883,584 |
| Write BC | 2,998,272 | 2,998,272 |
| Total BC | 5,881,856 | 5,881,856 |
| KVD-to-TSM bytes | 117,440,512 | 117,440,512 |
| Ordinary global KVD read/write bytes | 17,825,792 /102,760,448 | same |
| Matrix-load instructions | 1,179,648 | 1,179,648 |
| BF16 MMA /AIU /BF16 conversions | 655,360 /16,384 /1,311,744 | same |
| Memory dependency, per issue | 1.72 | 1.71 |
| TSM dependency, per issue | 1.34 | 1.38 |
| Compute dependency, per issue | 0.79 | 0.80 |
| Compute TFU WAR, per issue | 0.39 | 0.40 |
| Sync, per issue | 1.14 | 1.14 |
| Instruction fetch, per issue | 0.29 | 0.24 |

These ratios are **not wall-time shares**. Fewer issued instructions and
lower fetch ratio did not produce a latency gain. Resource limits remain
register4/shared5/warp8/architectural64 CTAs/CU; the grid supplies only
128/72=1.7778 CTAs/CU (~14.22 warps/CU). Capacity, supplied warps and eligible
warps are separate limits; none improved in this experiment.

Per-PC data are bound to all36 native matrix-load sites and their exact
register operands, partitioned using the real gate's KH/PR/UPDATE mapping:

| Matrix-load group | Sites | Executions, both | Compute-dependency samples, control→new |
|---|---:|---:|---:|
| KH | 16 | 524,288 | 886→884 |
| PR | 8 | 262,144 | 350→366 |
| UPDATE | 12 | 393,216 | 444→379 |
| Whole kernel, all opcodes | — | — | 3,565→3,620 |

The targeted load-PC samples fall65 while the full kernel's compute samples
rise55. This is consistent with a small local effect that does not alter the
whole call, **not** a measured65-cycle saving or proof of cancellation. The
public per-PC export does not separate TFU WAR from other compute waits.
Memory samples fall7,813->7,495 and sync5,751->5,468; these too are sampling
counts, not additive execution-time buckets. Near-zero movement in elapsed
state time remains the deciding observation.

Thus the hypothesis “this UPDATE-only lookahead fixes the observed timing
loss” is not supported. The broader possibility of a better KH/PR/H/K
delivery schedule is not disproved: ten immediate reuse sites remain in
those other phases and their source loops were deliberately unchanged.

## Next decision, not another unmeasured promotion

Do not deepen UPDATE register arrays again: two/four slots already produced
the same native scheduling in the bounded compile screen, and the effective
two-slot edit now has no measured speed benefit. Keep the unchanged control
for subsequent paired experiments.

The next distinct layout target is **H/snapshot**, covering its C-fragment
writer, transposed MMA reader AND canonical16B global publisher together.
H contributes262,144 transposed reads and524,288 BF16 stores at this geometry;
these are work counts, not an independent attribution of all remaining BC.
No scalarized snapshot publication, duplicate H copy, or hidden layout repair
may be described as free. Check emitted work/resources as well as the maps.

K already uses AIU.swzl/matching ld.swzl with both K@H and Kt@scaledV:
262,144 normal and262,144 transposed loads. A simple shared transpose only
swaps which reader needs trans. A useful change must prove both consumers
and explicitly price a second view if one is required. BC and register
counts remain diagnostics; full ACU kernel sum decides whether it helps.

This turn only analyzes/records the uploaded experiment; no new kernel,
scheduler, numerical tolerance, runner or default selector is changed.

## Evidence and replay

Upload SHA256:
`564de266956ef11c6cd548ccd26493666a9c81d016058a24f73bc08c0a32c354`.
620 unique regular files;619 checksums;144 source snapshots byte-equal to
the captured SHA. Tracked source/build inputs are clean; two untracked
operator report files are listed separately, not mistaken for build inputs.
All binary/fixture/device receipts agree, complete4/4/7 kernel inventories
and15/15 native opcode sums close. An omitted executed PC fails its sum.
The report omits exactly16 trailing native alignment NOPs per state body;
every actual matrix/MMA site and operand is checked, none silently dropped.

30 scalar +30 B-layout +30 UPDATE-lookahead cases each pass8 repeats,
GVA/tail/nonzero-state/output-only, independent2% recurrence and RAW-BIT
delivery equality. Both g=-0.1 and-1 numerical admissions pass. The old16
WY cases/32 delivery admissions pass too. Six source/five native negatives
remain red, including the old-schedule-same-work counterfactual.

Binding SHA256:
`d1b14a0168e924d6e3e17bdc95e645cf607ff24c1ea037fe7c45a8e0fadd3c16`.
DSO SHA256:
`8104bbcdccfcb3930fab9d2ae3ea741b2639a161934e87227e7dd4fa3eeded53`.
Physical UUID `019ee024-8860-091c-0000-0000007aff6d`;
SDK2.1.1-a5c56e; FLA0.6.0/Triton3.4.0 with process-local CUDA13 parser
compatibility. Capture tool is exactly
`/sim/eec/shared/junfu.qx/asight/bin/acu`.

[Condensed machine-readable evidence](../dev/ppu/results/residual_warps8_operands_acu_20260925.json)
retains every kernel time, raw metric names, phase-bound load samples,
opcode histograms and identities. Immutable archive and host-only replay:
`/workspace/gdn-warps8-operands-acu-analysis-20260925`:
`analyze.py`, `reimport.sh`, `export_stalls.py`, `import_stalls.sh`,
`summarize.py`. Final replay uses `analyze.py --native`; no uploaded source,
script or library was executed and no PPU job was started here.
