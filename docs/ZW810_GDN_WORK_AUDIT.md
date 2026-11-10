# ZW810 GDN: account for work before parallelism

Scope: the eight round-2 cells in [the KDA contract](ZW810_GDN_FLA_KDA.md),
source `38e8e91`, B1/Hk16/Hv32 or48, T2048 or8192, K=V128, C64,
both natural-log gates, asymmetric nonzero initial state and final state.
All counts below are audited from archive
`9961c2362ed3485d52ef52020ae0b58fcebfbcfd0aa9bc64a5964aa65a55d419`.
They do not describe all model shapes or another PPU generation.

User priority: first eliminate unjustified mathematical/repeated execution
work; only then investigate warp/grid/resource parallelism. This is a
diagnostic order, not a claim that minimizing instructions always minimizes
latency. No kernel or default selection was changed by this audit.

## 1. Measured executed instructions: approximately twice FLA

`dev/ppu/zw810_fla_kda/analyze_work.py` revalidates the eight-cell denominator,
three-role capture cycles, numerical/capture bindings and exact CSV inventory.
It requires exactly one **Instruction Statistics / Executed Instructions / inst**
row per launch. This is NOT `sass__inst_executed_per_opcode`; don't subtract
one metric from the other or divide it by an MMA count from another metric.
All repeats and both FLA fill kernels are counted, not averaged away as one
fill. Counters in this table are identical for the two g values.

| T/Hv | Solve ours / FLA | State-equivalent ours / FLA | Output ours / FLA | Complete call ours / FLA |
|---|---:|---:|---:|---:|
| 2048/32 | 14,595,072 / 9,546,752 (1.529x) | 40,218,752 / 18,514,432 (2.172x) | 19,683,328 / 8,462,336 (2.326x) | 74,875,008 / 37,192,704 (2.013x) |
| 2048/48 | 19,548,672 / 14,194,176 (1.377x) | 60,328,128 / 26,702,592 (2.259x) | 29,524,992 / 12,840,960 (2.299x) | 109,968,576 / 54,847,488 (2.005x) |
| 8192/32 | 52,129,792 / 38,187,008 (1.365x) | 158,859,392 / 72,778,240 (2.183x) | 78,733,312 / 33,849,344 (2.326x) | 291,233,920 / 147,336,192 (1.977x) |
| 8192/48 | 78,194,688 / 56,776,704 (1.377x) | 238,289,088 / 105,186,048 (2.265x) | 118,099,968 / 51,363,840 (2.299x) | 436,850,880 / 217,532,928 (2.008x) |

Here ours is the declared single-TF32 candidate; state/output are identical
to the public default. FLA state-equivalent is **W/U + recurrent state**, not
its recurrent kernel alone. Example at2048/32: FLA7,438,336 W/U instructions
plus11,076,096 state instructions. Comparing only state would overstate our
excess. The complete-call column also includes prefix and fills.

Normalize further by `B * Hv * (T/64)` chunk-heads. For2048/32, this is1024:
ours/FLA solve14253/9323, state39276.125/18080.5, output19222/8264
instructions per chunk-head. State includes initial/final handling once per
sequence, hence fractional normalized values and a slight T dependence.
The work JSON retains raw per-kernel/per-repeat counts and normalized values.

## 2. Mathematical matrix-work ledger: not twice as much GEMM

Let `G` be chunk-local log-decay prefix, `D=diag(exp(G))`,
`B=diag(beta)`, `L=tril(B * gated(K K^T), -1)`, and `P=(I+L)^-1`.
For one incoming state H, the real-valued identities are:

- Residual: `R=B(V-D K H)`, `Vnew=P R`.
- FLA W/U: `W=P B D K`, `U=P B V`, `Vnew=U-W H`.
- Both then update `H'=exp(G_last)H + K^T diag(exp(G_last-G)) Vnew`.
- Both output `scale * (D Q H + causal(gated(Q K^T)) Vnew)`.

This is algebra, **not BF16 bit equivalence**: the two paths round at different
boundaries. Never delete a conversion or move scale/gating across a rounding
boundary under a delivery-only label.

The following counts are **source-model dense MACs per complete chunk and
value head**. They include zeros inside matrix tiles and exclude elementwise,
diagonal solve, conversion, address, copy and synchronization work. Multiply
MAC by2 for FLOP; don't present these as measured native opcode counts.

| Matrix work | Ours | FLA | Conclusion |
|---|---:|---:|---|
| KKT lower ten16x16 blocks, reduction128 | 327,680 | 327,680 | Same ten blocks; upper six skipped by both. |
| Inverse off-diagonal merges | 16 products of16x16x16 | 16 products of16x16x16 | Same block-product count, different association/ownership. |
| State-equivalent | KH1,048,576 + PR524,288 + update1,048,576 = **2,621,440** | W524,288 + U524,288 + WH1,048,576 + update1,048,576 = **3,145,728** | Residual uses **16.7% less** matrix arithmetic overall, but one extra GEMM on the recurrent chain. |
| Output | QK524,288 + QH1,048,576 + AV524,288 = **2,097,152** | **2,097,152** at the captured BV128 geometry | Same logical matrix work. |

The retained solve's TF32x3 vs single-TF32 was a real excess of compensation
products: per chunk the32 equivalent m16n16k8 calls become96. Round2 removes
that precision difference experimentally, yet complete-call instruction count
falls only about1.1%. Remaining state/output instruction excess is not evidence
of twice the mathematical GEMM work.

Source bindings:

- `csrc/gdn_chunk/gdn_wy_solve_tf32x1_ppu.cu`: ten BF16 KKT blocks; gap1/2/3
  have3x2,2x3,1x4 off-diagonal block products =16.
- `csrc/gdn_chunk/gdn_wy_residual_full_chunk_ppu.cu`: KH, PR, K-transpose update.
- `csrc/gdn_chunk/gdn_wy_residual_warps8_hvlayout_ppu.cu`: output QK precedes
  the two-panel loop, then QH and AV cover disjoint V columns.
- Captured, hash-bound FLA `ops/gated_delta_rule/chunk_fwd.py:120–291`,
  `wy_fast.py:79–106`, `ops/common/chunk_delta_h.py:211–302`,
  `ops/common/chunk_o.py:82–155`. Use those snapshot contents, not moving main.
  The captured output grid `(T/64,Hv,1)` with B1 and V128 plus its source
  flattening establishes one V tile (BV128), not guessed from autotune options.

## 3. Repetitions proved in source, and attribution still missing

| Check | Finding | What it does / does not establish |
|---|---|---|
| State V32 slicing | Four CTAs own disjoint V columns. KH/PR/update outputs do not overlap. K and P input planes are staged independently by each CTA. | Repeated operand delivery, not four copies of the same GEMM result. Changing V slices changes work supply too; defer that axis. |
| Output V64 panels | QK is outside the panel loop and computed once. QH and AV panels cover different columns. | Rules out the proposed repeated-QK explanation for our2.3x instruction count. |
| Output Q reuse | Our QK phase reads shared Q, then each QH panel reads shared Q again. FLA's K loop uses a loaded `b_q` for both QH and QK. | A concrete reuse seam. Native counts are needed to price it; `tl.load` source is not proof of a free register layout conversion. |
| Residual intermediates | BF16 H snapshot, residual, Vnew and gated Vnew are written/read through explicit shared publication layouts. | Preserve distinct rounding and global consumers. Register retention must prove mapping and account for shuffles/conversions, not assume free removal. |
| Diagonal solve | Our lanes16–31 repeat lanes0–15's column arithmetic (`lane%16`); only lanes<16 store. FLA uses an in-register row formulation. | A source-level duplicate, not a promise of halving warp instruction issue by masking lanes. FLA's dense row formulation also has zero arithmetic; compare actual executed code before claiming a cheaper diagonal algorithm. |
| Head grouping | Raw KKT/QK share inputs across Hv/Hk heads, but beta/gates/P differ. Both implementations form results per value head. | Common possible reuse, not an ours-only explanation. Sharing ungated products needs a separately priced materialization contract. |

Do **not** label all extra moves as SWZL repair or all extra instructions as
unsupported-PTX emulation. Bind per-PC counts to source/native sequences first.
Likewise `expf` versus FLA `exp2` is an explicit arithmetic/lowering difference,
not a proven share of the current gap. This round does not change fast math.

Next ledger categories: necessary MMA; diagonal/elementwise arithmetic;
FP32/BF16/TF32 conversion; shared/global loads and stores; address/control;
barriers/waits. Keep PC warp executions and profiler aggregate metrics separate.
Only then choose an isolated removal/reuse candidate, before a parallelism sweep.

## 4. Extract the missing instruction detail without another device run

The uploaded raw CSV contains aggregate counts, not enough per-opcode/PC
information to allocate the full excess. Read the original `.acurep` files
on box and require actual native PC/count coverage. After pulling this branch:

```bash
python tools/export_gdn_kda_details.py \
  --run /workspace/actlize-la-gdn-solve-tf32x1-20261109T224926Z-2748938 \
  --acu /sim/eec/shared/junfu.qx/asight/bin/acu
```

No compilation, device query, warmup or GPU launch. The script invokes the
**same captured ACU CLI with `--import`**, verifies the original report hash,
and exports raw/details metrics plus an embedded `--stats-rule` native table
for all kernels in repeat0 of four geometries
at g=-0.1 for default/candidate/FLA (12 reports). Fixed repeat0 is diagnostic
selection, not fastest-run selection or a replacement for all90 timing samples.
Every re-imported duration/symbol/grid/block and aggregate instruction count
is checked against its original CSV, including those read inside the native
rule. Native PCs are checked for complete action coverage and execution counts.
Missing reports/SASS/counts are INCOMPLETE, not zero or PASS. The rule writes
a separate completion denominator only after all actions have been exported:
old ACU can return rc0 even when a rule fails halfway. Missing optional fields
stay blank, not synthetic zero. No external Python ACU API is required on box;
the rule uses the CLI's embedded `HgRules`, without importing torch or json/gzip.

The new `/workspace/actlize-la-gdn-work-details-*/results.tar.gz` contains
compressed text: raw/details metric tables, native PC instructions/execution
counts/live registers/stall samples and provenance, plus per-action exact
source-mnemonic counts. It excludes operands, binaries and original large reports.
Upload only this small tar. No PPUProfiler, JIT or remote build is involved.

Local validation uses an existing REAL four-kernel QUactlize report. Both
SDK2.1.1's ACU (v2.1.1_20260725, data15000) and SDK2.2.1's ACU export the
embedded rule. An independent SDK2.2.1 Python reader agrees on all2761 PCs,
native instructions, function IDs, warp/thread execution counts and live
registers, plus all symbols/geometries/durations/aggregate instruction counts.
Total PC warp executions are1,964,224; the distinct aggregate instruction
counter sums to1,975,040. Those denominators must not be merged.
This validates these readers on this report, **not the site's older v2.0.0
reader**, GDN device performance, or the as-yet-unreturned GDN PC export.
Testing uses isolated matching userspace; no system runtime was replaced.
Host negatives cover changed report hash, missing cells/roles/repeats/kernels,
changed symbol/grid/duration/unit, empty SASS, absent execution counts and
lost metric-correlation entries. No production route changes accompany it.

### Exporter correction after the returned failure

The490B upload SHA256
`b521103d604c68c506b52cc365877d6f9d15bed8b551f0e99ca6f374ab2212ab`
contains only `INCOMPLETE / exports=[]`: the site's standalone Asight install
does not ship `extras/python/acu_report`. The previous script incorrectly
required that package. No GDN instruction detail was collected, and no kernel
failed or ran during this attempted export. The mistake was our installation
assumption, not a reason to rebuild the kernel or change the box SDK.

The first correction used the captured CLI only. Negative tests recreated
the missing Python package while proving the import command contains no
executable-to-profile/device-query arguments. Both wide raw and long metric-row
CSV formats are supported; wide count selection comes from the captured
`InstructionStatistics.section`, never a guessed counter name. Duplicate PCs,
missing/extra actions, wrong symbols, empty/negative counts, no executed work,
CLI errors and protobuf errors even with rc0 remain failures. Helpers are not
dropped from source counts. The optional `tools/acu_report_details.py` remains
developer-only for independent local validation, not a box dependency.

### Second correction: old CLI ignores the requested source page

The43048B upload SHA256
`9940b34b67455cca9cf9264aa3df7b19c1d316262258fd3347aa548cc3f02c72`
contains three **byte-identical**108399B raw/source/details files, all SHA256
`9d46ecb323b68a7830ed6183ebd3bc6a5852e9b60059cff0ed786e65d010ca37`.
Every file is an aggregate metric-row table, not native instructions; all
commands returned rc0 and empty stderr. The site capture CLI is
v2.0.0_20251231-4f7cd70 / data12006, executable SHA256
`7d750971b45b0bb1f4367df6b8b0f37656d1be89894d40b39ceb2010f865b2b6`.
This is not a renamed source column. Our newer-CLI test did not establish
old-CLI source-page capability; treating that test as sufficient was an error.
No new GDN opcode evidence or speed verdict follows from either failed export.

The default now uses `tools/acu_work_rule.py` through the **same** capture CLI's
`--stats-rule` interface, not `--page source`. The rule retains every action
(including FLA fills) and helper PC. Its function IDs are native IDs, not the
new source page's ordinal namespace. A repeated metric table cannot pass as
source data. Tests also plant a failure after a valid first action: missing
completion remains failure even if ACU itself returns0. Changed per-action
symbol/grid/duration/instruction count and truncated denominators remain red.

Cross-reader diagnostic caution: on the local four-kernel report, the old
SDK2.1.1 rule reports all not-issued-stall samples as0 whereas the new reader
reports1828 total; old access-sector/data-byte attributes are absent and remain
blank. General stall fields and the required instruction/count/register fields
agree. Preserve capture-reader identity and raw diagnostics; neither absent
fields nor zero samples establish zero latency. This is one more reason not
to silently substitute a newer reader and mix its stall evidence into the
old capture's tables. The exact site v2.0.0 rule run still requires the user's
offline confirmation; failure must remain INCOMPLETE, never a device PASS.
