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

**2026-11-10 native evidence is now complete. No further export/upload is
needed for this work ledger.** Section 5 identifies a software BF16 conversion
fallback accounting for42.6–44.0% of default native PC executions. This is
not excess GEMM arithmetic. Section 4 retains the historical exporter failures
and their corrections; its requests to re-export have been satisfied.

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
  --acu acu
```

No compilation, device query, warmup or GPU launch. The script invokes the
**`acu` resolved through the caller's PATH, with `--import`**, verifies the original report hash,
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
The original `capture_profiler` is preserved separately from `reader_profiler`;
the latter's resolved executable, sections, hashes and version are recorded.
Each reader uses its own sections/ASIGHT_HOME. A different offline reader is
allowed, but different original kernel counts, timings or aggregate instruction
counts remain failures. Timing comparison uses the **original CSV's displayed
decimal precision**, not binary-float equality between differently formatted
exports; metadata and instruction counts remain exact. Counter selection still comes from the original
hash-bound capture section, not a new reader's preferred metric. Per-PC
diagnostics are reader-scoped and must not be mixed across reader versions.

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
This validates these readers on this report, **not GDN device performance**
or the as-yet-unreturned GDN PC export.
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

The second attempted fix used `tools/acu_work_rule.py` through the capture CLI's
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
old capture's tables.

### Tool-selection correction: use PATH's acu, not the old capture executable

The returned d1f0039 invocation conclusively rejects `--stats-rule` on the
hard-coded site v2.0.0 executable (rc1). The user clarified that **PATH's `acu`
is the newer version, supports this interface, and must be used instead**.
Do not keep trying native export options on the old CLI, install a replacement
SDK, or require uploading large reports. The active exporters/runners now
default to PATH's acu, with explicit overrides still honored. The native rule
itself is unchanged. This correction changes tool selection, not GDN kernels.

The old equality check tying the offline reader to the capture executable has
been replaced by separate capture/reader provenance. The actual report and
original values remain immutable. Host tests run the complete12-report export
against a distinct reader, retain original identities, and plant one changed
instruction count to ensure that different-reader support does not weaken
the evidence check. PATH symlinks resolve to the actual executable's companion
sections; a cwd file named acu or missing PATH entry cannot silently substitute.

### Returned re-import: decimal formatting, not changed time/counts

Archive `c20499105fe8d7cbfd95ad01ca5d86a641c2affd135ac41f7ba8b34aeb2cdaa8`
(49787B) binds the new reader to executable SHA256 `9e5de950...`, exactly the
local SDK2.2.1 version. All four default kernel names, IDs, process IDs,
grid/block geometries and aggregate instruction counts match. Only displayed
Duration precision differs: old metric rows use0.01ns, new wide CSV uses
0.000001ns. For solve,41931.76ns versus41931.764706ns is the same value at the
old display precision, not device drift. The raw-dictionary float equality in
our exporter incorrectly rejected this. The instruction counts are unchanged:
377856 /15428608 /40218752 /19683328 (prefix/solve/state/output).

The fix derives the decimal quantum from each original lexical value and its
unit, then checks that re-imported time rounds to that original value. It does
not introduce a percentage performance tolerance or alter historical timing
receipts/verdicts. A more precise original export tightens the check. Original
CSV plus a per-kernel comparison JSON now accompany the native export; any
actual difference prints the field and both values. Negatives plant a solve
time of41931.765001ns(outside the original rounding bin), one extra instruction,
and a changed grid; each must fail. Unit conversion also preserves the original
decimal quantum rather than reusing the textual exponent in another unit.

This upload stops before the native rule and contains no FLA detail. It closes
the formatting diagnosis, **not** per-opcode attribution or another performance
comparison. Continue the same offline command against existing reports; there
is no need for a new GPU capture.

## 5. Completed native ledger: a compiler-version guard selects software BF16

Returned archive SHA256:
`f782633493ff326e9eb9ed28db6578d59b9c7359b26b943e79a4472919bff8fd`.
It contains12/12 selected exports,60 captured actions,74749 static PC rows
and150 files. All bound file hashes, original report receipts, action inventory,
grid/block, aggregate counts and original-precision duration checks pass.
The source manifest SHA256 remains
`ad59ceb5e2964f2f23efd478f25e676d1bfa56e8f2f1fd339099991ce6a81697`.
This is offline analysis of the existing source38e8e91 captures, not new
GPU execution or a replacement for the90 original timing samples.

Transport note: the uploaded2413258B file contains a complete, CRC-valid
1730917B gzip member followed by682341 uninterpreted bytes. The original is
preserved unchanged. The first member has all150 expected regular files,
including the complete summary and all bound hashes; no unsafe tar paths or
links. The extra bytes are not evidence and do not require another upload.
The original large reports remain on box; their receipt hashes, not local
rehashed copies, bind this small export.

Reproducer: `dev/ppu/zw810_fla_kda/analyze_native_work.py` consumes the extracted
offline export and rechecks its12-report denominator, hashes, full native
completion markers and original captures before producing `native-work.json`.
The compact60-action ledger is retained in
`dev/ppu/zw810_fla_kda/results/native-f7826334.json`; it retains both metric
families, category totals, MMA counts and software-conversion denominators.
The executable reader is PATH-resolved SDK2.2.1 ACU hash9e5de950; the old
capture-reader identity remains separately recorded. Source locations in this
bundle are `<invalid>`: the attribution below uses actual operands plus pinned
source and an isolated compiler probe, not invented source-line bindings.

### 5.1 Measured MMA work closes the mathematical ledger

At B1/T2048/Hk16/Hv32, C64, K=V128, g=-0.1, repeat0:

| Work | Default | TF32x1 candidate | FLA |
|---|---:|---:|---:|
| Solve BF16 m16n16k16 | 81,920 | 81,920 | 81,920 |
| Solve TF32 m16n16k8 | 98,304 | 32,768 | 32,768 |
| State-equivalent BF16 m16n16k16 | 655,360 | 655,360 | 786,432 |
| Output BF16 m16n16k16 | 524,288 | 524,288 | 524,288 |

FLA state-equivalent includes262144 W/U +524288 recurrent MMA. Per chunk-head,
the BF16 counts are80 KKT,640 residual-state versus768 FLA W/U+state, and512
output on both sides. These ratios hold at all four returned geometries.
No repeated output QK or duplicated mathematical V panels explains the gap.
The default's extra65536 TF32 MMAs are real compensation work, already isolated
by the prior TF32x1 candidate; that candidate does not remove the BF16 fallback.

### 5.2 Do not mix two executed-instruction metrics

The following are **sums of native per-PC warp executions**, including every
kernel and both FLA fills. They are not `pu__inst_executed.sum` from Section1.

| T/Hv | Default PC sum | TF32x1 PC sum | FLA PC sum | Default software-BF16 core | Share of default |
|---|---:|---:|---:|---:|---:|
| 2048/32 | 64,577,664 | 63,737,984 | 37,192,704 | 27,525,120 | 42.6% |
| 2048/48 | 94,580,928 | 93,361,344 | 54,847,488 | 41,287,680 | 43.7% |
| 8192/32 | 250,200,192 | 246,947,968 | 147,336,192 | 110,100,480 | 44.0% |
| 8192/48 | 375,300,288 | 370,421,952 | 217,532,928 | 165,150,720 | 44.0% |

For2048/32 the default aggregate counter is75,708,544, not64,577,664.
FLA's two measures happen to agree here; ours do not. Preserve both. We do
not claim an exact hardware explanation for this counter difference or charge
the difference to missing PCs: all action/function/PC completion checks pass.

### 5.3 These masks implement rounding, not address generation

The pinned actlize423253c header `include/cutlass/bfloat16.h:120` selects native
conversion only under:

```cpp
#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100) && (__HGGCCC_VER_MAJOR__ >= 11)
```

The local HGCC2.2.1 device pass reports `__HGGC_ARCH__=100` and
`__HGGCCC_VER_MAJOR__=2`. Thus the CUDA-style major-version threshold selects
the **software** constructor. This is a compiler-version-family guard problem,
not absent PPU BF16 hardware. A compile-only probe using the unmodified pinned
header reproduces the same15-operation core; the same probe's explicit
`ppu.cvt.rtte.bf16.f32` emits one `v.cnvt.bf16.f32.rtte`. No device ran.
The locally runtime-patched hgcc executable hash differs from the box receipt;
its diagnostic is a local reproduction, not an assertion that their entire
toolchains are byte-identical. The actual box PC stream independently proves
the software branch ran.
The reproducible source is `dev/ppu/zw810_fla_kda/bf16_lowering_probe.cu`, SHA256
`517b5119807734721618e225acefd340de5b07dfbfbcc583a96333e781c27c07`.
The pinned header SHA256 is
`ca917597bcaf0b522f3d034ec36a333728a4e92883dee531ab5e14d67b56c8df`.

For example, state starts the repeated sequence at nativePC0x418119e0:
mask0x7f800000 checks exponent; mask0x7fff checks sticky bits; mask0x18000
checks rounding/retained LSB; add0x10000 supplies a rounding increment.
Fifteen executed mask/compare/select/execution-mask operations implement the
finite-input core. The later right shift, store, waits and constant moves are
**not** charged to those15; their work remains elsewhere in the ledger.
The special-value branch has zero active executions in this fixture, but must
still be validated before replacing the constructor. FLA directly emits
scalar `v.cnvt.bf16.f32.rtte` and packed `v.pcnvt.bf16x2`.

| Default stage,2048/32 | Logical BF16 results per chunk-head | Conversion warp executions | Measured15-op core instructions |
|---|---:|---:|---:|
| Solve inverse publication | 4,096 | 131,072 | 1,966,080 |
| State: H snapshot + residual + Vnew + gated Vnew | 16,384 +8,192 +8,192 +8,192 | 1,310,720 | 19,660,800 |
| Output: attention + final output | 4,096 +8,192 | 393,216 | 5,898,240 |
| Total | 57,344 | 1,835,008 | 27,525,120 |

There are1024 chunk-heads; logical scalar results /32 independently reproduces
the dynamic conversion counts. The parser requires the complete instruction
multiset, rounding literals, equal execution counts and non-overlapping PCs,
then checks this independent logical denominator. It does not call every
`and`, `cmp`, `mov` or `emsk` a conversion.

The software core alone contributes5,505,024 `v.and`,3,670,016 each of
`v.cmp.ne`, `s.lop.emsk`, `s.xor`, and1,835,008 each of `v.cmp.gt.i16`,
`v.cmp.eq`, `s.or`, `v.add.i32`, `s.and`, `v.csel`. This explains the dominant
mask/compare excess concretely. It does **not** explain every `s.mov` or
`v.mov.v2s`: e.g. full-call vector-to-scalar transfers are1,607,680 versus
FLA512. Those require a separate operand/descriptor/control attribution and
are not charged to this conversion core.

### 5.4 Next isolated change, and what is not promised

Fix the BF16 conversion selection **before** changing grids, warps, chunk size
or recurrence. Preserve all rounding locations, layouts, mathematical products,
global consumers and source order. Start with a narrowly scoped PPU conversion
adapter/candidate; do not globally fake `__HGGCCC_VER_MAJOR__=11` to activate
unreviewed code elsewhere in the dependency. A later actlize guard correction
needs its own target/SDK coverage.

A conservative instruction-only counterfactual replaces the15-op core with
one native scalar conversion, leaving shifts/stores/setup untouched. For2048/32
that removes25,690,112 PC executions from the default and gives38,887,552 versus
FLA37,192,704 (1.046x). This is **not** a compiled candidate count, latency
prediction or evidence that FLA has been beaten. Real compilation can change
lifetimes/scheduling/resources, and necessary memory/sync work remains.

Before a box handoff: compile the exact bodies, require native conversions and
no software signatures; retain the old branch as control; verify BF16 finite
ties/signs/subnormals and the NaN/Inf contract, plus all existing recurrence
and RAW-BIT delivery gates. On box compare complete calls across the registered
eight cells, not just one kernel; preserve the Hv48/T8192 winner. User owns that
device iteration. This audit changes no kernel, routing or benchmark thresholds.

Attribution negatives mutate a rounding-mask bit, the increment, an operation,
the path count or closing mask. Each must fail. Interleaved stores/conversions
remain outside the15-op charge, and native BF16 instructions do not match the
software signature. The source/header/probe identities and raw operands remain
available under the task's `work-detail-upload/` evidence directory.
Local scoped regression:286 tests PASS, including9 new attribution tests;
log `work-detail-upload/tests-native-final.log`. This is not a claim that the
entire repository suite is green: the previously recorded solve-static
source-equality failure remains out of this scope and was not weakened.

Round 3 now implements this isolated adapter and exact-body compile gates;
see [the registered round-3 contract](ZW810_GDN_FLA_KDA.md#round-3-conversion-only-native-bf16-candidate).
Local compile/host checks pass; device semantics and latency await the user's
box run. The public default and this work attribution remain unchanged.
