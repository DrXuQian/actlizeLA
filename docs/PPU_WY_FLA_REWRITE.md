# FLA-structured C++/actlize forward rewrite

Decision2026-09-24: stop making the next box experiment another address
micro-ablation. Build an independent execution path aligned to the actual
FLA forward dataflow and operand layouts. The current admitted mask1520 path
and original strong-decay implementation remain controls, not deleted or
silently replaced. This document is a design/feasibility checkpoint, not a
completed rewrite or speed result.

Revision2026-09-24: separate demonstrated instruction/dataflow problems from
unmeasured replacement hypotheses. The five-stage design below is a reference-
aligned candidate, not a claim that five launches inherently beat three.

Latest implementation checkpoint: [structural split prepare](PPU_WY_SPLIT_PREPARE.md).
Prefix/KKT-solve/WU are separate opt-in kernels, now device-admitted. The
[new uploaded ACU result](PPU_WY_SPLIT_ACU_20260924.md) is269.306 vs FLA223.583
us; old AIU control304.995 us. User target is **FLA1.5x on all
ACU kernels**, not merely parity and not a complete-API-event win. This takes
precedence over historical implementation order/performance goals below.

Implementation checkpoint: [matched native AIU/SWZL state/output candidate](PPU_WY_AIU_PAIR.md).
It replaces manual global-input placement with paired bulk delivery and an
explicit internal layout. It does not yet change the three-stage graph,
prepare publication or joint QH/QK reuse. This is a separately measurable
delivery step, not completion of every rewrite item below.

## Current priority after the three-arm ACU capture

After split prepare, the remaining matched-phase excess is58.5% state,
34.7% prepare,6.7% output. State scheduling/delivery is now first priority,
then W/U's conditioning/shared roundtrip. The write-amplification mechanism
below has been fixed in the candidate, not the retained control. State/output
bodies are unchanged. Additional120.251 us is needed for the new149.055 us
target; a state-only change is insufficient. No default promotion.

Next opt-in implementation: [single-buffer state operand pipeline](PPU_WY_STATE_PIPELINE.md),
mask62960 with split30192 retained. It overlaps current K/U with W@H and
next W with the update using the same shared storage. Local native/lifetime
proofs do not establish a device speedup; the linked three-arm command is
the next measurement, not another claimed1.5x result.

### Before split prepare (historical prioritization)

The [2026-09-24 verified result](PPU_WY_AIU_ACU_20260924.md) supersedes the
historical instruction/time figures below. Current AIU prepare/state/output
are135.854/121.071/47.123 us versus FLA83.428/90.198/42.748 us. Including
FLA's6.491 us fills, totals are304.048 vs222.864 us. The user selects
**ACU time**, not complete Python API time, as the next optimization metric.

Remaining matched-phase excess is60% prepare,35% state,5% output. Resume the
structural plan here, not another series of descriptor micro-ablations:

1. Separate preparation responsibilities as prefix, KKT+solve, and W/U,
   with independent CTA/shared/register budgets. The current fused prepare
   carries70,144 B/CTA and128 threads through unlike phases; FLA W/U has
   25,600 B/CTA and256 threads. Its observed active warps are37.11 vs our
   fused prepare's11.76. This is evidence of a resource-layout difference,
   not proof that splitting alone provides the52.426 us saving. Use native
   actlize tensor tiles and vector publication for W/U. Include the cost of
   extra launches and intermediate traffic in the entire prepare comparison.
2. Preserve the admitted AIU state as control. Inspect the generated recurrent
   loop's operand reuse, prefetch/wait placement and publication lifetimes
   against the selected FLA loop. Both necessarily recur over chunks; both
   already keep FP32 state in registers. Barrier counts are unchanged by AIU
   but sync waiting rises. Do not remove barriers or claim overlap without
   verifying the generated schedule and dependencies.
3. Keep current AIU output first: only4.375 us remains against FLA. Joint
   QH/QK reuse remains a later independent test, not a prerequisite for the
   much larger prepare/state gaps.

Vector-only W/U publication is a useful causal control within the prepare
rewrite, not a claim that one store change closes the whole gap. Preserve
the current TF32 residual terms, BF16 boundaries, FP32 recurrence, independent
numerical gate and old path. The AIU patch did not deliver the five-stage
graph; the linked split-prepare candidate is its first implementation, now
device-measured as described above. State pipeline/joint output reuse remain open.

## Diagnosis ledger: what is established

Counts below are measured warp executions from the verified native reports,
not source lines or static instruction sites. Both state and output execute
524,288 BF16 MMAs in each implementation.

| Observed quantity | WY | FLA | What it establishes |
|---|---:|---:|---|
| State `v.madl.i32` | 3,167,744 | 75,264 | Much more integer multiply-add work around the same matrix arithmetic |
| State `v.mov.v2s` | 756,736 | 0 | Scalar-address operand delivery costs absent from the reference |
| State matrix-load instructions, all b32x4 families | 720,896 | 655,360 | 1.10x load instructions; not enough by itself to explain2.165x total instructions |
| Output matrix-load instructions, all b32x4 families | 786,432 | 458,752 | 1.714x matrix loads at the same MMA count: reuse/delivery must change, not merely the opcode spelling |
| Output `v.mov.v2s` | 851,968 | 0 | The same scalar-address seam affects output |
| Prepare W/U useful output / KVD store traffic | 32 MiB /512 MiB | 32 MiB /32 MiB | Our scalar `vmem.st.b16` publication amplifies this interface16x; this is not16x HBM traffic |

Native instruction neighborhoods show the WY vector-to-scalar base moves
feeding SWZL loads, while FLA NCOM loads directly consume lane-address VGPRs.
This is a demonstrated operand-form difference. It is not a claim that all
v2s moves are avoidable without changing layouts or that removing their
instruction count translates directly into a particular number of us.

### SWZL address preparation is not post-load layout repair

The local SDK2.1.1 [four-cell compile probe](../dev/ppu/l015_swzl_address_class_compile.cu)
holds the load family fixed and changes only the variability of its cube base:

| Load family | Cube base | Static `v.mov.v2s` sites | Actual load address |
|---|---|---:|---|
| SWZL | fixed | 0 | `[0x0] @sreg2` |
| SWZL | varies with warp | 1 | `[sreg3] @sreg2` |
| NCOM | fixed cube, lane-addressed rows | 0 | `[0x0 + vreg5 * 0x1]` |
| NCOM | warp cube, lane-addressed rows | 0 | `[0x0 + vreg5 * 0x1]` |

In the warp-base SWZL case, `v.mov.v2s sreg3, vreg5, 0x20` occurs
**before** the load and its destination is the load's base-address register.
It does not rearrange the loaded BF16 fragment. The fixed-base SWZL case
disproves the blanket claim that SWZL necessarily requires a v2s move.
Ordinary `v.mov.b32` and register-fragment conversions are separate costs;
this probe does not classify them all. No cross-family numerical, bank-conflict
or latency equivalence is claimed: the shared-layout contracts differ.

The user's AIU pairing distinction is important. A correctly matched
`AIU.swzl -> shared -> ld.swzl -> MMA` path need not perform a software
unswizzle between the matrix load and MMA. AIU handles the write-side physical
layout, while the matching load delivers its native operand fragment.
Selecting a cube/stage and supplying its address/descriptor are still work;
AIU input delivery alone does not prove those instructions disappear.

The captured WY path is **not** that complete AIU pairing. `stage()` in
`gdn_wy_common.cuh` and `state_stage()` in `gdn_wy_state_copy.cuh` use
per-thread `gdn_arch::async_copy16` into software-computed swizzled shared
addresses; `wy_mma.cuh::load()` then adjusts the cube base and calls SWZL.
Register-produced snapshots/conditioned values also have explicit shared
stores. These are three different questions: write-side placement, load-side
address operand preparation, and any required accumulator-to-operand remap.
Do not call all three SWZL overhead or unsupported-PTX emulation.

Provenance is ours, not an upstream PPU optimization: upstream `aa04271`
uses NVIDIA `cp.async` and SM75 LDSM atoms with its own shared layouts.
Port commit `1af3d5c` introduced `shared_copy.cuh`'s AIU-swizzled physical
layout and SWZL adapter while preserving the per-thread async-copy structure.
Later WY helpers inherited that choice (`gdn_wy_common.cuh::stage` was
factored in `a712a7d`). Correctly constructing the swizzled bytes in software
can be numerically valid, but is not equivalent to using AIU to perform the
placement. This delivery decision belongs in our optimization debt.

Project rule: an AIU.swzl-written matrix tile must use a matching ld.swzl
consumer. Keep cube geometry, pitches, transpose and fragment ownership
bound together. Do not change a single endpoint to plain/NCOM on the same
bytes. For eligible global input tiles the native AIU/SWZL pair is the
starting design; a manual-copy alternative needs an explicit reason and
complete delivery evidence, not just a compiling load atom.

Consequently the delivery design must compare a properly matched AIU/SWZL
path for eligible global input tiles against a proved lane-address NCOM
path, not decree that every SWZL load should be replaced. Internal values
born in registers cannot be turned into AIU global inputs for free; their
store/consumer layout or register reuse must be designed separately. Preserve
tails, GVA strides, transpose semantics and fragment ownership in either path.

The saved address-only prototype gives one local causal check: replacing
general shared-layout arithmetic reduces state static `v.madl.i32` sites
297->107 and body instructions2300->2018, with all16 old bodies preserved
in that initial compile. Its52 static v2s sites remain52, because it retains
the same load family. That proves a removable address-arithmetic component,
but is NOT a measured dynamic reduction or speed result. It also explains
why address-only changes are not the entire structural solution.

Three limits on attribution:

- The existing SWZL instruction itself is **native**, not an observed
  emulation sequence. We have not proved an unsupported-PTX compatibility
  fallback is the cause of this captured performance gap. The user's warning
  is a required audit, not permission to label all extra instructions emulation.
- The70 KiB fused prepare and repeated output exchanges are verified structure;
  their individual latency penalties are not isolated. Splitting prepare
  and replacing output delivery are testable candidates, not already-measured wins.
- Inverse TF32 residual arithmetic is an intentional precision difference.
  It is not an instruction-usage bug and remains outside address/load savings.

What is ruled out as a sufficient explanation: missing tensor-core use,
extra state/output MMA work, a smaller state grid, or simply lower output
occupancy. FLA also retains a sequential state recurrence across chunks.

## What is wrong with the present implementation

The [verified uploaded profile](PPU_WY_SHARED_ACU_20260922.md) establishes
three gaps, not just state: prepare135.350 vs83.107 us, state145.466 vs89.490,
output66.476 vs43.385. BF16 MMA work agrees phase by phase. State/output
execute2.165x/2.352x as many native instructions; their HBM reads and global
write traffic agree. State occupancy is equal; output occupancy is higher
on our slower implementation. Thus a different grid or another few reused
exponents is not a sufficient explanation or an architectural plan.

We matched tensor formulas and gradually improved tiling, but retained a
generic shared-cube delivery layer. FLA compiles a tensor ownership/layout
schedule together with each dot. Its selected native code uses lane-address
NCOM loads; ours uses scalar-descriptor SWZL loads, generic pointer/layout
arithmetic and extra exchanges/materialization. This is a structural
implementation difference, not proof that the GDN/WY equations are wrong.
Nor does it prove the SWZL instruction itself accounts for every excess.

The reference is the uploaded, hash-verified FLA source, not a moving latest
checkout. Only forward is in scope. The relevant source-to-stage mapping is:

| Phase | Actual FLA forward source | Current C++ structure | Rewrite boundary |
|---|---|---|---|
| Prefix and solve | `utils/cumsum.py`, `gated_delta_rule/chunk_fwd.py` | Prefix, shared lower/inverse, solve, conditioning and W/U in one70 KiB CTA | Separate prefix and KKT+solve; tile-local solve ownership |
| W/U | `gated_delta_rule/wy_fast.py::recompute_w_u_fwd_kernel` | Scalar BF16 W/U publication; KVD interface16x useful output bytes | Dedicated tensor-core W/U with vector publication |
| State | `common/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | FP32 register state already resident, but repeated generic shared delivery | Explicit resident-state/operand layouts, compile-time address increments |
| Output | `common/chunk_o.py::chunk_fwd_kernel_o` | QK first, then two V64 panels reloading Q for QH, shared output exchange each panel | Joint QH/QK schedule using loaded Q, whole selected V tile, direct/vector epilogue |

FLA itself still recurs over the32 chunks in the state kernel. Do not replace
this with a new parallel-scan approximation or claim FLA removed recurrence.
Also, our FP32 state already lives in registers; moving it there is not a
new optimization. The difference is operand conversion/delivery and surrounding
state publication, not simply where the main state variable is declared.

## A concrete adapter gap is locally reachable

Pinned actlize423253c0 `include/cute/arch/copy_ppu.hpp` deletes all six plain
PPU0010 LDSM entrypoints: old inline-PTX grammar was unproved. That is not a
hardware limitation. SDK2.1.1 public `hggc_mma.h` exposes `awmma::ldmatrix`.
The [compile-only probe](../dev/ppu/l014_plain_ldmatrix_compile.cu) uses that
API, not guessed PTX syntax or copied SDK internals.

Local real SDK compilation/disassembly produces:

| Public SDK operation | Native body load | v.mov.v2s | Registers / stack |
|---|---|---:|---|
| `ldmatrix<no_trans,4>` | one `tsm.ld.ncom.b32x4` | 0 | 32 /0 |
| `ldmatrix<trans_16x16b16,4>` | one `tsm.ld.ncom.mt1616.b32x4` | 0 | 32 /0 |

These match the instruction families used by FLA. Compile-only reachability
does **not** prove lane/register ordering, numerical behavior, bank behavior
or a complete-kernel speedup. Those are explicit next gates. The existing
actlize SWZL atom and disabled interfaces have not been changed.

Some unsupported PTX can lower through slow compatibility sequences. A
compiling source instruction is therefore not sufficient admission. Inspect
the actual target body, transfers, barriers, registers and spills; distinguish
compatibility expansion from generic address arithmetic. Prefer the verified
native SDK interface before restoring an unproved inline-PTX spelling.

## Structural candidate and its discriminating checks

Use an independent `fla_aligned` namespace/path, not more conditionals inside
the current mixed experimental kernels. Proposed modules:

- `include/gdn_qsa/ppu/fla_aligned/`: ownership/layout plans, typed operand
  and accumulator views, vector copies and the proved native load adapter.
  No dynamic layout decomposition in the MMA inner loop. Share these maps
  with exhaustive tests anchored to native traits, not a parallel host model.
- `csrc/gdn_chunk/gdn_fla_prepare_ppu.cu`: prefix, KKT+solve and W/U kernels;
  each has an explicit resource/lifetime budget. Keeping KKT+solve fused
  avoids a large FP32 lower-matrix round trip. W/U separation permits its
  own warp/register schedule and coalesced publication.
- `csrc/gdn_chunk/gdn_fla_state_ppu.cu`: a recurrent state kernel with FP32
  resident state and tile-address invariants hoisted outside recurrence.
- `csrc/gdn_chunk/gdn_fla_output_ppu.cu`: QH and QK scheduled together,
  followed by gated causal PV, without the current two-panel output exchange.
- A thin C++ launcher/workspace contract: initially five math kernels in one stream,
  no Python between stages, no host synchronization or decay heuristic.
  Allocation-owning public API and prepared-workspace entry remain separately
  measurable. Defaults/original routing remain unchanged.

The rewrite has three independent acceptance questions:

1. **State:** does changing ownership/layout plus native operand delivery
   remove the large address/control excess while preserving the same update?
   Verify the actual MMA inputs, not just a standalone NCOM probe. Merely
   replacing SWZL with NCOM while retaining redundant conversions is insufficient.
2. **Output:** does joint QH/QK scheduling and register reuse reduce the
   measured786,432 matrix loads toward the reference458,752, at unchanged
   524,288 MMA work? Q currently comes from shared first for QK and again in
   each of two QH panels. That is not duplicate HBM Q loading. Other layout
   conversions also contribute; do not attribute all40 extra loads/warp to Q.
   A higher register count is acceptable if spill-free and faster end to end;
   FLA already uses170 registers versus our98 while taking less kernel time.
3. **Prepare:** can vector W/U publication close the demonstrated interface
   amplification, and does a separate W/U resource budget pay for additional
   inverse traffic/launches? Retain the existing fused prepare as the control.
   A result with more kernels but worse complete latency must be rejected.

Old paths need not be rewritten to conduct these checks. New layouts and
interfaces live in the independent path. Compose only validated stages;
do not conflate a new native atom, changed precision and a new scheduler
in one un-attributable experiment.

Internal workspace may use the reference's token-major W/U/Vnew and
`[B,NT,Hv,K,V]` snapshots; it is not an external weight artifact. Its strides,
units and allocation extents must be declared once and consumed by every
stage. All required upper-triangle/tail zeros must be written explicitly:
omitting FLA's fill launches is valid only if our producer does that work.
No alias may cross a live producer/consumer interval.

Implementation order is native-load/layout proof, state, output, then split
prepare. Each stage can be compared independently before composing the whole
new path. This is one structural rewrite delivered in verifiable stages,
not several untracked whole-kernel rewrites at once. Do not spend a box run
on the superseded address-only prototype.

## Arithmetic is not free to change

Retain BF16 boundaries, FP32 state, tails/GVA and initial/final-state semantics.
FLA source uses log2-prefix plus exp2; our admitted path uses natural-log
prefix plus expf. Its output scale expression and inverse block composition
also differ. Do not copy those expressions and call it a layout-only change.
Keep the admitted evaluation/reduction order for the structural delivery
port and require raw equality to its control plus the independent2% oracle.
Any necessary arithmetic change must be separated and declared, not hidden
under that raw-equality contract.

In particular, current inverse merges retain TF32 high/high plus two residual
terms (98,304 native MMA vs FLA32,768). Do not silently remove them to match
FLA's instruction count. This precision cost remains separately accounted.

## Performance goal and measurement contract

The first goal is to approach the reference's **GPU-stage execution**, not
declare victory from its Python dispatch cost. Current matched math anchors
sum to215.982 us; all seven FLA kernels including fills sum to222.482 us.
They are profiler anchors, not a promised public-API latency. The new path
must also beat the same-run current352-us-class public API; a faster kernel
with a slower complete call is not a production win.

Record both:

1. Balanced complete public-API event spans for both g=-0.1 and-1.0, with
   original/scalar/incumbent496/shared1520/FLA retained as applicable. No
   allocation or required kernel omitted from one side only.
2. Same-call kernel timeline: kernel durations and launch gaps separately.
   The current352.192-vs494.490 API result and347.292-vs222.482 ACU sums are
   different protocols. Their difference is NOT measured Python overhead.
   Graph/preallocated timing is additional evidence only when equally
   supported and applied to both sides; it must not replace public-API timing.

Native codegen must show the intended operand path, no spill, useful work
unchanged and materially reduced delivery overhead. Device gates keep16
cases, independent oracle, raw bits where arithmetic is unchanged,8 repeated
launches, GVA/tail/nonzero initial/output-only checks. Keep loss/UNRESOLVED
results. No selector promotion from compilation or static instruction counts.

## Saved checkpoints

The abandoned-as-next-handoff address-only prototype is preserved at branch
`wy-state-operands-20260924`, commit794b2e7, not merged into `ppu-backend`.
It compiled and passed its coordinate gate, but its Python family, complete
binary gate and regression seal are unfinished. It is not box-ready. Its
static footprint improvement does not certify a device speed result.

Rewrite feasibility evidence is under
`/workspace/gdn-fla-rewrite-evidence-20260924/`. The native probe is the only
new compiled component in this checkpoint; the five-stage path above remains
to be implemented. No PPU device code was executed locally.
