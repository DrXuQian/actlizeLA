# Next bounded direction: chunk-parallel auxiliary preparation

Design only, not implemented or measured. The expanded14-workload target
remains unmet. S39–S46 do not justify promoting any default route.

## Why this is different

The current auxiliary WG computes QK, KK, inverse and beta conditioning
inside each state CTA's recurrent chunk loop. These products depend on the
input chunk, not on incoming H. V64 splitting duplicates them across both
value slices and forces the auxiliary/state/load roles into one CTA resource
budget. The failed shared-H/register experiments did not remove this coupling.

Separate preparation into one CTA per (batch,value-head,chunk), then run the
existing state recurrence using its exact prepared QK/KK operands. Do not
replace the recurrence with an affine scan, change FP16 inverse/BF16 operand
rounding, or drop the final-state output. Keep this distinct from FlashInfer's
CP algorithm, which also parallelizes the state dependency using transfer
matrices and a fixup.

## Proposed code boundary

- New `prepared_aux_layout.cuh`: private artifact extent, byte/element units
  and shared/global consumer map. It is not a new public weight/state layout.
- New `prepare_aux_kernel.cuh`: parallel QK/KK/inverse, reusing actual
  scalar-GDN math and exact prefix helper; distinct tails and GVA heads.
- New `precomputed_state.cuh`: no auxiliary math WG; a loader role publishes
  the prepared operands. Preserve every actual producer/consumer arrival
  and async retirement requirement.
- Minimal explicit-option hooks in SM90 kernel/types/launch/bindings only.
  Keep ordinary fused S24/S38 binaries and legacy SM80 untouched. Public
  signature remains the same; temporary storage is internal and stream-owned.
- Harness explicitly accounts for **both** child kernels of one complete
  forward. Existing single-kernel controls/references keep their old accounting.

Before editing, register one initial geometry (V64 state, unchanged C64/D128)
and its resource/lifetime gates. Do not create an unbounded tile sweep.

## Costs, not just expected benefits

Two64x64 BF16 operands are16KiB per chunk/value-head. For B1/T2048/Hv32:
16MiB scratch written; two V64 state CTAs read32MiB, total48MiB logical
scratch traffic. B2 doubles this; T8192 multiplies it by4. Preparation also
adds Q/K input reads. No cache residency or free-bandwidth assumption.

Removing the aux WG could allow state192+loader24 across256 threads, but
two resident CTAs require both the actual register allocation and shared
allocation to fit. The inherited scaled-Q scratch is unused by scalar GDN;
omit it only in the explicit new path, with a fail-closed guard against using
the legacy KDA consumer. One-stage prepared QK/KK buffers may reduce shared
bytes further; their overwrite/release lifetime must be re-proved.
No occupancy or speedup is claimed until compiled resources and device
occupancy API confirm it. This H800 design is not a PPU1.7 resource guarantee.

## Admission and comparison

1. Actual producer/consumer maps for all4096 cells of both operands, every
   head/chunk/tail, independent tag oracle and stale-layout/missing-owner
   negatives. Bind prefix/inverse rounding and all six real bodies (two
   prepare gate types plus four state gate/initial variants).
2. Source/native protocol gates; no serialized WGMMA or silently changed
   fallback body. Query resources, do not infer occupancy from thread count.
3. Existing14 numerical cases, two overflow stresses, parent raw equality,
   eight-repeat stability; full prepared matrices checked when isolating a
   failure. No performance after a numerical failure.
4. Four-cell B1/B2 gate screen, then full-forward nsys finalists against both
   pinned libraries. Include preparation and any initialization/reduction.
5. Re-run the frozen14 workloads/28 scenarios before calling it a broader
   win. Preserve the existing reference numerical failure rather than
   adjusting its tolerance or treating its missing time as a win.

Failure to beat complete-call time rejects the experiment even if state alone
is faster or its occupancy rises. No shutdown until the actual goal closes.
