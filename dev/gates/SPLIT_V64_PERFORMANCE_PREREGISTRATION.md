# Split-V64 PPU performance preregistration

This document was written before the split-V64 PPU result was observed.

## Fixed subject

- Mathematical shape: `B=3, T=256, H=12, HV=24, K=128, V=128, C=64`.
- Baseline identity: `e2c9e6698358153c4197a7fde6e0a2895751d56d`.
- Baseline result: `1776.873970 us`, grid 72, 128 threads, 139,776 B shared,
  405,504 BF16 `m16n16k16` instructions and 11,520 TF32 `m16n16k8`
  instructions.
- Subject decomposition: two independent BV64 CTA owners per logical V128
  head.  The public tensor layout, chunk algebra and output/state dtype do not
  change.

## Predicted structural changes

For the same shape the subject must report:

- grid 144, block 128;
- 107,008 B shared per CTA;
- shared-memory block limit at least 2 on the 256-KiB/CU PPU;
- 516,096 BF16 `m16n16k16` and 23,040 TF32 `m16n16k8` instructions;
- zero spills;
- raw-bit correctness and stable output/state fingerprints.

The extra instructions are intentional: QK, KK, blocked inverse and W are
recomputed by both independent V-column owners.  The experiment asks whether
doubling resident warps from four to eight hides enough dependency latency to
outweigh that 27.27% BF16 and 100% TF32 recomputation.

## Verdict fixed before timing

- `ADMIT`: correctness passes, the two result envelopes do not overlap, and
  split-V64 is faster than 1776.873970 us.
- `UNRESOLVED`: correctness passes but the subject/baseline envelopes overlap.
- `REJECT`: correctness fails, any structural prediction above fails, spills
  appear, or the disjoint subject envelope is slower.

An admitted result is still an intermediate PPU tactic, not the final design.
The no-duplication successor is a real 256-thread BV128 collective with two
four-warp MMA cohorts and common QK/KK/inverse/W computed once.
