# Qwen3.5-35B-A3B T=2048 two-stage preregistration

This verdict was fixed before any PPU timing of the two-stage path.

## Fixed mathematical shape

- `B=1, T=2048, Hqk=16, Hv=32, K=128, V=128, C=64`.
- The head geometry is the published Qwen3.5-35B-A3B Gated DeltaNet shape.
- There are 32 chunks, 1,024 logical `(V-head,chunk)` cells, and grouped-value
  mapping `Hv:Hqk = 2:1`.
- Inputs, outputs, FP32 state, chunk algebra, and every BF16 rounding boundary
  are identical between v1 and v2.

## Fixed implementations

- Control: v1 legacy split-V64.  Grid 64; each CTA serially executes 32 chunks
  and recomputes QK, KK, inverse, and W for its V slice.
- Subject: v2 prepare + recurrence.  Prepare grid 1,024; recurrence grid 64.
  The caller supplies a 33,554,432-byte workspace containing one BF16
  `A(64x64)+W(64x128)+P(64x64)` record per logical cell.
- Both kernels use 128 threads and the existing 107,008-byte proved shared
  ledger.  There is no hidden device allocation or host synchronization.

## Structural postconditions

- v1 and v2 output/state must be raw-bit identical to the independent token
  recurrence for distinct and nontrivial paired-WY fixtures.
- Prepare and recurrence ownership must be exact-once over all 1,024 common
  cells and all 4,096 V columns respectively.
- A null workspace and a workspace one byte too short must fail closed.
- Expected per-logical-head/chunk AIU denominators:
  - v1: 1,792 BF16 MMA and 80 TF32 MMA;
  - v2: 1,408 BF16 MMA and 40 TF32 MMA.
- For the fixed shape this is 1,441,792 BF16 MMA and 40,960 TF32 MMA in v2,
  versus 1,835,008 and 81,920 in v1.  Zero spills remains a device
  postcondition.
- Logical workspace traffic is 32 MiB written by prepare plus 64 MiB read by
  the two recurrence slices.  Cache residency is not assumed by the verdict.

## Timing verdict

Run both paths from the same SHA, binary, device, and fixture.  Report all
sample envelopes and do not compare a historical number to the current v2.

- `ADMIT`: raw-bit admission and structural postconditions pass, the timing
  envelopes are disjoint, and v2 median is lower.
- `UNRESOLVED`: correctness passes but envelopes overlap.
- `REJECT`: correctness/structure fails, spills appear, or disjoint v2 is
  slower.

This is a milestone verdict, not the final performance admission.  The
same-shape PPU Triton/FLA trace supplied before v2 timing is:

| stage | duration |
|---|---:|
| chunk-local cumsum | 3.40 us |
| KKT + solve | 37.36 us |
| recompute W/U | 25.36 us |
| recurrent H | 84.28 us |
| output O | 83.04 us |

The kernel-duration sum is 233.44 us and first-start to last-end is 234.32 us,
so launch gaps total only 0.88 us.  v2 is expected to improve 10--20% over the
current approximately 1 ms v1 because its exact BF16 instruction reduction is
21.43%, but that does **not** close the product goal.  Final admission is
`<=234.32 us`; reaching it requires a subsequent H/O split so output work is
parallel over chunks instead of remaining inside 64 serial recurrence CTAs.
