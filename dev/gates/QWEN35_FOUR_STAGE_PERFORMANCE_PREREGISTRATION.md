# Qwen3.5-35B-A3B T=2048 four-stage preregistration

This verdict was fixed before any PPU timing of the four-stage path.

## Fixed subject and control

- Shape: `B=1, T=2048, Hqk=16, Hv=32, K=128, V=128, C=64`.
- Control: v2 two-stage `prepare -> combined recurrence`, measured on the same
  SHA, binary, fixture and device.
- Subject: v3 `prepare A/W/P -> U -> H recurrence -> O`.
- Stage grids are fixed at `1024 / 2048 / 64 / 2048`.  Only the 64-CTA H
  stage owns a cross-chunk dependency; U and O must never execute inside it.
- Workspace is `32 MiB` common BF16 A/W/P plus `80 MiB` value seams:
  BF16 U, BF16 H-start in resident-MMA B-operand order, and FP32 Vnew.

## Numerical admission

- v1, v2 and v3 output BF16 payloads and final FP32 state must be raw-bit
  identical on the independently anchored T65 tail fixtures, for distinct and
  paired-WY keys and both zero/nonzero initial state.
- The BF16 materialization proof must cover A, W, U, P and H-start.  Vnew is
  stored as FP32 because O rounds Vnew while the state update rounds
  `exp2(gamma_last-gamma) * Vnew`; reducing that seam to BF16 changes the
  declared arithmetic.
- Common cells, U cells, H owners and O cells must all be exhaustive exact-once.
  Missing one denominator cell, dropping the BV tile from an address, and
  treating a local BV64 column as a global V128 column are required-red plants.

## Timing verdict

The supplied same-shape Triton/FLA trace is a product target, not a control:
`234.32 us` from first kernel start to last kernel end.  The measured v2
control is approximately `5254.39 us`, but historical timing cannot substitute
for the same-run control.

- `ADMIT`: correctness and structural admission pass, v2/v3 sample envelopes
  are disjoint, and v3 median is lower.
- `UNRESOLVED`: correctness passes but the envelopes overlap.
- `REJECT`: raw-bit/coverage fails, spills appear, or disjoint v3 is slower.

ACU must report all four kernels separately.  A lower aggregate time without
the registered `1024/2048/64/2048` launch topology is VOID.  This milestone
tests decomposition; final product admission remains `<=234.32 us` and is not
weakened if the first decomposed implementation misses it.
