# Qwen3.5 post-cumsum Triton-alignment preregistration

This verdict is fixed before the first PPU execution of ABI v4. It must not be
edited to fit the resulting numbers.

## Scope and measured reference

Shape: `B=1,T=2048,Hqk=16,Hv=32,K=128,V=128,C=64`, forward only.
The caller supplies chunk-local `gamma_log2_cumsum`, so Triton's cumsum launch
(`3.40 us`) is outside both v4 and the primary comparison.

The measured Triton post-cumsum reference is:

| Stage | Grid role | Time |
|---|---|---:|
| KKT + triangular solve | `(sequence, V-head, chunk)` | 37.36 us |
| recompute W + U | `(sequence, V-head, chunk)` | 25.36 us |
| serial H recurrence | `(sequence, V-head, BV64)` | 84.28 us |
| output O | `(sequence, V-head, chunk, BV64)` | 83.04 us |
| total | four ordered launches | 230.04 us |

## Admission before timing

- L205 must report raw-bit v4 equality to the independent token recurrence and
  v1 on distinct/paired WY, zero/nonzero state, and the T65 tail.
- L214 must report exact-once grids `1024/1024/64/2048`.
- Workspace must be `1024 * 90112 = 92274688` bytes and contain only
  A/W/U/H-start/Vnew. A materialized causal P voids the alignment claim.
- ACU must show exactly four device kernels in this order. Missing, fused, or
  additional post-cumsum kernels make the run `VOID`, not slow.

## Fixed performance verdict

- `ALIGNED`: public-ABI aggregate median is at most `253.044 us` (1.10x the
  230.04-us reference), and no individual ACU stage exceeds 1.25x its matching
  reference stage.
- `PARTIAL`: all admission checks pass and v4 is disjointly faster than the v3
  control, but either aligned bound is missed.
- `NOT ALIGNED`: all admission checks pass but v4 is not disjointly faster than
  v3, or any stage is more than 2x its matching reference.
- `VOID`: correctness, grid, workspace, kernel-count/order, finite-writeback,
  or stable-replay admission fails.

The first report diagnoses alignment only. No tile, thread-count, shared-memory
or instruction optimization may be introduced to reinterpret that report.
