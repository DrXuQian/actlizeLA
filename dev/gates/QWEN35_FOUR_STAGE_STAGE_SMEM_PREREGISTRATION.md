# Qwen3.5 four-stage stage-local shared-memory preregistration

This verdict was fixed after the first four-stage decomposition timing and
before timing the stage-local shared-memory variant.

## Frozen control and subject

- Shape: `B=1,T=2048,Hqk=16,Hv=32,K=128,V=128,C=64`.
- Same-SHA two-stage control: median `5273.461914 us`, range
  `[5258.705902,5287.427902] us`, raw-bit/stable PASS.
- First four-stage control: median `6636.763763 us`, range
  `[6636.202240,6648.165894] us`, raw-bit/stable PASS.
- Subject changes only the dynamic shared-memory liveness ledger.  Stage
  grids remain `1024/2048/64/2048`, workspace remains `32 MiB + 80 MiB`, and
  all arithmetic/materialization boundaries remain unchanged.
- Required stage ledgers are prepare/U/H/O =
  `57856/41472/98816/66048 B`; v1 and v2 remain `107008 B`.

## Admission

- The independent CUDA device oracle must keep v1/v2/v3 output BF16 and final
  FP32 state raw-bit identical for distinct/paired WY and zero/nonzero state.
- L204 must instantiate all four exact shipping kernel types with the four
  distinct ledgers.  ACU must report those resources, zero spills, and the
  registered four grids; otherwise the timing is VOID.
- `REJECT`: correctness/resource admission fails, or the optimized four-stage
  sample envelope overlaps/is slower than the original four-stage range.
- `PARTIAL`: it is disjointly faster than the original four-stage control but
  remains slower than the two-stage range.
- `ADMIT`: it is disjointly no slower than the two-stage control.

The supplied Triton five-kernel span `234.32 us` remains the product target.
Neither `PARTIAL` nor `ADMIT` weakens that target.
