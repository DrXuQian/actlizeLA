# PPU chunked-GDN forward: cuLA gap ledger

> Experimental branch status: the first resident-fragment conversion is
> locally wired, but has not yet passed a real hgcc build, PPU RAW-BIT check,
> or ACU 1,408-MMA/zero-spill postcondition. Stable `main` remains the device-
> validated QK/KK-AIU plus scalar generated-product implementation.

Scope is the forward path only.  There is no PPU chunked-GDN backward API,
kernel, dispatcher, or gradient workspace in this repository.

“All Tensor Core” means every forward operation that is mathematically a dense
matrix product uses PPU AIU.  It does not mean that `exp2`, masks, elementwise
gates, address generation, or a triangular dependency chain become MMA
instructions.

| Layer | cuLA SM90/SM100 | PPU before this work | Closure order |
|---|---|---|---|
| QK and KK products | GMMA/WGMMA or UMMA | PPU AIU, proved builder route | closed |
| Generated-operand products | Tensor Core products with register/shared operand transforms | six products were scalar inner loops | first: resident-fragment PPU AIU |
| Triangular inverse | scalar small diagonal inverse plus Tensor Core block updates | scalar 64-row forward substitution | second: AIU block updates; keep the irreducible base solve scalar |
| Gate/mask/rounding | vector/SIMT around MMA | vector/SIMT | retain; preserve every BF16 seam |
| Operand transport | TMA pipelines and warp-group role separation | generated BF16 materialization plus register gather | third: profile, then pipeline/role-specialize |
| Scheduling | persistent, role-specialized work | one 128-thread CTA per chunk-head work tile | third, after arithmetic coverage is closed |

## First closure denominator

The six generated-product kinds are:

1. `W = A @ scaled_K`;
2. `U = A @ scaled_V`;
3. `QH`;
4. `WH`;
5. causal `P @ Vnew`;
6. state update `K^T @ scaled_Vnew`.

They form 11 logical product instances and 1,152 PPU m16n16k16 operations per
full C64 chunk.  QK/KK add 256, for a dense-forward total of 1,408.  Dynamic
ACU admission must use that denominator; merely finding an MMA opcode would be
a false green because the old QK/KK route already emitted one.

The first closure deliberately keeps the production BF16 boundaries and the
139,776-byte shared-memory ledger unchanged.  It is complete only when the
local product/address/tail oracle is exhaustive, the independent recurrence is
still exact, the shipping PPU target builds, ACU reports the full denominator
with zero spill, and device output/state remain correct.
