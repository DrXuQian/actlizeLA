# Residual BC attribution: native transpose, publication, and the model boundary

Read-only analysis at `920e21c`; device kernel unchanged from captured
`dfe76c3`. No layout, scheduler, math, routing or box run was changed.
Scope B1/S2048/Hk16/Hv32/K=V128/C64, V32, grid128, both four/eight warps.

## We already use native ldmatrix.trans

`aiu::Tile::load<true>` selects actlize's
`ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16`, compiled to
`tsm.ld.swzl.b32x4.s0.t1.trans1`. It is not a software transpose, a separate
transpose kernel, or an unimplemented intrinsic falling back to shuffles.

There are two distinct meanings of transpose:

- The recurrence really includes `K^T @ scaledV`.
- A matrix such as H is mathematically used as `[K,V]` in `K @ H`, but
  the native MMA B fragment is addressed as `(N,K)`. Loading the current
  `[K,V]` shared placement into that fragment needs a transposed load.
  The load modifier does not change the GEMM's mathematical result.

The real production traits/offsets produce the following exact dynamic load
budget. Native instruction counts independently match the summed inventory.

| Source read | Source line in eight-warp TU | Four warps | Eight warps | Why trans? |
|---|---:|---:|---:|---|
| H snapshot | 58 | 131,072 | 262,144 | `[K,V]` shared to native B `(V,K)` |
| residual R | 95 | 65,536 | 131,072 | `[C,V]` shared to B `(V,C)` |
| scaledV | 126 | 65,536 | 131,072 | `[C,V]` shared to B `(V,C)` |
| K in state update | 130 | 262,144 | 262,144 | `K^T` is the update's mathematical A operand |
| **All transposed matrix loads** | | **524,288** | **786,432** | |
| K in projection, normal | 62 | 262,144 | 262,144 | normal A operand |
| P inverse, normal | 99 | 131,072 | 131,072 | normal A operand |

The factor outside each per-CTA/chunk inventory is128 CTAs *32 chunks.
The new transposed loads are H/R/scaledV reuse split among more warps,
**not additional K-transpose work**. It is also not another global load:
the measured KVD-to-TSM bytes remain112MiB.

Enumerating actual native A/B fragment slots shows that each normal-load
register contains two adjacent BF16 values in one physical32-bit word;
each transposed-load register takes its two BF16 values from different
physical words. This is a proved address/fragment distinction. How the
native TSM internally services those halfwords is a separate hardware fact;
it cannot be inferred just by writing `trans` in C++.

Direct negative control: execute the real actlize **normal** reader on tagged
bytes in the unchanged layout, compare against the required native B map.
Removing trans alone is wrong on3840/4096 H values and1920/2048 R/scaledV
values. Only diagonal elements survive; deleting the modifier is not a fix.

## Where can the transformation move?

| Plane | Can producer placement replace the transposed reader? | Constraint |
|---|---|---|
| R, scaledV | Yes: write directly in B orientation during the existing BF16 store | Internal-only intermediates; paired normal reader, no extra transpose pass |
| H snapshot | Possible only with a second-consumer redesign | Existing16B publisher also reads this buffer to write canonical `[K,V]` global snapshots; transpose placement breaks its contiguous vector assumption |
| K | Merely changing one shared orientation does not remove both uses | Projection needs K, update needs K^T; on eight warps both have262,144 loads. Reorienting K swaps which use needs trans under the existing MMA ownership |

R/scaledV already have a four-warp B-oriented implementation. Its existing
writer shortcut is **not** eight-warp aware: reusing it verbatim gives
1536/2048 incorrect destinations,1024 out of bounds. The logical placement
can be reused, but the producer base/fragment ownership must be rederived.
This was tested as a failing local negative, not installed in a kernel.

K could have two physical views, but creating/storing them adds storage or
delivery work unless a separate mechanism proves otherwise. Swapping GEMM
operands/transposing its result is another ownership redesign, not a free
flag change. No such candidate is implemented or timed by this analysis.

Do not assume `AIU_LOAD<...,Trans=true,...>` performs that transformation:
in the pinned actlize b16 specialization, true/false use the same bulk-copy
assembly form and differ in coordinate argument ordering. This wrapper flag
alone is not proof of a differently oriented physical destination. Inspect
the complete descriptor/layout/copy contract. R/scaledV/H are register-
generated intermediates anyway, not global tiles for AIU to reload for free.

## BC is not only the transposed-load counter

The eight-warp report measures3,932,160 read conflicts and4,046,848 write
conflicts. Writes are50.72% of the total. The exact source/native inventory is:

| Scalar write family | Executed warp instructions |
|---|---:|
| BF16 H snapshot | 524,288 |
| BF16 R | 262,144 |
| BF16 Vnew | 262,144 |
| BF16 scaledV | 262,144 |
| FP32 final H, once after recurrence | 16,384 |
| FP32 gates and beta | 16,384 |

These counts are unchanged from four warps. Scalar BF16 V reads add262,144;
gate/beta reads change147,456->163,840; vector snapshot/Vnew/final publication
reads stay102,400. All seven native shared opcode classes close exactly,
so the accounting has not silently omitted stores, scalar reads or publishers.

Two controlled contrasts localize meaningful changes:

1. Eight-warps adds262,144 trans loads and1,048,576 read conflicts. Earlier
   four-warp B-layout replaces131,072 trans loads by normal loads and removes
   524,288 read conflicts. Both imply an average increment of4 counted events
   per affected trans load in these mappings, not a universal opcode latency.
2. Four-warp B-layout changes only the placement of the524,288 R/scaledV
   scalar stores, leaving their opcode count fixed; write conflicts decrease
   by1,048,576. This is also a **writer-placement** improvement, not just a
   benefit from the normal reader. The scalar C-fragment producer matters.

One empirical account matches all six existing arms from the V16, B-layout
and eight-warp captures exactly:

```text
read_BC  = 4 * trans_matrix_loads + 3 * scalar_BF16_loads
write_BC = 3 * scalar_BF16_stores + 7 * final_FP32_stores
           - 2 * B_oriented_R_and_scaled_stores
V16 uses coefficient3 instead of7 for its final_FP32 stores.
```

For the current eight-warp arm this yields3,932,160 and4,046,848. **These are
evidence-calibrated coefficients, not independently proved bank-service
rules or measured per-PC conflict counts.** In particular the residual
786,432 read events are consistent with scalar V reads; the aggregate alone
does not uniquely prove that all other unchanged readers contribute zero.
The V16 cross-check is useful, but does not turn a fitted account into an
independent hardware proof.

## Why the old simple bank model is insufficient

The32-bank/4-byte model used by the earlier host layout probe is an explicit
model, not a documented native TSM request decomposition. For a representative
C-fragment scalar BF16 store, production offsets give these lane bytes:

```text
old: 0,2,4,6, 32,34,36,38, ..., 144,146,148,150, ..., 240,242,244,246
B:   0,32,64,96, 2,34,66,98, ..., 14,46,78,110
```

Across the complete enumerated domain, both have the same simple signature:
at most one distinct32-bit word per modeled bank, two distinct BF16 halves.
But hardware write-BC differs by1,048,576. Therefore this simple signature
is demonstrably **not sufficient to predict the counter**. Scalar subword
service, request grouping and native matrix-load phases are missing from it.
Do not change bank width until a guessed model happens to match and call
that a hardware discovery. Full-word broadcasts and separate halfword stores
also cannot be counted interchangeably.

The final FP32 row-major C store has a clear strided pattern:
`word = (lane/4 + 8*(slot/4))*32 + lane%4 + 4*(slot%4)` for the local tile.
It places eight distinct words on each of four modeled banks. The V16 pitch
halves that modeled multiplicity. This supports that final-store mechanism,
but its16,384 instructions occur only once, not32 times per sequence.

The installed per-PC HgRules API exposes execution/stall counts, not bank
conflict events. The public [PPU programming guide](https://help.aliyun.com/en/document_detail/2871803.html)
also warns that matrix-load row/column patterns can cause conflicts, but does
not specify this native transposed-load service/metric mapping. Precise
per-PC conflict attribution remains unproved without that missing layer.
Static address maps, native counts and the A/B deltas above do not depend
on making up that layer.

## Consequence

The first justified layout target is the **paired R/scaledV producer and
reader**, not removal of `trans` and not a generic claim that AIU/SWZL is
broken. It covers262,144 of786,432 trans reads in eight-warps (one third),
plus524,288 scalar stores. H and K remain separate work with additional
consumer constraints. The measured four-warp variant reduced BC without
reducing time, so there is no static promise of latency or1.5xFLA here.

Reproducible host diagnostic: `dev/ppu/l026_wy_residual_bank_inventory.cpp`.
Evidence: `/workspace/gdn-residual-bc-static-20260924` (`verify.sh`,
`inventory.csv`, `negative.log`, `reconcile.py`, `reconciliation.json`).
The diagnostic uses production native layouts, production offset helpers,
and the real actlize normal-reader simulator for the dropped-trans negative.
No kernels were modified, built for a device, or executed on a device.
