# GDN native AIU/SWZL delivery audit

Scope: PPU0010/SDK2.1.1, current residual and retained WY forward paths.
Reference read on2026-09-24: QUactlize85a2352
`quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl`
(`DefaultGemm_AIU_Operand`, matched cube write/read construction), its pinned
actlize `copy_ppu0010_aiu.hpp`, and GDN's `wy_aiu.cuh`. The reference is a
producer/layout/consumer contract, not just using an instruction with SWZL in
its name. All cube dimensions, element/byte pitches, transposes and native
MMA lane/slot ownership must agree. QUactlize's standalone Marlin is a
separate cp.async/plain-ldmatrix policy; it is not the AIU authority here.

| Stage / values | Current global-to-shared | Current shared-to-register | Applicability / remaining gap |
|---|---|---|---|
| Prefix scan / g | scalar global reads | shared carry + warp scan | Not a matrix product; replacing with ldmatrix is not appropriate. |
| KKT / K | `Key::stage`: native AIU.swzl, padz tail | `Key::load`: native ld.swzl, native BF16 MMA | Already uses the QUactlize-style pair. |
| Solve / lower,inverse,temp | produced inside CTA as FP32 | scalar shared reads, explicit TF32 high/residual conversion and MMA | Not missing a global AIU copy. A matrix-load rewrite must preserve the three-product precision algorithm and diagonal dependencies; separate candidate. |
| Retained materialized W/U / K,V,inverse | native AIU.swzl | native ld.swzl before BF16 MMA | Already paired; conditioning writes share the declared layout. This kernel is absent from the residual algorithm. |
| Retained WY state / W,K,U | native AIU.swzl | native ld.swzl | Already paired; old scalar/cp.async arms remain only as counterfactuals. |
| Residual state / K,P,V | native AIU.swzl | native ld.swzl for K/P; per-element V read during residual calculation | Already paired; input K/P staging must not be duplicated just to add blocks. |
| Residual state / H snapshot | FP32 registers -> BF16 shared stores | transposed ld.swzl, plus vector global publication | Not an AIU input. Layout must serve both matrix and global publication consumers; retained in this experiment. |
| Residual state / residual,scaled-V | registers -> BF16 shared stores | transposed ld.swzl in control | No external publication; test direct B-oriented native SWZL microcubes, with non-transposed ld.swzl. |
| Output / Q,K,H,Vnew | native AIU.swzl | native ld.swzl (including H/V transpose) | Already paired. |
| Output / causal attention and output fragments | register production -> declared SWZL shared stores | ld.swzl for attention, vector publication for output | Cannot replace register stores with a global-load instruction for free. |
| Original reset/Hillis-Steele control / chunk operands | per-thread cp.async with software SWZL addresses | PPU SWZL adapters, some explicit fragment remaps | Separate retained algorithm/control, not the current residual candidate; not silently relabeled as native bulk AIU. |

Thus the optimized GDN path is **not still using a generic NVIDIA shared
delivery layer for its bulk operands**. Its load atoms are the same PPU
primitive family used by QUactlize's mature AIU route. This does not claim a
fully tuned collective, no BC, optimal occupancy, or equal performance.

Next smallest controlled change is the two internal B operands. Keep the
same V32/grid/threads, byte counts, BF16 rounding and barriers. The wider
transposed32x64 cube was deprioritized locally despite correct mapping because
it added111 native static sites and two registers. Fixed16x16 microcubes
make the writer an affine lane base plus immediate slot offsets; verify the
actual device body, not just a constexpr source declaration. Bank-address
enumeration is a model with explicit transaction assumptions, not an ACU
conflict measurement. Device RAW-BIT equality and full-call ACU remain
mandatory; no default routing change follows from local compilation.

The user clarified that zero added work is a design preference, not a veto:
the final selection is based on same-correctness complete ACU kernel time.
R1 has NOT been measured and is not proven slower. The smaller codegen/resource
cost makes microcubes the first bounded experiment, not an admitted winner.
