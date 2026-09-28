# What the measured S50 / FlashInfer SASS does and does not establish

2026-09-27. H800/CUDA12.8, B2/T2048/Hq16/Hv32/C64/D128, BF16 gates,
zero initial state. This is the remaining high-head gap, not a comparison
against an old scalar implementation or S67's rejected code.

S50 source170f34f, binaryaf0c4081e354f77f37f0073f4521c3d41717995655087e3bc2aa1b7fedc73ed9.
FI source5d9f8c8d97fa53e22952ce8672f475d235f07478. Its actual loaded JIT
object3149ca0e773cb068b4d4f07b48694ddde763311640b750b19fe5c2855d4562ec
matches the S50 B2 nsys receipt; complete embedded cubin
2ac4802c189420aad6c016c1197cb64141848824fd7bdb015fc0617f6375e877.
Both use512threads, two state WGs, roles24/104/192/192. No stale geometry
or four-C++-specializations versus one-DSL-body denominator.

## Equivalent recurring intervals

Native PC ranges: S50 aux1590–5800/state d810–10190; FI aux3660–6fd0/
state cd10–f520 (hex, inclusive). Count only the actual BF16/no-initial
symbol. Prologues, tail clones and outlined cold wait blocks are excluded.
These are static instruction positions, **not executions or stall cycles**.

| Per steady chunk body | S50 | FI |
|---|---:|---:|
| Auxiliary sites |1064|920|
| Auxiliary HGMMA / inverse HMMA |16 /14|16 /14|
| Auxiliary MUFU / FSETP / FMUL |30 /30 /212|32 /0 /160|
| Auxiliary full IMAD family |48|13|
| Of those, IMAD.MOV.U32 (moves, not address multiplies) |23|2|
| Auxiliary LDL |3|0|
| State sites |665|642|
| State HGMMA |28|28|
| State scalar BF16 conversion / packed F2FP |32 /96|0 /96|
| State LDL / STL |0 /0|0 /0|

The auxiliary gap is144sites (+15.7%); state23 (+3.6%). They are not additive
latency fractions: producer/state roles overlap, predicates differ and
hardware stalls are not measured here. NCU counters remain permission-blocked.

## Source and dependency attribution

FI's `delta_rule_sm90.py:qk_and_kk_epi` explicitly uses
`cute.math.exp2(..., fastmath=True)`. Our standard `exp2f` keeps the
underflow correction (compare, optional half-input and RN square). Thus
part of the extra arithmetic is an actual numerical/codegen contract
difference, not more inverse matrix work. Removing it unconditionally is
not a legitimate delivery optimization: the prior extreme-input negative
already showed differences. S60's proved fast-domain branch preserved RAW
but lost/unresolved six screens; lower fast-path counts were not enough.

The three recurring auxiliary LDLs (34e0,4820,4bb0) read stack slots8,0,4.
Their only stores are1390,1410,13d0 **before** the loop. They hold invariant
coordinate/address values, not the state matrix and not a spilling WGMMA
accumulator. Nearby inverse/MMA source-line annotations alone obscure this.
Trace their uses (for example slot0 feeds shared-address addition at4990;
slot4 feeds4d40) before proposing state storage as their remedy.

The32 state scalar conversions immediately follow NewV retirement. The
logical FP32-to-BF16 conversion occurs while retile materializes the RS
operand; FI converts in logical order first. S69 isolates that exact seam
on S50, reusing the existing RNE array converter. S35 previously combined
this with different chunk clones on S24 and was unresolved, so it is not
an already-admitted win. S69 must retain O2/KV's disjoint live operands and
their common wait0; its initial native gate removes128whole-body scalar
sites without changing matrix groups, stack16/104B or TMA work. Four B2 nsys
captures now close the test: S69 weak/strong118.897/119.569us versus fastest
FI114.129/114.913us. All parent-paired S50 ranges overlap despite lower S69
medians, so no formal parent speed promotion; FI still wins, QLA still loses.
Evidence: `/workspace/gdn-sm90-packed-newv-20260927/s69-nsys` (264complete
forwards, exact local SQLite re-extraction).

On2026-09-27 the user explicitly authorizes an exp2-only fastmath macro,
matching FI's arithmetic choice. S74 is an arithmetic variant, not a claim
that standard and fast exp2 are generally raw-equivalent. Its default stays
standard; the independent2% output/state gate remains unchanged. The generated
fast bodies remove64 underflow comparisons and128 conditional multiplies
per actual type while preserving all64EX2sites and matrix/pipeline counts.
Full-body positions fall6440->6240(noinitial),7224->7024(initial). Normal14
CPU cases and two extreme diagnostics pass; performance is still in progress.
Keep this experiment distinct from the prior exact range-proof attempts.

S67 and S68 are separate evidence, **not the diagnosis of S50**: S67 one-WG
state emitted C7512 and one wait per MMA; S68 exact FP32 shared parking
restored async batches and removed zero-initial spills, but all six high-head
screens lost by~28–34%. Correct maps/async lowering did not pay for the
extra shared traffic and reduced state parallelism. Keep S50; no routing
change. Neither experiment supports declaring the reference gap exhausted.

Raw audit: `/workspace/gdn-sm90-sass-review-20260927/{inspect_matched.py,
matched.json,s50-lineinfo.sass,launch.sm_90a.cubin}`. Parent measured receipt:
`/workspace/gdn-sm90-paired-tail-20260927/s50-nsys/flashinfer-g-0.1/receipt.json`.
S68 maps, four-image local/remote comparison,14CPU/parentRAW+2stress and
six screens: `/workspace/gdn-sm90-state-park-20260927`.
