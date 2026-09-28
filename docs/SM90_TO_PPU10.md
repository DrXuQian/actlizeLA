# SM90 -> PPU1.0 migration status

Updated 2026-09-28. The user suspended the old 1.5x FLA goal; the current
criterion is **architecture-independent transfer completeness**, with
unchanged numerical admission and measured non-regression before promotion.
Historical experiment registrations remain historical, not current targets.

Not all portable mechanisms are closed. Implemented opt-in candidates are
also not the same thing as an integrated public/default selector.

| Retained SM90 mechanism | PPU1.0 status | What remains |
|---|---|---|
| Scalar gate outside matrix products; residual formulation without W/U materialization | Implemented and admitted in residual path | Do not replace PPU inverse precision to imitate Hopper |
| Prefix/relative gate coefficient reuse | Implemented gate-cache, included in full-chunk control | Reuse is CTA-local, not one global evaluation across all V slices/output |
| Static register indexing and invariant address bases | Implemented static diagonal + paired-layout producer bases | Not a claim that every dynamic address calculation is gone |
| Full chunks specialized, guarded tail retained | Partial: state/solve/output full-C64 candidates compiled and device-admitted | S%64!=0 still falls back for the whole call; interior-full + final-tail partitioning unported |
| No-initial first-chunk KH/QH elimination | [Implemented as first-chunk](PPU10_FIRST_CHUNK.md); local compile/algebra/native proofs pass | Device admission and both-gate ACU pending; explicit state/tails unchanged |
| Keep inverse intermediates with their final owner | Partial / applicability not closed | PPU diagonal is already warp-local; off-diagonal sm.temp[warp] is also warp-private. Investigate register retention at that actual seam, not a fictional cross-warp reduction |
| Convert NewV before operand rearrangement; paired conversion | Partial / applicability not closed | PPU already casts into consumer-oriented shared planes. Native paired BF16 conversion/ownership benefit remains unproved; keep separate unscaled/scaled rounding |
| Workload-dependent V tile and warp geometry | Existing V16/V32, 4/8-warp candidates | PPU-specific selection not integrated. Earlier V16 doubled input traffic without a meaningful win; do not copy H800's winning tile |
| Unified shape-driven public selection | SM90 integrated; PPU optimized residual variants explicit | PPU numeric/backend selection and non-regression admission remain separate integration work |

## Measured anchors and limits

The [static-solve + gate-cache composition](PPU10_STATIC_SOLVE_ACU_20260928.md)
reduced complete-call time by about 3% at both gates. The
[full-state specialization](PPU10_FULL_CHUNK_ACU_20260928.md) is about169 us
for B1/S2048/Hk16/Hv32/K128/V128, with the same public input and state contract.

The subsequent full-solve/output six-cell upload is complete, not pending:
measured source `d7c5c663bf87902251e29f6bc59aefd18d7488c3`, package HEAD
`8bbd3dc`, outer tar SHA256
`924e82b0fc4503471aa41ce5ffd2de1b72893b3dfdc7cb9ed0cf925549e94d21`.
All four deliveries passed 36x8 admission; all90 kernels and six manifests
were reconciled. Complete-call A/Bs, g=-1 / -0.1, in microseconds:

| Candidate | Paired control -> candidate, g=-1 | Paired control -> candidate, g=-0.1 | Conclusion |
|---|---:|---:|---|
| full-chunk-solve |169.88353 -> 168.86882|168.91177 -> 170.18293|Mixed signs; no promotion|
| full-chunk-output |169.45647 -> 168.71706|169.92823 -> 168.46941|Small observed gain, pending repeats|
| full-chunk-both |169.63412 -> 169.32471|168.95294 -> 168.89471|Too small to claim robust gain|

Most output-only whole-call variation was in unchanged stages. Full solve's
static body shrank but executed instruction count grew 1.36%: two address
reciprocal chains moved before the single-issuer mask. Static instruction
savings are not speed evidence. No default changed as a result.

## Architecture-specific: do not mechanically transplant

- TMA, WGMMA asynchronous O2/KV overlap, named-barrier opcodes and warpgroup
  register redistribution need their hardware contracts. PPU1.0 uses native
  AIU.swzl paired with matching ld.swzl and warp MMA. Do not emulate Hopper
  instructions just to call a port complete.
- Hopper's FP16 inverse versus PPU's FP32/three-product TF32 inverse is a
  numerical algorithm difference, not a missing architecture-independent port.
- Rejected fast-exp2 or synchronization-removal experiments were not retained
  SM90 wins and are not mandatory migration items.
- A fused PPU kernel would be a separate scheduling experiment, not an
  equivalent copy of the cooperative WGMMA pipeline.

## Closure order

Finish first-chunk device evidence; then mixed full/tail handling; then
resolve actual inverse retention and conversion/publication applicability;
finally integrate admitted PPU selections behind the common API. For each
item record implemented+validated, already-equivalent, or inapplicable with
evidence. A correctly tested slower candidate can close the investigation
without being forced into the shipping path. Do not restart H800 for this.
