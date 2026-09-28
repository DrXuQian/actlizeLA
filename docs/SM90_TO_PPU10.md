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
| Full chunks specialized, guarded tail retained | Full-C64 candidates device-admitted; [mixed full-prefix/final-tail state](PPU10_MIXED_TAIL_ACU_20260928.md) device-validated and four-cell ACU evaluated | Mixed state saves1.40–3.47% whole-call in single captures; opt-in pending repeats, not default-promoted. FP32 recurrence stays in one CTA. Mixed-tail solve/output remain generic |
| No-initial first-chunk KH/QH elimination | [Implemented and device-validated](PPU10_FIRST_CHUNK_ACU_20260928.md); both-gate ACU evaluated | Mixed timing (-1: +1.11%, -0.1: -0.41%); keep opt-in, no default promotion. Explicit state/tails unchanged |
| Keep inverse intermediates with their final owner | [Implemented and device-validated; investigation closed](PPU10_INVERSE_REGISTER_ACU_20260928.md) | Both complete-call captures slightly slower (+0.54%/+0.17%); retain control, candidate opt-in only. BC -8.16%,84->76regs, same shared-limited capacity; lower scratch traffic is not a speed win |
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

The first-chunk follow-up at `21cee5e` is also complete: both deliveries pass
36x8 device admission and 16x8 signed-zero/None-vs-explicit-zero edge cases.
Complete-call control -> first-chunk is 167.75058 -> 169.61707 us at g=-1,
168.95765 -> 168.27294 us at g=-0.1; one ACU capture per arm/gate.
State/output each omit exactly 8,192 MMA, but executed instructions grow
1.272%/3.423%. Native code repeats accumulator initialization across the new
history guard and increases address moves. BC and global-to-shared staging
bytes stay fixed. This closes the migration experiment without admitting a
default optimization; detailed evidence is in the linked result record.

The mixed-tail state follow-up at `d8b8b85` is closed:36 retained x8 and261
edge cases x8 pass. Complete-call control -> candidate, S2049 at g=-1/-0.1:
181.90000 -> 177.30707 /182.26295 -> 175.94001us; S2111:
180.73941 -> 178.21471 /182.64117 -> 178.03883us. All four signs improve;
same-cell FLA is227.71–232.09us. Each pair retains675840 state MMA and same
BC/staging bytes; executed state opcode sums fall9.42%/8.64% by length.
This is state prefix-predicate removal, not less math/traffic or more active
warps. S2048 and default routing remain unchanged; repeated performance
admission is still separate from closing the migration investigation.

The inverse-register follow-up at `3227b18` is closed:36 retained x8 and256
all-extent/gate/initial cases x8 pass RAW-BIT and the unchanged2% oracle.
Complete-call control -> candidate at g=-1/-0.1 is169.28529 ->170.20471 /
168.67236 ->168.96294us; FLA225.28706 /225.51236us. All30 kernels and43 old
native/resource controls are accounted for. The removed6MiB read+6MiB write
scratch traffic is replaced by98304native shuffles and49152additional
selections. BC falls8.16%, but total solve instructions grow0.287% and
shared capacity still limits5CTA/CU despite84->76registers. One capture per
arm/gate establishes no observed benefit, not a significant regression or a
universal rejection of register retention. Keep control/default unchanged.

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

First-chunk, mixed full/tail state and inverse-retention device investigations
are closed. Three items remain: conversion/publication applicability,
PPU-specific shape/geometry selection, and unified public selection. Next,
resolve conversion/publication applicability; finally integrate admitted PPU
selections behind the common API. For each
item record implemented+validated, already-equivalent, or inapplicable with
evidence. A correctly tested slower candidate can close the investigation
without being forced into the shipping path. Do not restart H800 for this.
