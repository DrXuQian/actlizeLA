# Integration and PPU1.0 transfer boundary

Status: plan/analysis, not a new shipping route or PPU performance claim.
Registered 2026-09-27 03:04:25 UTC. Updated 06:29 UTC: the user relaxed closure
to majority wins and remaining median losses <=5%, and explicitly ordered
H800 shutdown before local cleanup/PPU1.0 transfer. That performance condition
is met; see `SM90_H800_CLOSURE_20260927.md` for the complete denominator.

## Close performance before replacing the public path

Keep the fixed14-workload x2gate inventory, exact pinned references and all
complete-forward kernel sums. A lower graph/API median is not admission.
Retain the reference numerical failure separately; it is not our speed win.
Composite per-experiment wins do not certify a newly linked library. Under
the revised user order, stop the H800 after durable evidence, then perform
local cleanup with source/native-equivalence gates. A changed integrated
binary remains device-unverified until it has its own numerical/performance
admission; do not relabel the retained composite evidence.
Save raw reports, source/dependency/build identities and all losing results,
verify the local artifact copy, commit/push the code, then close the exact
user-authorized H800 instance. Do not stop another GPU job or change hardware
settings. An unmet target is not permission to silently stop the machine.

## Clean code deliverable

- Keep `cuda_sm80`, `ppu_aiu`, and `sm90` as separate execution families.
  Share public shape/dtype/oracle contracts, not incompatible pipelines.
- Within SM90, have one explicit configuration description for value tile,
  role budgets and pipeline counts. Dispatch only admitted configurations;
  do not copy whole collectives for each shape or scatter benchmark IDs
  (`S24`, `S50`, etc.) through production code.
- Keep the scalar gate, auxiliary product/inverse and recurrent state/output
  responsibilities separate. Document every intermediate's layout, rounding,
  owner, publication and async-retirement lifetime at its real seam.
- Preserve cuLA/FlashInfer attribution for inherited implementations. Keep
  our changes in narrow helpers/traits where the actual semantics agree.
  Do not unify FP16 and TF32 inverses merely because both are triangular.
- Retain immutable control binaries/evidence; archive rejected experiment
  branches and diagnostic tools outside the default build/dispatch graph.
  A speed rejection is not a reason to erase evidence or alter the oracle.
- Consolidate the experiment runner's repeated candidate conditionals into
  a validated manifest when its current measurements finish. The manifest
  must bind candidate/parent/build flags and the exact workload denominator;
  tests must reject a missing case or changed parent. Refactoring is followed
  by source/native-equivalence checks where applicable and full numerics.

## What can feed back into PPU1.0

These are transfer candidates, not an authorization to replace measured PPU
winners with an H800 configuration. PPU1.0 uses actlize AIU/SWZL and warp MMA;
PPU1.7 uses the separate CUTLASS3.6/Hopper family. H800 is neither PPU target.

| Mechanism / evidence | PPU1.0 applicability | Required next proof |
|---|---|---|
| Cache **relative** decay `exp(last-prefix[t])`, distinct from `exp(prefix[t])`; H800 S24 improved after exact operand-map proof | Promising. `gdn_wy_residual_warps8_hvlayout_ppu.cu` still computes `row_decay` in each state CTA. Price extra prefix/publication work, not just fewer consumer exponent instructions. | Keep exact rounded FP32 prefix/subtraction and `expf` semantics; prove all consumers and shared/global lifetimes. Measure complete prepare+state+output, not state alone. |
| Full-chunk and tail specialization; H800 S16 removed repeated dynamic work while retaining the tail | Candidate if PPU native code still repeats valid/mask logic in full chunks. | Inspect actual PPU body first; real owner/initialized-address maps for every tail. Retain final causal selects. Source `constexpr` alone is not native simplification. |
| Keep last inverse partials in their row-owning warp; S55 closes four FI cells | Transfer the **ownership idea**, not the FP16 implementation. Current PPU solve explicitly retains a FP32 diagonal and three-product TF32 rounding contract. | Derive actual PPU atom/lane map and partial lifetimes; preserve each declared cast/add/product, or admit a separately named numerical algorithm. No silent FP16/one-TF32 substitution. |
| Value slicing and role resource tradeoff; V64 wins underfilled H800 cases but loses larger grids | Existing PPU V16/V32/8warp experiments already expose the same tradeoff. H800 does not establish a new PPU selector. | Recheck grid supply, resident capacity **and** issue readiness with PPU's real CU/register/shared limits. Count duplicated K/P/auxiliary work; retain the measured PPU winner. |
| Independent O2/KV operand lifetimes; S50 improves H800 high-head cases | Async overlap mechanism is Hopper-specific. PPU1.0 does not inherit WGMMA commit/wait groups. | Only consider portable operand reuse/lifetime changes that actually lower PPU load/compute hazards. Keep AIU.swzl matched to ld.swzl; inspect native scheduling and full-call ACU. |
| Producer-proved EX2 fast domain / explicit indexed dispatch, S59/S60 | CUDA12.8-specific experiment; no PPU benefit established. S59 failed its native branch gate; S60 preserved actual branches and exact numerics but lost/overlapped every screen. | Inspect PPU standard math lowering and special-value behavior independently. Never strip a PTX prefix or assume `brx` compatibility is efficient native execution. |
| Asymmetric buffer-depth admission | Directly portable verification method, not a tuning value. QK=KK=2 had hidden that KK storage named the QK depth. | Give independent axes unequal values in actual-type probes and verify each pipeline's own storage/reader capacity. Equal defaults cannot test independence. |

The first row refers specifically to
`csrc/gdn_chunk/gdn_wy_residual_warps8_hvlayout_ppu.cu`, not a hypothetical
SM90-like PPU directory. Keep the PPU backend's selected source as the
authority when applying the proposed coefficient reuse.

H800 observations use nsys all-kernel sums. Future PPU comparisons retain
the site's `/sim/eec/shared/junfu.qx/asight/bin/acu`; no claimed PPU speedup
from H800 timings, source checks, static instruction counts or lower BC alone.
Detailed bound evidence: `SM90_AUXILIARY_SASS_20260926.md`,
`SM90_VALUE_SPLIT_20260926.md`, `SM90_PAIRED_STATE_20260927.md`,
`SM90_LOSING_CELLS_20260927.md`, and retained PPU residual/layout ACU reports.
