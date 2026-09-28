# Residual B-layout: fewer bank conflicts, no measured latency benefit

## Verdict

Do not promote `residual-blayout`. The intended native load change executed,
correctness passes, read/write bank-conflict counters fall, but the modified
state kernel is effectively unchanged in this single ACU capture. Default
routing is unchanged; retain this arm as a counterfactual.

Source `3c7da093fe34606a7a3e51bed20aae261048ebc8`, B1/S2048/Hk16/Hv32/
K=V128/C64, g=-1.0, PPU-ZW810 72 CU. All15 kernels report1.700GHz.
These are **ACU kernel durations**, not Python/API event spans. The complete
sum includes all4/4/7 kernels and both required FLA fills.

| Stage, microseconds | Residual control | B-layout | FLA |
|---|---:|---:|---:|
| Prefix | 2.45353 | 2.45176 | 2.76588 |
| KKT/inverse | 52.40706 | 52.42647 | 43.97471 |
| W/U + state (ours algebraically combined) | 130.48412 | 130.33353 | 36.07529 +91.18529 |
| Output | 45.33353 | 47.00824 | 43.67588 |
| Required separate fills | none | none | 4.29059 +2.28235 |
| **Complete kernel sum** | **230.67824** | **232.22000** | **224.24999** |

State changes-0.15059us (-0.115%); complete sum+1.54176us (+0.668%).
Unmodified stages account for+1.69235us, including output+1.67471us.
They execute the same symbols and identical opcode counts in the same DSO;
do not attribute that variation to the changed state source. This is neither
a demonstrated speedup nor proof of a repeatable regression.

Candidate is3.55% slower than this capture's FLA sum. The1.5x goal would
require149.50000us, not reached. If all other stages were frozen at their
current101.88647us, state would have to fall to47.61352us. A small BC change
cannot be presented as closure of that gap. This is budget arithmetic, not
a prediction that such a state time is attainable.

## The layout change did run and reduced BC

| State metric | Control | B-layout | Change |
|---|---:|---:|---:|
| Read bank-conflict counter | 2,883,584 | 2,359,296 | -18.18% |
| Write bank-conflict counter | 4,046,848 | 2,998,272 | -25.91% |
| Total BC | 6,930,432 | 5,357,568 | -22.70% |
| BC / useful BF16 MMA | 10.575 | 8.175 | -22.70% |
| Shared-load instructions | 1,429,504 | 1,429,504 | unchanged |
| Shared-store instructions | 1,343,488 | 1,343,488 | unchanged |
| Read BC / shared-load instruction | 2.01719 | 1.65043 | -18.18% |
| Write BC / shared-store instruction | 3.01220 | 2.23171 | -25.91% |
| KVD-to-TSM transaction bytes | 112MiB | 112MiB | unchanged |
| Ordinary KVD global-load bytes | 17MiB | 17MiB | unchanged |
| KVD global-store bytes | 98MiB | 98MiB | unchanged |
| Registers / stack / shared B | 242 /0 /45,568 | 242 /0 /45,568 | unchanged |
| Grid / CTA threads | 128 /128 | 128 /128 | unchanged |
| Achieved warps/CU | 7.10 | 7.09 | effectively unchanged |
| Eligible warps / WE active cycle | 0.25 | 0.25 | unchanged |

These are measured transaction bytes, not unique payload or HBM bytes.
DRAM reads are84,300,672 versus84,301,056B, effectively identical. Bank counters
have the same useful-work and request-count denominators here, unlike V16,
so the lower totals do not come from performing fewer matrix/shared operations.
The local conflict-free phase model was never a proof that the whole kernel
would have zero BC;5.36M counted conflicts remain.

Native per-PC evidence:

- Useful BF16 MMA655,360 and BF16 conversions1,310,720 are identical.
- Non-transposed SWZL393,216->524,288; transposed524,288->393,216.
  Exactly131,072 loads change orientation; total917,504 is unchanged.
  This agrees with8 targeted loads x512 launched warps x32 chunks.
- Native opcode sum14,680,704->14,649,472 (-0.213%). The distinct PU counter
  is14,746,240->14,715,008; do not silently equate these two metric scopes.
- `v.mov.v2s` falls32,768, while `s.wait` increases32,768; other descriptor/
  address changes yield the small net instruction reduction. A wait opcode
  is not a fixed number of stalled cycles, so this does NOT prove one cost
  exactly cancels the other.

Memory-dependency ratio1.19->1.02 and TSM-dependency0.74->0.69 decrease.
Sync0.51->0.53 and fetch0.21->0.24 increase. They are per-issue ratios,
**not additive/disjoint wall-time shares**. No exact microsecond allocation
to BC, waits or fetch follows from these values.

## What this falsifies, and the next discriminating change

The expectation that this particular B-intermediate conflict reduction
would materially accelerate the complete call is not supported. The counters
improve, but useful issue availability and the serial recurrence do not.
This does not prove all remaining BC is harmless, nor that AIU/SWZL is useless.

Every chunk still follows KH -> residual publication -> P@residual ->
scaled-V publication -> state update, with the next chunk depending on the
updated state. The layout change does not remove those products, publication
boundaries,32 chunk iterations, or the grid-limited supply of warps.

Next preferred isolated experiment remains V32/grid128 with8 instead of4
warps per CTA, keeping K/P/V staged once per CTA/chunk. It targets independent
work/issue supply without the V16 duplication of global-to-TSM inputs.
The ownership model exists, but **this kernel is not yet implemented or timed**.
Shared operand reads and synchronization can still grow: preserving global
payload is not a promise that all on-chip traffic stays fixed. Compile the
real body and measure resource/traffic/latency tradeoffs. Do not silently
combine B-layout and8 warps or call occupancy itself a win.

## Evidence binding and replay

Archive SHA256:
`580e0df3c019f7a8c9bcd7f6e31b320918ac063190dbacccf42c06d41d567330`.
607 files /606 checksums /131 source files verify against the capture SHA.
Tracked sources are clean; two loose operator reports are recorded as such.
Device UUID019ee024-8860-091c-0000-0000007aff6d.
Loaded device-library SHA256:
`bad6466a28860b42be159e5c0ff14f30f84175b7e7cb5341f1b125ab5b5f4994`;
binding SHA256:
`3bb5d765b984795c231717adc7aae84841679c60aa24822cf6b0725772b833b4`.

Thirty residual cases and30 B-layout cases x8 repeats, tail/GVA/initial-state/
output-only checks pass; B-layout is byte-equal to residual. Old16 WY cases
and32 delivery admissions also pass. Both g=-0.1/-1.0 are numerically admitted,
but only-1.0 is profiled. Independent2% oracle unchanged; API_TIMING=NOT_RUN.

Capture used `/sim/eec/shared/junfu.qx/asight/bin/acu`. Host-only local SDK
reimport reproduces selected raw counters and closes all15 per-PC opcode sums;
omitting an executed PC breaks each sum. Ambiguous duplicate device-attribute
rows from the text exporter are preserved but not used as numerical evidence.
No uploaded binary/source/command was executed locally.

Replay/evidence: `/workspace/gdn-residual-blayout-acu-analysis-20260924`
(`capture.tar.gz`, `analyze.py`, `reimport.sh`, `analysis-verified.log`,
`parsed.json` and per-PC exports).
Condensed machine-readable result:
[residual_blayout_acu_20260924.json](../dev/ppu/results/residual_blayout_acu_20260924.json).
