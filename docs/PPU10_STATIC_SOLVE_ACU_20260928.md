# PPU1.0 static solve: measured improvement, not yet 1.5x FLA

Source `77fee23e71e2f0dde68e8594ec884eabe4a8822b`, experimental branch
`agent/ppu10-solve-gate-cache-20260928`. This closes the device measurement
of [the preregistered composition](PPU10_GATE_CACHE_STATIC_SOLVE.md).
Default dispatch and SM90 are unchanged.

## Complete forward, both decay regimes

Shape: **B1/S2048/Hk16/Hv32/K128/V128/C64**, BF16 inputs, FP32 state,
zero initial state and final state enabled. Each cell is one complete ACU
capture, **not** a median or public-API event span. Count every kernel:
control 4, candidate 4, FLA 7, including its two fills.

| Gate | Gate-cache control | + static solve | FLA | FLA / candidate | Target FLA / 1.5 | Remaining |
|---|---:|---:|---:|---:|---:|---:|
| Strong, g=-1 | 185.71001 us | **180.07000 us** | 224.47999 us | **1.24663x** | 149.65333 us | 30.41667 us |
| Weak, g=-0.1 | 187.10236 us | **181.19353 us** | 226.11882 us | **1.24794x** | 150.74588 us | 30.44765 us |

Observed full-call reduction: **3.04% / 3.16%**. Retain the candidate for
confirmation and subsequent experiments. One capture per arm/gate does not
establish variance, a universal winner or automatic production admission.
The 1.5x goal is **NOT REACHED** in either regime.

| Stage | Strong control → candidate (us) | Weak control → candidate (us) |
|---|---:|---:|
| Prefix, unchanged symbol | 2.60824 → 2.47588 | 2.47588 → 2.79235 |
| Solve, only changed stage | **52.50118 → 44.87588** | **52.48118 → 44.95353** |
| State, unchanged symbol | 87.69765 → 89.06412 | 89.32824 → 89.56824 |
| Output, unchanged symbol | 42.90294 → 43.65412 | 42.81706 → 43.87941 |

Solve saves 7.62530 / 7.52765 us (**14.52% / 14.34%**). Unchanged stages
contribute +1.98529 / +1.61882 us of variation, leaving a whole-call saving
of 5.64001 / 5.90883 us. Their complete dynamic opcode dictionaries are
identical between arms; do not attribute the variation to changed state/output
code. Cache/arrival effects and ordinary capture variation are not separated.

FLA's seven stages (strong / weak, us): cumsum 2.86353 / 2.53294;
BF16 fill 4.37647 / 4.34235; KKT+solve 44.48412 / 44.81765;
W/U 35.84941 / 35.66941; FP32 fill 2.26882 / 2.23706;
state 89.29588 / 91.30588; output 45.34176 / 45.21353.
Our residual state also performs work FLA materializes in W/U; their state
rows alone are not identical work.

## Mechanism: indexing/control removal, not BC or precision reduction

These counts are identical in both decay captures. Per-PC executions sum to
ACU's built-in `sass__inst_executed_per_opcode`. Solve opcode PCs and matrix/
FMA operands bind to the uploaded ISA; 16 unreported trailing alignment nops
are explicitly accounted for.

| Solve measurement | Control | Static |
|---|---:|---:|
| Native static sites, including padding | 1,678 | 1,472 |
| Dynamic opcode total | 14,767,104 | 11,416,576 |
| Dynamic diagonal-region instructions | 5,013,504 | 1,658,880 |
| Diagonal indirect-register reads | 196,608 | **0** |
| Diagonal `s.wait` with `pipe_flush` | 466,944 | **0** |
| All `s.wait` instructions | 1,637,376 | 1,043,456 |
| Diagonal ordered FP32 RTTE FMAs | 491,520 | 491,520 |
| BF16 / TF32 MMA | 81,920 / 98,304 | 81,920 / 98,304 |
| Registers / uninstrumented stack | 84 / 0 B | 84 / 0 B |
| Launch shared memory | 49,664 B | 49,664 B |
| Shared bank conflicts | 2,408,448 | 2,408,448 |
| KVD → TSM transaction bytes | 16 MiB | 16 MiB |

Dynamic instructions fall **22.69%**. The diagonal loses 3,354,624
instructions; outside it the net change is +4,096, closing the whole-solve
delta of -3,350,528. The same 120 coefficient words per lane-context are read
using fewer, wider loads. The three-product TF32 inverse, FP32 FMA order,
BF16 rounding boundaries and output stores remain intact.

This is **not a BC reduction or occupancy gain**: register/shared/warp block
limits remain 12/5/16. Achieved solve warps/CU are 18.99→18.95 (strong),
18.91→19.02 (weak). Fetch-stall per-issue ratio falls .54→.38/.40, but sync
rises .98→1.26 / .97→1.25 and memory dependency rises .55→.63/.62 while the
kernel gets faster. These changing-denominator ratios are not additive
wall-time shares. `pu__inst_executed.sum` is a separate counter:
14,920,704→11,570,176; do not substitute its scope for opcode sums.

## Identity and admission

- Both archives: **762/762 checksummed files**, exact manifest denominator;
  287 source snapshot files match `77fee23` in each.
- Same PPU-ZW810, 72 CU, UUID `019ee024-8860-091c-0000-0000007aff6d`,
  64 MiB L2. All 30 kernels (4+4+7, twice) report CE **1.700 GHz**.
- Capture: `/sim/eec/shared/junfu.qx/asight/bin/acu`,
  `v2.0.0_20251231-4f7cd70`, data version 12006. Local host-only re-import
  used the SDK reader; it executed no GPU work or uploaded code.
- Compiler: **HGCC 2.2.0-dev, built June 3 2026**. Driver:
  `2.1.2-r7b50d071022`; `ppu-smi` SDK banner: `2.0.2-d2568e060312`.
  Torch 2.9.0/runtime 12.9. These are separate identity fields, not an
  inferred SDK 2.1.1 version.
- Device library SHA256:
  `18c106b392e3cc4d2a55832b35a83f258064bc36085c35306b10f4d7c20deee5`.
  Binding SHA256:
  `dd1032d9dddbe62043ca721fb26afb2301eb82655487b664f36f4d95cc95d357`.
  Both gates and PPU arms load this same pair.
- **30 residual cases + 30 per delivery**, each eight repeats: unchanged
  independent 2% oracle plus RAW-BIT equality to scalar residual PASS;
  tails, nonzero state, GVA, variable metadata and output-only covered.
  Existing WY 16-case suite also passes. Captured fingerprints match
  independent preflight and admission on the same inputs.
- Strong output/state errors: .00632911 / .00365524; weak:
  .00485437 / .00293219. No tolerance adjustment. Residual-to-old-WY
  RAW-BIT equality is not claimed: they have different associations.
- Replayed native negatives reject old indirect indexing and missing math;
  analysis negatives reject a missing kernel, old solve relabeled as static,
  different device, and an omitted executed PC.

Two provenance/inspection limitations remain explicit:

1. Runner `source.diff` is saved **before** `git submodule update`: it records
   pre-bootstrap actlize `e893892`, then restores pin `423253c` before compiling;
   capture records that pin. The stale diff does not prove `e893892` was
   compiled. The separate build-time submodule receipt/header closure is not
   bundled. Retain the diff and ordering explanation. No dependency was
   changed during analysis.
2. Local GNU objdump 2.38 mislabels this binary's IBT `.plt.sec`/`.plt.got`
   targets, making the original host-call checker fail. Reading ELF
   JUMP_SLOT/GLOB_DAT relocations resolves 87 entries; the unchanged strict
   checker then proves regular/static solve selection and common state/output.
   Wrong-call and missing-entry negatives still fail. No uploaded binary is
   executed and no required call is waived.

Only 4/37 uploaded device bodies match the local SDK build, including both
solves. Actual box evidence is the resource/performance authority; other
stages are not claimed identical across toolchains.

## Next bounded experiment

Keep static solve as the opt-in experimental parent; reconfirm against
gate-cache when profiling the next candidate. State is now the largest stage,
**89.1–89.6 us**, about half the call. Returning to old solve or changing its
precision would not address that stage.

Next local screen: specialize state for **full C64 chunks**, removing generic
tail/address work only if the actual native code still contains it. Preserve
generic tails, coverage, precision, publication and AIU/SWZL contracts.
Reject the screen if the compiler already removed that work or if it merely
trades it for spills/duplicated traffic. Do not rebrand the rejected metadata
lookahead as this experiment. This is a testable next axis, **not a promise
of another 30.4 us**. Closing that gap through state alone requires roughly
59 us there, another 34% reduction.

Machine result: [static_solve_acu_20260928.json](../dev/ppu/results/static_solve_acu_20260928.json).
Raw archives, audit/import scripts, native PC exports and full parsed metrics:
`/workspace/actlizeLA-migration-20260928/ppu10-tuning/acu-analysis-20260928/`.
Archive SHA256 strong:
`b437a6184cd927b294a034dea306df709cc646d32498b9ba81e00cb22bb507f9`;
weak: `4615c9df964a094d308270c43bc54ccc7d3ca0650ad83bba004ac9c722a88fa3`.
