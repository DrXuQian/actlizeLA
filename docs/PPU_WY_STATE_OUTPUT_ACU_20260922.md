# State/output WY capture: instruction overhead remains in all three stages

## Verdict

The uploaded mask48 run is valid. Scalar prepare + tiled state/output wins
the registered **full-API** comparison against FLA at both supplied gates.
Nevertheless, all three device stages remain slower than FLA in the paired
ACU captures. These observations use different measurement protocols and do
not establish that our kernels are faster than FLA's kernels.

The next bounded target is **state's copy/address and gate-evaluation
instruction work**, not another occupancy increase, reduced precision, or
a different recurrence. Prepare and output have independently measured
instruction gaps too. Keep the original route, scalar and all-tiled controls,
numerical criteria and automatic routing unchanged.

## Verified evidence

- Archive: `gdn-qsa-acu-20260922T080504Z-2605202.tar.gz`.
- SHA256: `9511c6c63e7baf73a726f4b335f44bea263718a51aa526a5eb123f31ea523e84`.
- 542 unique regular files; all 541 checksums and the exact manifest
  denominator verified. No archive code or device binary executed locally.
- Measured binary source: `90eafeb80f0b7e77e02bfc4a32333a26c202498d`, empty
  source diff. Both loaded WY DSOs match the preceding comparison manifest:
  binding `cea8bf102302a0afc9fbb7bfd634184dfad43970e9ae6a71146abad2a4dc9b47`,
  library `304401414a9e25baf1c03f4d73fa0905bce12a8d9d564f0a3669eeaef26809a2`.
- Capture: B1/S2048/Hk16/Hv32/K128/V128/C64, g=-1.0, BF16 inputs,
  zero initial state, FP32 WY/FLA final state. Same input digest
  `5b1677989c9b708f`, oracle, outputs, FLA source identity and torch2.9.0.
- Both capture receipts identify PPU-ZW810, 72 CUs, UUID
  `019ee024-8860-091c-0000-0000007aff6d`. All ten actual kernels report
  CE1.700 GHz and DRAM1.800 GHz. The preceding comparison's measured
  properties string also contains this same UUID; it matches the receipts.
  This run has stronger identity evidence than the earlier generic warning
  about a properties-only record without a dedicated UUID field.
- Actual WY kernels: `gdn_wy_prepare<false>`, `gdn_wy_tiled_state`,
  `gdn_wy_tiled_output`. There is no accidental tiled-prepare/mask56 capture.
- Preflight6 calls and subject1 call are separate. Subject warmup0, no
  verification device kernels inside ACU. Installed FLA0.6.0/Triton3.4.0
  source snapshots and parser-only CUDA13 compatibility identity agree with
  the preceding comparison.
- Included API comparison: **2 gates x 8 roles x 16 samples**, 10 calls per
  sample, five warmups. Recomputed every median and direct envelope verdict.
  Numerical log: 16 cases x 6 WY variants, including 80 candidate admissions
  with scalar raw equality, eight repeats and GVA/output-only checks. Both
  capture errors are `[0.0063291085, 0.0036524015]`, within the unchanged2% gate.

Local evidence is retained at
`/workspace/gdn-wy-state-output-acu-analysis-20260922/`: safe archive parser,
`parsed.json`, analysis log, native reimports and measured per-PC exports.
`analyze_bundle.py` SHA256:
`0571cb7651687b5ed79b517171d3c14950a87b42116d47ba3696b47f497dc04d`.

## Full-API verdict, including the newly supplied weak case

Microseconds, unprofiled full-public-API event spans. No samples trimmed.

| Role | g=-0.1 median [min,max] | g=-1.0 median [min,max] |
|---|---:|---:|
| Original | 952.442 [917.168,986.292] | 478.898 [425.924,517.936] |
| Scalar WY | 720.480 [713.948,730.228] | 710.210 [705.772,722.768] |
| Tiled prepare | 771.690 [764.832,780.264] | 767.866 [758.656,779.024] |
| Tiled state | 458.826 [455.464,465.844] | 458.998 [453.540,465.760] |
| Tiled output | 703.704 [696.876,712.272] | 693.508 [688.492,699.056] |
| **Tiled state+output** | **443.810 [440.356,448.788]** | **439.486 [437.684,444.076]** |
| Tiled all | 451.954 [445.056,455.492] | 446.700 [443.052,454.620] |
| FLA | 770.394 [492.972,954.344] | 730.812 [486.836,968.548] |

The pair wins against scalar, state-only and FLA at both gates under the
unchanged disjoint-observed-envelopes rule. It wins against original only
at the weak gate; strong is UNRESOLVED. Pair versus all is UNRESOLVED at
both gates. Original final state is BF16, not WY/FLA's FP32: preserve that
comparison boundary and the original strong/reset route.

FLA's wide distributions remain visible. The 1.7359x/1.6629x median ratios
are descriptive of these runs, not established stable kernel speedups.
Do not infer the cause of FLA's API variability from the profiler run or
subtract profiled kernel sums from API medians to label host overhead.

## Matched ACU stages

Times below are **profiled device durations**, not API latency. Prepare's
FLA counterpart is prefix cumsum + KKT/solve + W/U. Its two fill kernels
are accounted for separately.

| Stage | WY us | FLA us | WY/FLA | Absolute gap us | WY executed instructions | FLA executed instructions |
|---|---:|---:|---:|---:|---:|---:|
| Prepare | 161.33059 | 83.07294 | 1.942x | 78.25765 | 46,814,208 | 17,228,800 |
| State | 199.59765 | 90.02412 | 2.217x | 109.57353 | 34,386,048 | 10,040,320 |
| Output | 63.96294 | 43.64294 | 1.466x | 20.32000 | 21,836,800 | 9,166,848 |

Instruction column is `pu__inst_executed.sum`. FLA prepare constituents are
2.81647/43.87235/36.38412 us. Fills are 4.33882/2.16471 us. Profiled sums,
including those fills: WY 424.89118 us, FLA 223.24353 us. No claim that these
are complete-call latencies or a throughput result outside this shape.

| Stage | Grid x threads, WY / FLA | Registers, WY / FLA | Shared bytes, WY / FLA | Achieved warps/CU, WY / FLA |
|---|---|---:|---:|---:|
| Prepare / FLA solve | 1024x128 / 1024x32 | 84 / 256 | 70,144 / 6,144 | 11.33 / 13.72 |
| Prepare / FLA WU | 1024x128 / 1024x256 | 84 / 100 | 70,144 / 25,600 | 11.33 / 37.17 |
| State | 128x128 / 128x128 | 232 / 256 | 49,408 / 50,432 | 7.10 / 7.09 |
| Output | 1024x256 / 1024x256 | 98 / 170 | 49,408 / 49,152 | 37.69 / 22.58 |

All listed stacks are 0. State no longer has the previous two-warp mapping:
matching grid, useful warp count and state store footprint did **not** close
its gap. Output already has *more* achieved warps than FLA. Simply asking for
more occupancy does not explain or remove either relative slowdown.

### Arithmetic and traffic constraints

| Quantity | WY prepare / FLA math prepare | WY state / FLA state | WY output / FLA output |
|---|---:|---:|---:|
| BF16 MMA instructions | 344,064 / 344,064 | 524,288 / 524,288 | 524,288 / 524,288 |
| TF32 MMA instructions | 98,304 / 32,768 | 0 / 0 | 0 / 0 |
| KVD global-store request MiB | 512.25 / 60.125 | 98 / 98 | 16 / 16 |
| DRAM read bytes | 25,446,016 / 49,452,160 | 92,555,008 / 92,551,552 | 67,385,472 / 67,411,968 |

KVD metric is `kvd__bytes_pipe_lsu_mem_global_op_st.sum`, **not DRAM traffic**.
The scalar prepare's narrow-store request footprint has indeed returned to
512.25 MiB. Its KVD-to-L2 stores are 32.25 MiB and measured DRAM writes 128 B;
calling the 512.25 MiB HBM traffic would be wrong. The previous vector-store
experiment did not establish an isolated performance win: do not repeat it
as if fewer request bytes guarantee the required speedup.

State store instructions also match: 102,400 warp instructions per side.
Asynchronous global-to-shared request bytes are 144 MiB versus 165.478515625 MiB;
ordinary global load request bytes are 1 MiB versus 2 MiB. Our state is not moving
three times as much global data. Its measured shared bank-conflict counts
are lower too, so a blanket "more bank conflicts" explanation is unsupported.

Prepare's TF32 difference is intentional: `tf32_product` uses high/high and
two high/residual terms; FLA's captured solve uses plain TF32. Removing those
terms changes precision and is not an address-only optimization. Keep the
existing BF16 boundaries and FP32 state in every delivery experiment.

## Dynamic opcode ledger and one localized hot loop

Recovered **measured** execution counts per PC from both native reports
using ACU's host-side `HgRules` interface. Not static instruction counts
multiplied by guessed loop trips. Every per-PC sum equals the profiler's
`sass__inst_executed_per_opcode` aggregate. Deliberately dropping an executed
PC fails closure for all ten kernels. Native scalar metrics also agree with
the archived text exports.

SASS totals differ slightly from `pu__inst_executed.sum` on WY. Keep these
counter definitions separate; no fabricated equality or adjustment:

| Stage | WY SASS sum | FLA SASS sum | Net extra SASS instructions |
|---|---:|---:|---:|
| Prepare | 46,582,784 | 17,228,800 | 29,353,984 |
| State | 34,204,288 | 10,040,320 | 24,163,968 |
| Output | 21,560,320 | 9,166,848 | 12,393,472 |

Selected exact opcode deltas, **WY minus matched FLA**:

| Opcode | Prepare | State | Output |
|---|---:|---:|---:|
| `v.madl.i32` | +7,172,096 | +6,140,416 | +3,096,576 |
| `v.shll.b32` | +2,538,496 | +2,077,184 | +958,464 |
| `v.shrl.b32` | +1,973,248 | +2,010,624 | +991,232 |
| `v.shra.i32` | +1,884,160 | +1,199,616 | +573,440 |
| `v.and.b32` | +1,612,800 | +1,625,600 | +720,896 |
| `s.mov.b32` | +2,108,416 | +1,485,824 | +1,202,176 |
| `v.mov.v2s` | +589,824 | +756,736 | +851,968 |
| `s.wait` | +1,682,432 | +411,136 | +468,992 |
| `v.fma.f32.rtte` | +1,257,472 | +835,584 | +8,192 |

This is a subset, not an exhaustive sum; negative deltas are retained in
the local ledger. `v.madl` often uses multiplier1 in these address sequences;
the mnemonic does not mean all these instructions are integer multiplications.

The tiled-state **W staging loop** is a concrete first target. In this
report its PCs `0x55809eb8` through `0x5580a040` comprise 50 static instructions,
including one 16-byte-per-lane `vmem.ld.tsm.zfill.b32x4.kp1`. The measured loop
sum is **6,537,216 instructions**, 19.11% of state's SASS sum. The copy itself
executes 131,072 times, exactly:

```
128 CTAs x 4 warps x 32 chunks x 8 copy iterations = 131,072
```

Source: `gdn_wy_common.cuh::stage<64,128,128>` and
`shared_copy.cuh::RowLayout`. The generated loop repeatedly performs signed
coordinate decomposition: sign shifts, power-of-two division/remainder
corrections, cube/swizzle arithmetic and execution-mask loop control. The
shared matrix-load instructions themselves are not the whole cost. The K
copy has the corresponding high-frequency address sequence, with a tail
predicate, and U/publication have smaller loops. This identifies real work
to remove, **not 6.54M guaranteed removable instructions or a 19% time saving**.

There is a separate gate-evaluation difference: state executes 278,528
`v.exp2.f32` versus 49,152 for FLA. Our source calls `expf` for every value
fragment slot (16 per-lane slots, hence 16 warp-level evaluations per chunk)
plus decay; FLA broadcasts gate
factors over value columns. Several extra FP32 min/max/FMA operations belong
to the general exponential lowering. First reuse identical row factors;
do not silently replace the exponential approximation or its domain/rounding.
Output already hoists its QH row factor, and its exp2 count is actually
*lower* than FLA's; this is not a universal explanation for all three stages.

Warp-state counters are cycles/issue, not percentages of wall time. For
example our state has *lower* memory-dependency/sync ratios than FLA while
executing far more instructions. Summing those ratios into a time budget
would obscure the measured instruction-work difference.

## Next bounded plan; no implementation or routing change in this review

1. **State-only address/copy candidate.** Preserve the current 4-warp tile,
   shared layout, vector width, tail zfill and every numerical operation.
   Hoist invariant cube offsets; use proven nonnegative tile coordinates and
   compile-time copy iteration offsets instead of repeated signed division
   inside generic rolled loops. Exhaustively prove old/new offsets and tail
   predicates, then inspect the actual generated body. Do not infer success
   from a pragma or merely moving C++ expressions. Keep instruction footprint,
   registers/spills and full-state correctness visible.
2. **Separate state row-factor reuse candidate.** Evaluate the same `expf`
   arguments once per unique owned row and reuse them. Preserve evaluation
   semantics, BF16 conversion boundaries, FP32 recurrence and K order; raw
   equality remains required. Keep this independent from the copy candidate
   so its instruction/time effect can be attributed.
3. **Then propagate only proven address changes to output and prepare as
   separate ablations.** Their extra instruction counts are established, not
   assumed to vanish with state. Prepare's solve precision and 70 KiB fused
   lifetime need their own later decomposition decision; do not lower TF32
   precision to make a delivery comparison appear faster.

Numerical admission remains 16 cases, independent 2% oracle, candidate/scalar
raw equality and 8 repeats. Compare weak and strong full-API samples with all
existing controls under the unchanged envelope rule. Kernel instruction
reductions without latency reduction are a valid losing result, not grounds
to alter the criterion. No new box run is required to read this archive.
