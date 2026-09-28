# PPU1.0 paired NewV conversion: device verdict

Measured source `8b20c126730e022f9123cff3296a24ef047ccf8a`. The rule in
[PPU10_PAIRED_CONVERSION.md](PPU10_PAIRED_CONVERSION.md) was registered before
the candidate edits. Both registered cells are present and valid.

**Correctness and migration applicability are closed. Both complete-call
captures are slightly lower, but a stable speedup is not established. Keep
the candidate opt-in for repeated confirmation; default routing stays fixed.**
The tiny observed differences are not a reason to substitute a new default
or to call the paired instruction twice as fast.

## Complete-call ACU results

B1/S2048/Hk16/Hv32/K128/V128/C64, BF16 inputs, FP32 state, initial=None.
Times below are microseconds, summing all actual kernels: four control,
four candidate, seven FLA, including both FLA fills. These are not API-event
spans or a sum that omits setup kernels.

| Gate | Full-chunk control | Paired conversion | FLA | Candidate latency change |
|---:|---:|---:|---:|---:|
| -1.0 | 170.48530 | 169.95706 | 228.44646 | -0.310% |
| -0.1 | 169.48530 | 169.37412 | 225.61353 | -0.066% |

The registered classification is `LOWER_BOTH_PENDING_REPEATS`, not a
statistically established performance win. There is one ACU capture per
arm/gate, not a repeated latency distribution. The retained control is
1.340x/1.331x FLA here; the suspended 1.5x target is not an admission rule.

| Gate | Stage | Control | Candidate | Candidate minus control |
|---:|---|---:|---:|---:|
| -1.0 | prefix | 2.36412 | 2.35059 | -0.01353 |
| -1.0 | solve | 44.67059 | 44.64882 | -0.02177 |
| -1.0 | state (changed) | 80.02588 | 79.87824 | -0.14764 |
| -1.0 | output | 43.42471 | 43.07941 | -0.34530 |
| -0.1 | prefix | 2.54882 | 2.52765 | -0.02117 |
| -0.1 | solve | 44.48824 | 44.34647 | -0.14177 |
| -0.1 | state (changed) | 79.76059 | 79.58529 | -0.17530 |
| -0.1 | output | 42.68765 | 42.91471 | +0.22706 |

Unchanged stages contribute -0.38060/+0.06412 us of variation. At g=-1,
72.1% of the observed total reduction comes from unchanged stages, whose
native bodies and executed opcode dictionaries are identical. Do not credit
that variation to the conversion edit. The changed state's observed
reductions are only 0.184%/0.220%; these captures do not establish their
repeatability or identify a cycle-level critical-path saving.

## The native pair executes; it does not remove a pair's delivery cost

The actual candidate is `gdn_wy_paired_conversion_state<true>`. Each of its
eight `v.pcnvt.bf16x2` sites executes 32,768 times, for 262,144 pair
instructions. Both gates have the same execution-count changes:

| State opcode/work | Control | Candidate | Change |
|---|---:|---:|---:|
| Scalar `v.cnvt.bf16.f32.rtte` | 1,310,720 | 786,432 | -524,288 |
| `v.pcnvt.bf16x2` | 0 | 262,144 | +262,144 |
| `v.shrl.b32` | 12,288 | 274,432 | +262,144 |
| BF16 MMA | 655,360 | 655,360 | 0 |
| BF16 shared stores | 1,310,720 | 1,310,720 | 0 |
| CTA barrier instructions | 132,096 | 132,096 | 0 |
| Total executed per-opcode instructions | 12,153,984 | 12,121,216 | -32,768 (-0.270%) |

The paired conversion plus high-half extraction exactly replaces the
524,288 scalar conversion instructions. The remaining instruction change is
`s.wait -37,888`, `s.mov.b32 +4,096`, `smem.ld.b32 +1,024`: net -32,768.
Do not describe this as a 2x reduction of the whole conversion/delivery chain.
The number of BF16 values is preserved:
`786432 + 2*262144 = 1310720`.

This is the expected cost of the current PPU owner/layout contract: the
same-lane NewV pair has scattered shared destinations. The upper half needs
extraction, and the original 16-bit publishers remain. No shuffle, new
rounding boundary, rearrangement pass, or changed matrix work is hidden in
the result. Source/native operand tracing and device RAW-BIT checks agree.

Actual box HGCC 2.2.0-dev resources differ from the earlier local SDK2.1.1
compile facts. Full-state registers are **120 -> 122**, not 120 -> 120;
native static sites are 1098 -> 1100, not 1085 -> 1086. Per-PC capture covers
1082/1084 sites; the omitted sixteen sites are trailing padding in each body.
Generic candidate/control each have 1181 static sites and 124 registers.
Both candidates have zero stack. The local record is retained as local
evidence, not silently substituted for the box's compiler output.

Shared allocation stays 46,080 B; register/shared/warp resource limits stay
4/5/8 CTA slots. Active warps/CU are 14.12 -> 14.12 at g=-1 and
14.12 -> 14.13 at g=-0.1; eligible warps/WE stay 0.35. Total shared bank
conflicts remain 4,440,064 (read 3,014,656, write 1,425,408), with the same
11,239,424 shared-read and 2,703,360 shared-write transactions. State KVD
load/store/staging bytes stay 18/50/112 MiB. This is not an occupancy,
traffic-volume or bank-conflict improvement.

The opcode sum reconciles with `sass__inst_executed_per_opcode`. The distinct
`pu__inst_executed.sum` metric is 12,317,824 -> 12,285,056; do not merge the
two denominators. Small per-issue stall-ratio changes are not disjoint
wall-time shares or proof of the cause of a sub-percent timing difference.

## Admission and provenance

- 36 retained cases x8 and 256 all-extent/gate/initial-state cases x8 pass
  RAW-BIT residual equality and the unchanged independent 2% recurrence
  criterion. Actual case keys, both control/candidate rows and repetitions
  were reconciled, not just the printed PASS trailers. Both full and generic
  device entries execute the new conversion.
- Two cells, 30 kernels; all nine outer and 833 entries per inner manifest
  verify. All 358 captured source files per cell match the measured SHA.
- Same binary/device/fixture and preflight/subject receipts in each pair;
  FLA source hashes verified. All 30 kernels report 1.700 GHz.
- All 44 old native instruction/operand/resource records match the previous
  same-box-compiler build. Uploaded full/generic native bodies and linked
  dispatch pass the local read-only audit and its source/native/link plants.
- Eight analysis negatives reject a missing kernel, an old state under the
  candidate identity, a wrong device, and missing/duplicate cells.

Device: PPU-ZW810, 72 CU, UUID `019ee024-8860-091c-0000-0000007aff6d`.
Actual HGCC compiler: 2.2.0-dev. Torch2.9.0/runtime12.9; FLA0.6.0 and
Triton3.4.0, native GVA, backend override disabled, existing process-local
CUDA13 version-parser backport. Native audit subrecords saying `NOT_RUN`
refer to the analyst not executing device code; they do not supersede the
uploaded numerical device admission.

Capture used `/sim/eec/shared/junfu.qx/asight/bin/acu`. Local SDK ACU only
imported those reports to recover per-PC counters; it launched no device
work and executed no uploaded source or binaries.

Complete retransmitted archive SHA256:
`b11c8345defeed8852bc68a06d12c9f2e468041a029596d064df481031ad3493`.
Library SHA256:
`a6c2fc32a5def6fa8bcfc5580457a17ed1b2f33a67c366d242cf6ad019cc3da7`.
The earlier 5,242,880-byte truncated upload was not used for any verdict.
Structured record:
[`paired_conversion_acu_20260928.json`](../dev/ppu/results/paired_conversion_acu_20260928.json).
Raw imports and auditor are under
`/workspace/actlizeLA-migration-20260928/ppu10-tuning/paired-conversion/acu-analysis-20260928`.

Migration completeness can advance without declaring a stable speed win:
this mechanism is implemented, device-correct, and measured. Keep it opt-in;
the two remaining items are PPU-specific geometry selection and unified
public selection. Any default promotion still needs its own non-regression
admission, not the mere fact that this experiment ran.
