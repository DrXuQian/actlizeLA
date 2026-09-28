# First-chunk migration: correct, mixed timing, no promotion

Measured source `21cee5e5cdfc51a5251a8e66f53cea5069373029`; this is a result
record, not an amendment to [the preregistered experiment](PPU10_FIRST_CHUNK.md).
The current goal is architecture-independent SM90 migration coverage, not
the suspended 1.5x FLA target. **Device correctness closes this migration's
implementation; the observed timing does not admit it into the default.**

## Evidence and scope

Upload `first-chunk.tar.gz`, SHA256
`84a8df044175784bb2669a5ce07e04a1c1b63232de8a0e9a9f467941d4df994d`.
Outer manifest and both 799-file inner manifests validated; 324 source files
per gate match the measured git revision. No uploaded code was executed.
Only local report import, native inspection and read-only validation ran.

- PPU-ZW810, 72 CUs, UUID `019ee024-8860-091c-0000-0000007aff6d`.
- Driver `2.1.2-r7b50d071022`, HGCC `2.2.0-dev` (Jun 3 2026), Torch 2.9.0 /
  runtime 12.9. All 30 captured kernels report CE 1.700 GHz.
- B1/S2048/Hk16/Hv32/K128/V128/C64; BF16 inputs, FP32 state, initial=None,
  final state enabled. Both scalar log-gates, -1 and -0.1, use paired inputs.
- Capture used `/sim/eec/shared/junfu.qx/asight/bin/acu`; SDK ACU was used
  locally only to import the saved reports and export counters, not capture.
- Complete public calls contain control/candidate/FLA **4/4/7 kernels**.
  FLA's two fill kernels are included. Two cells, 30 kernels total, none missing.
- Each cell is **one ACU capture per arm**, not a repeated timing distribution
  or API-event median. The raw counter/opcode sum is reconciled with per-PC
  execution counts; our PCs are bound to actual uploaded native code.

The binary and compact evidence hashes are in
[`first_chunk_acu_20260928.json`](../dev/ppu/results/first_chunk_acu_20260928.json).
The complete extracted evidence, audit scripts and per-PC exports remain in
`/workspace/actlizeLA-migration-20260928/ppu10-tuning/first-chunk/acu-analysis-20260928/`.

## Numerical admission

Both full-chunk and first-chunk deliveries pass **36 cases x8 repeats**:
independent unchanged 2% output/state oracle, RAW-BIT against the residual
control, tails, supplied state, GVA, output-only and input immutability.
An additional **16 cases x8** covers S64/65/128/2048, both gates, None versus
an explicit positive-zero state, and signed zeros in Q/K/V. None/explicit
zero fingerprints agree. No tolerance or routing changed.

The same-box compiler's 40 old native instruction/operand sequences and
resource/ABI records remain identical to the parent upload. Both new native
guards, the linked host/binding and unchanged publication are verified.
Missing-kernel, old-body-as-candidate, device-mismatch, missing/duplicate-cell,
old-MMA-count and corrupt-zero-counter negative controls reject.

## Complete-call ACU times

Microseconds; positive delta means slower. FLA is context, not a new target.

| Gate | Control | First-chunk | Delta | FLA |
|---|---:|---:|---:|---:|
| -1.0 | 167.75058 | 169.61707 | +1.86649 / +1.113% | 225.27236 |
| -0.1 | 168.95765 | 168.27294 | -0.68471 / -0.405% | 225.25823 |

| Stage | -1 control -> first | -0.1 control -> first |
|---|---:|---:|
| Prefix, unchanged | 2.37235 -> 2.35118 | 2.57000 -> 2.44353 |
| Solve, unchanged | 44.26176 -> 44.52059 | 44.24000 -> 44.14647 |
| State | 79.27000 -> 79.70059 | 78.92824 -> 78.36588 |
| Output | 41.84647 -> 43.04471 | 43.21941 -> 43.31706 |

Output is slower in both captures; state has mixed signs. Unchanged stages
contribute +0.23766/-0.22000 us variation. Do not average away the strong-gate
regression or claim these single captures establish a small stable gain/loss.
Registered verdict: **MIXED_SIGN_NO_PROMOTION; retain the control**.

## The omitted work is real, but the instruction saving is not

The following execution counts are identical at both gates. Matrix counts
exactly match the dimension-derived predictions, so this is not a missing
candidate/binding or an ineffective guard.

| Counter | State control -> first | Output control -> first |
|---|---:|---:|
| BF16 MMA | 655,360 -> 647,168 | 524,288 -> 516,096 |
| Total executed opcodes | 12,153,984 -> 12,308,608 | 11,069,440 -> 11,448,320 |
| Total instruction change | +154,624 / +1.272% | +378,880 / +3.423% |
| Static native sites, box HGCC | 1,098 -> 1,044 | 1,408 -> 1,428 |
| Registers/thread | 120 -> 122 | 70 -> 70 |
| Stack bytes | 0 -> 0 | 0 -> 0 |
| Shared allocation bytes | 46,080 -> 46,080 | 49,408 -> 49,408 |
| Shared bank-conflict counter | 4,440,064 unchanged | 1,179,648 unchanged |
| Global-to-shared staging bytes | 117,440,512 unchanged | 83,886,080 unchanged |

Whole-call executed opcodes increase 35,019,904 -> 35,553,408. State becoming
statically shorter does not establish a dynamic instruction reduction.

Delta accounting (candidate minus control):

| Opcode | State | Output |
|---|---:|---:|
| v.mov.b32 | +262,144 | +245,760 |
| v.mov.v2s | +118,784 | +190,464 |
| s.wait | -281,600 | -10,240 |
| s.mov.b32 | +5,120 | -65,536 |
| s.cmp.eq.i32 | +32,768 | +8,192 |
| s.cbr.nz | +30,720 | +16,384 |
| s.cbr | +23,552 | 0 |
| s.and.b32 | included below | +16,384 |
| tsm.ld.swzl.b32x4.s0.t1.trans0 | -16,384 | -12,288 |
| v.mma.f32.bf16.m16n16k16 | -8,192 | -8,192 |
| v.reg.dchk | included below | -2,048 |
| Remaining net opcodes | -12,288 | 0 |
| **Sum** | **+154,624** | **+378,880** |

### Native duplicate initialization, not a layout conversion claim

Actual HGCC 2.2.0-dev code has zeros both before and inside the new history
guard, writing the same accumulator registers:

- State: guard PC `0x1e48`; eight accumulator zeros before it execute 32,768
  times/site. The repeated zero sites `0x1e80,1e90,1ea0,1eb0,1ec0,1ec8,1ed0,1ed8`
  execute 31,744 times/site: 128 CTAs x8 warps x31 nonfirst chunks.
  This adds 253,952 zero instructions; net extra setup moves add 8,192,
  closing the +262,144 `v.mov.b32` delta.
- Output: guard PC `0x2538`; 16 accumulator zeros at `0x24b8..0x2530`
  execute 16,384 times/site. Sixteen repeated zero sites in `0x2550..0x25e0`
  execute 15,872 times/site: 992 nonfirst CTAs x8 warps x2 panels.
  This adds 253,952 zero instructions; removing one 8,192-execution setup
  move closes the +245,760 delta.
- The output history region also contains 12 address `v.mov.v2s` sites,
  each executed 15,872 times, exactly the +190,464 delta. Their scalar
  destinations feed `tsm.ld.swzl` addresses; e.g. PC `0x2540` produces
  sreg8 consumed at `0x2568`. This is not evidence of a new fragment-layout
  repair. AIU.swzl and ld.swzl remain paired, and source layout is unchanged.

These are directly observed instruction costs. They do not assign each
microsecond of timing change to one opcode, prove a specific compiler-pass
cause, or justify disabling a compiler initialization safety option.
Lower memory-dependency ratios also are not wall-time savings by themselves.

## Disposition and next boundary

Mark first-chunk **implemented + device-validated + measured**, retain the
opt-in counterfactual, and do not promote it. No new box run is required to
classify this upload. Short-sequence speed remains unmeasured.

If revisiting this small S2048 saving later, the discriminating experiment is
first-chunk peeling or branch-local accumulator lifetime, with a native gate
requiring the duplicated zeros/address setup to disappear. It is not done
here, and changing numerical scope or deleting snapshot publication is not
an acceptable substitute.

The next migration item remains interior-full/final-tail partitioning. Then
resolve inverse register retention, paired-conversion applicability and
shape-driven public selection; no claim that all portable transfers are done.
