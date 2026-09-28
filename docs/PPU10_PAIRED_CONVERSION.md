# PPU1.0 NewV paired conversion

Local admission: 2026-09-28. Device follow-up: **correctness PASS; both
complete-call captures slightly lower, stable speedup not established**.
See the [paired ACU verdict](PPU10_PAIRED_CONVERSION_ACU_20260928.md).
The original local evidence and registered decision below remain intact.
Parent: `e1627d0968604c6fde9d7273824abe3850b99b04`.
Opt-in: `gdn_chunk_residual(..., delivery="paired-conversion")`.
Same-binary control: `full-chunk`; scalar residual remains the RAW-BIT
numeric anchor. Public/default routing is unchanged.

## What transfers, and what does not

SM90's retained Value128Paired path converts FP32 NewV in accumulator order
before operand retile. PPU already converts into consumer-oriented shared
planes, so moving conversion before delivery is already present. The new
experiment tests the remaining part: native paired BF16 conversion.

Keep two independent rounding inputs, `BF16(x)` and
`BF16(x * relative_gate)`. In particular, do not copy a rounded-unscaled
operand into the scaled path. This is a conversion/delivery change, not
Hopper's inverse precision or a change to the recurrence.

The local SDK 2.1.1 capability probe (`dev/ppu/l039_pair_lowering.cu`) compiles
scalar, actlize and SDK conversion variants with scattered and packed
destinations. Actlize's `NumericArrayConverter` and the SDK intrinsic both
lower to native `v.pcnvt.bf16x2`. For scattered destinations, however, the
upper half also needs `v.shrl.b32 ..., 0x10`; two 16-bit stores remain. For a
contiguous packed destination the compiler even pairs the scalar source
automatically. Source-level pairing alone therefore proves no saving.

The actual PPU MMA C-slot map makes same-lane NewV pairs **nonadjacent** in
the current B-oriented shared layout. A packed 32-bit store would require a
different owner/layout or an exchange; it is not part of this experiment.
Keep existing scalar publishers, AIU.swzl/ld.swzl layouts, shared capacity,
workspace, barriers, prefix/solve/output kernels and all mathematical work.

`wy_paired_conversion.cuh` performs only the two array conversions;
`gdn_wy_paired_conversion_ppu.cu` instantiates full-C64 and generic state
bodies. Both full chunks and tails execute the new conversion, with the
existing tail predicates. There is no whole-call fallback that lets the
tail tests escape the new code. Reversing only the conversion seam in the
source checker reconstructs both complete original state bodies.

## Actual local code generation

SDK 2.1.1, PPU1.0, actlize `423253c`. These are native **static** counts,
not executed instruction counts or device performance:

| Property | Full-C64 control → candidate | Generic control → candidate |
|---|---:|---:|
| Static instruction sites | 1085 → 1086 | 1166 → 1167 |
| Registers/thread | 120 → 120 | 124 → 124 |
| Stack bytes | 0 → 0 | 0 → 0 |
| Shared bytes | 46080 → 46080 | 46080 → 46080 |
| BF16 MMA sites | 20 → 20 | 20 → 20 |
| NewV scalar BF16 conversion sites | 16 → 0 | 16 → 0 |
| NewV paired conversion sites | 0 → 8 | 0 → 8 |
| NewV high-half extraction sites | 0 → 8 | 0 → 8 |

Matrix loads, publication stores and barriers remain fixed. No new shuffle
or indirect register access is introduced. All **44 old device bodies**
retain identical native instructions, operands and resource records; the
new library has 46 bodies. Thus there is no predicted register/occupancy or
static-footprint gain to claim. Only the box A/B can establish a benefit.

## Local correctness and negative controls

L039 uses the shipping converter, actual PPU C/B traits, production writer
offsets and actlize's ld.swzl reader simulation. It checks every lane, slot,
warp and V slice across all 64 valid chunk extents:

- 512 contexts; 2,097,152 published BF16 values and 2,097,152 reader values.
- 524,288 same-lane pairs checked to be nonadjacent in this shared layout.
- 1,048,576 rounding outputs: all 65,536 FP32 high words, eight low-word
  boundary patterns and both planes, against an independent integer RNE
  oracle and the old scalar casts. These are not all possible FP32 inputs.
- 99,640 boundary witnesses distinguish early BF16 rounding before scale.

The local paired gate has 31 expected-red checks: five host plants, one
omitted-tail invocation, eleven source plants, twelve native plants and two
linked-dispatch plants. Wrong half selection, wrong row coefficient,
premature rounding, packed-store assumptions and a missing coverage context
all fail. Native tracing binds the original FP32 accumulators through the
scale, low/high pair operands and extraction to all sixteen real shared
stores. Operand swaps and wrong store offsets fail even when opcode counts
are unchanged. Both C launch dispatch and the actual linked Python binding
must reach the new kernels.

Local regression: **272 CPU tests PASS, 30 CTest cases PASS**. Four
NVIDIA-only device modules are separately SKIP, not passed tests. The WY
whole-library native/resource gate passes its 76 existing negative controls;
the original 15-kernel backend also compiles, links and passes its native
audit. No local PPU execution is claimed.

## Registered device decision and one-command handoff

The experiment was registered locally at 2026-09-28 22:12:10 UTC before the
candidate edits. The old 1.5x FLA target remains suspended. Run without other
work on the selected PPU:

```bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_paired_conversion_box.sh
```

Optional `PPU_SDK=/path/to/PPU_SDK` overrides an inherited SDK root. ACU
defaults to `/sim/eec/shared/junfu.qx/asight/bin/acu`; output is a fresh
directory under `/workspace`. The runner builds once, checks the native and
linked candidate, then runs retained 36 cases × 8 repetitions and 256
extent/gate/initial-state cases × 8. Candidate/control must be RAW-BIT equal
to scalar residual and pass the unchanged 2% independent recurrence oracle.
Same-input FLA admission also precedes profiling.

Capture B1/S2048/Hk16/Hv32/K128/V128/C64, initial=None, at both g=-1 and
g=-0.1. Compare the sum of **all actual kernels in each complete call**:
control/candidate/FLA are expected to have 4/4/7 kernels, including FLA fills.
There is no kernel-name or launch-count filter and no API timing substitute.
Unexpected kernel inventories must be explained, not silently discarded.

Lower complete-call time at both gates retains the candidate for repeated
confirmation; mixed or slower timing retains the control. Report variation
in unchanged stages separately. Lower conversion counts alone do not admit
a faster default. Neither outcome changes the numerical contract or routing.

Upload only the final printed **`paired-conversion.tar.gz`**. It contains
both ACU bundles, measurement SHA and binary hashes, numerical/native
admission receipts, raw reports and device/compiler identities. The packer
uses the recorded measurement SHA rather than the checkout's later HEAD;
missing/empty/symlink evidence or an existing archive fails closed.

After this device verdict, the remaining migration work is PPU-specific
geometry selection and unified public selection. Neither is hidden inside
this conversion experiment.
