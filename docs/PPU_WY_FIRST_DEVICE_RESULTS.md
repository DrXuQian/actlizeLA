# WY first device result: improvement, not FLA alignment yet

Evidence: user-supplied log from
`/workspace/gdn-wy-fla-dd70e5d-20260921T041641Z`, source
`dd70e5d579013c18ab2c0913557353cfb20b6566`. The artifact itself has not been
downloaded/verified locally. No new device measurement is claimed here.

## Fixed scope and outcome

B1/S2048/Hk16/Hv32/K128/V128, BF16, native GVA, natural-log gate, zero initial
state, return output and final state, no QK normalization. WY/FLA return FP32
final state; the unchanged original returns BF16 final state. This is not a
claim of identical internal rounding across all three arms. WY device
admission: 16 cases PASS under the unchanged 2% output **and** state rule.
All three comparison arms report 8/8 stable repetitions at both decay values.
Default routing and the original strong-decay reset/replay path remain intact.

| Gate | Original median [range], us | WY median [range], us | FLA median [range], us |
|---|---|---|---|
| -0.1 | 915.720 [910.096, 960.612] | 715.232 [713.744, 726.492] | 480.806 [474.344, 852.296] |
| -1.0 | 437.670 [420.084, 529.332] | 715.126 [712.672, 725.712] | 507.562 [474.676, 805.576] |

Unchanged decision rule: **disjoint observed envelopes**, not a confidence
interval and not a comparison of medians alone.

- Weak: WY-WINS against original. Strong: ORIGINAL-WINS against WY.
- WY versus FLA: **UNRESOLVED at both gates**. Ratios 1.4876 and 1.4089 are
  descriptive median ratios, not an admitted winner/loser verdict.
- The strong original/FLA ranges also overlap. Do not promote the original
  median's apparent advantage to a proven win from this run.
- No result establishes the predeclared WY <=1.10x FLA alignment target.
  Against a conditional 500 us target, WY needs about 215 us (30%) less of
  its own latency; it is still about 43% slower. That engineering target
  neither changes the admission rule nor justifies discarding samples.

Weak FLA's first three batches are 823.048/852.296/839.772 us, while its last
nine have median 480.092 us. Strong FLA also has slow batches **later** in
the run. Neither "only initial warmup" nor "steady 500 us" is established.
Clock/power, other jobs, host submission and backend behavior are competing
explanations, not observations supplied by this timing log.

## What the clock actually measures

`bench_ppu_wy_fla.py` uses 12 samples per arm, ten complete API calls between
each event pair, then takes the median of those twelve **batch averages**.
Eight admission calls and five warmups precede timing. The three roles use
all six balanced orders; each arm synchronizes before its event pair. No two
arms intentionally overlap. Validation/digests occur after the end event.

The event interval includes device work and exposed host submission gaps;
API allocation/dispatch is not magically excluded. It is neither a single
kernel timer nor a CPU wall-clock total. Post-sample checks are outside the
interval but can affect subsequent cache/clock state. No cause for the FLA
spread can be established without further measurement.

The terminal PASS means numerical admission and measurement completed. It
does not mean a performance win. New output spells this out and labels median
ratios `DESCRIPTIVE_MEDIANS_NOT_ADMISSION`; the timing algorithm, sample
selection, tolerance and envelope rule have not changed.

## Next bounded experiment

Capture explicit WY and FLA on identical inputs, reusing both dd70e5d WY
binaries. [The ACU handoff](PPU_GDN_ACU.md) archives old raw API samples
separately from new profiled stage counters and the three-stage resource map.
Do not recompile, change routing or replace a mathematical stage before this
measurement can identify the expensive stage. The original strong path stays
available; shortening weak recurrence does not justify replacing its winner.
