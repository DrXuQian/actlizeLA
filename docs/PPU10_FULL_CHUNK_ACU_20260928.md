# Full-C64 state: measured PPU1.0 result

Verdict: **observed gain at both gates; about1.33–1.35xFLA, not1.5x**.
Keep the opt-in candidate for repeat confirmation. Default routing unchanged.
One ACU capture per arm/gate; these are summed kernel durations, not medians
and not API-event spans. No numerical criterion or preregistration was changed.

Source `41ade14b34d142a77c33f2cb73b2482b975fe0f4`, clean source snapshots;
control `residual-gate-cache-solve-static`, subject `residual-full-chunk`.
Read the [registered experiment](PPU10_FULL_CHUNK_STATE.md) for the fixed scope.
Both captures use B1/S2048/Hk16/Hv32/K128/V128/C64, BF16 inputs, FP32 state,
zero initial state, final state enabled, natural-log gates-1/-0.1.

## Complete call, every kernel included

| Gate | Control us | Full-C64 us | FLA us | Time reduction vs control | FLA / candidate |
|---|---:|---:|---:|---:|---:|
| -1 |174.84529|168.92412|225.33294|3.39%|1.33393x|
| -0.1 |178.40706|169.36236|227.87529|5.07%|1.34549x|

Kernel denominators are4/4/7. FLA's two fill kernels are included; its solve,
W/U, recurrence and output are all counted. Candidate captures contain the
new full-chunk state symbol, not a renamed control or unused specialization.

| Stage | Strong control→subject us | Weak control→subject us |
|---|---:|---:|
| Prefix, unchanged |2.46294→2.47588|2.46294→2.48118|
| Static solve, unchanged |44.60941→44.43471|44.33765→44.69941|
| State, changed |85.45412→79.70412|89.19529→79.58706|
| HV output, unchanged |42.31882→42.30941|42.41118→42.59471|

The changed state saves5.75000/9.60823us (6.73%/10.77%). The three unchanged
stages contribute-0.17117/+0.56353us variation; do not credit it to the state
edit. Their native dynamic opcode dictionaries are identical between arms.
Different observed state gains at the two gates do not establish a gate-
dependent optimization effect: native counts are identical across gates, and
there is only one capture each. Repeat before making a stable selector claim.

Same-run1.5xFLA limits are150.22196/151.91686us. Another18.70216/17.44550us
must be removed from the complete candidate call:11.07%/10.30% of its current
time. No use of the prior180/181us control to inflate this run's improvement.

## What actually changed

Both gates have the same instruction-accounting result:

| State counter | Control | Full-C64 |
|---|---:|---:|
| Total native dynamic instructions |13,648,000|12,153,984|
| `v.csel.b32` |262,144|0|
| `s.min.i32` |32,768|0|
| `s.lop.emsk` |237,568|98,304|
| `s.cbr.az` |157,696|18,432|
| `v.mov.v2s` |1,120,256|989,184|
| `v.mov.b32` |844,800|673,792|
| `v.reg.dchk` |131,072|0|
| `s.wait` |1,717,248|1,755,136|
| `s.mov.b32` |140,288|177,152|

Net reduction1,494,016instructions,10.9468%. This is the actual generated
removal of runtime tail predicates and associated control/address work,
not just a shorter source file. Some instructions increase: neither all moves
nor all waits improved. All per-PC totals close on ACU's complete native
opcode counter, including every unchanged stage and FLA kernel.

Useful state BF16 MMA stays655360; AIU loads16384; SWZL loads917504normal+
262144transposed; CTA barriers132096. Floating arithmetic and matrix/global
load/store dictionaries stay unchanged. The1024 fewer BF16 conversions are
one zero-constant setup per warp: captured `s.mov sreg19,0` reaches
`v.cnvt.bf16.f32.rtte vreg76,sreg19` once per warp. Recurring BF16 rounding
is not removed, and all output/state bits remain equal to scalar residual.

Actual box native state size is**1181→1098**, not the local SDK's1166→1085.
Box compiler isHGCC2.2.0-dev; local compiler2.1.1. Both have124→120registers,
zero stack and46080shared bytes. All37 existing native bodies match the
previous uploaded box build exactly. Only4/38 current box bodies match local
codegen, so box resources/PCs, not local instruction counts, are authoritative.

## This is not a bank-conflict or occupancy win

- Read BC3014656, write BC1425408, total4440064: unchanged at both gates.
- KVD→TSM staging112MiB, KVD ordinary-load transactions18MiB,
  KVD stores50MiB and L2→KVD10MiB: unchanged. No doubled delivery volume.
- Grid128x256threads unchanged. Active warps/CU14.18→14.12 strong,
  14.19→14.11 weak; eligible warps/WE0.36→0.35/0.34. Lower registers did
  not increase achieved occupancy in this experiment.
- Fetch per-issue ratio0.32→0.19 strong,0.31→0.20 weak; memory-dependency
  ratios1.80→2.04/1.78→2.04 and sync1.12→1.21/1.22 increase while time
  falls. These changing-denominator ratios are not disjoint wall-time shares.

The remaining state is still about47% of the complete call. Next profiling
work should localize its remaining operand-address delivery and shared/AIU
waits against the already paired SWZL layout, not blindly increase grid/warp
count or claim that the unchanged aggregate BC explains the residual gap.
No new kernel change is made by this result review.

## Admission and provenance

Both archives:772/772 data-file checksums,297/297 tracked source snapshots
equal the capturedSHA.36numerical cases x8 repetitions per selected delivery,
independent2%oracle, RAW-BIT equality to scalar residual, tail/GVA/initial-
state/output-only and input immutability all PASS. Both capture gates share
the same exact binding/native binaries, device UUID, fixture contract and
FLA identity; every kernel reports1.700GHz CE. Source and preceding-build
diffs are empty. No uploaded executable was run during local analysis.

PPU-ZW81072CU, UUID019ee024-8860-091c-0000-0000007aff6d,
driver2.1.2-r7b50d071022; deviceSDKbanner2.0.2-d2568e060312,
actualHGCC2.2.0-dev (Jun3). Torch2.9.0/runtime12.9.
Capture tool `/sim/eec/shared/junfu.qx/asight/bin/acu`.

- Native DSO SHA256: `a9c7e8ba1ed8784650b059bdf97e1a97a0cc30ea1f33af86d19fd52ef1bd171e`
- Binding SHA256: `964b7ce186960a39e9a7766ddbc71ffe73d2e467a75fd2e25e89e189076bc24d`
- Strong archive: `09e87641461a41599cba00e265f1ae59174af84032b4319d87677fef8166a7d8`
- Weak archive: `df4d430d3c922681ee64cc5ea0398cea1fb20c57dd29256f928f60a31eaf65f2`

[Machine-readable measured result](../dev/ppu/results/full_chunk_acu_20260928.json)
retains every stage, opcode dictionary, selected metric, input/output hashes,
and the actual-tool identity. Reproducible import/audit scripts and original
archives are in`/workspace/actlizeLA-migration-20260928/ppu10-tuning/full-chunk/acu-analysis-20260928/`.
Analysis negatives reject an omitted full-call kernel, the old state presented
as the candidate and a changed device. Uploaded native/linked-host negative
gates also pass. Default routing remains unchanged.
