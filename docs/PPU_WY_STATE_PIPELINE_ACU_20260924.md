# State operand pipeline: measured ACU verdict

**All-kernel sum: 271.483 -> 252.191 us, 7.11% lower.** The changed state
kernel falls122.196 ->105.335 us (-13.80%). Same-run FLA is224.775 us,
including both fill kernels: we remain12.20% slower. The user's1.5x target
requires149.850 us, not reached. Keep62960 as an opt-in experimental winner
in this measured scope; do not change automatic routing.

## Admission and identity

Uploaded `/root/acu.tar.gz`, SHA256
`a735ad2ad07a9390358664dbe88a1ad0036fb3961d8d7ae0007e0c3d3c30bc6f`:
580 unique regular files, exact579-entry checksum denominator, all verified.
Measured clean source is`8eaaa5668cf0a3ff75b31e75cc73db60ffef0e39`, from
`/workspace/gdn-wy-state-pipeline-8eaaa56-20260924T053109Z-1608578`.

- Binding SHA256: `95c82fd7e94554c5ec3ebbcd9fb0875cbed4eccd395fd403f4c6f1513486dad5`.
- Device DSO SHA256: `8faaae58e3e77283033111d757b4a76287d1a05a1d60ebbad2dbfcb674e1e59a`.
- PPU-ZW810,72CU, UUID`019ee024-8860-091c-0000-0000007aff6d`.
- All17 captured kernels report1.700GHz. Source/loaded libraries, input,
  reference, selected masks, launch symbols/grids and output receipts agree.
- Box banners: HGGC2.2.0-dev (June3), ACUv2.0.2/data15000, Torch2.9.0,
  runtime12.9, Python3.12.3. Capture used`/usr/local/PPU_SDK/asight/bin/acu`.
  The user's subsequent site-tool instruction applies to future captures;
  it does not relabel this report's actual provenance.

Shape **B1/S2048/Hk16/Hv32/K128/V128/C64**, native GVA1:2, g=-1,
BF16 input/output, zero FP32 initial state and FP32 final state, scale1/sqrt128,
no QK normalization. Control is split-prepare30192, subject state-pipeline62960.
FLA0.6.0/Triton3.4.0 retains disabled backend dispatch and only the recorded
CUDA13/PTX90 parser compatibility backport.

Device admission passes16 distinct cases x2 deliveries x8 repeats, unchanged
independent2% oracle and scalar-WY RAW-BIT equality, tails/nonzero state/GVA/
output-only. Both weak and strong gates pass preceding numerical admission;
only strong g=-1 is profiled. Outputs match those admitted receipts exactly.
API_TIMING=NOT_RUN. Each arm has one complete profile, not a repeated timing
envelope or a cross-shape production admission.

Archived exports and host-only native report re-import agree. Per-PC sums
close all17 kernels against ACU's independent opcode totals; omitting any
selected executed PC makes each closure fail. No uploaded code/binary was
executed locally. The requested site tool is not installed locally; existing
reports were re-imported with the available localSDK2.1.1 parser, not recaptured.

## Complete ACU phase ledger (us)

| Phase | Split control30192 | Pipeline62960 | FLA |
|---|---:|---:|---:|
| Prefix |2.42824|2.47529|2.60353|
| KKT + solve |52.53882|52.12353|44.27294|
| W/U |47.33471|46.12941|35.70647|
| **Prepare subtotal** |**102.30177**|**100.72823**|**82.58294**|
| State |122.19647|105.33471|91.92118|
| Output |46.98471|46.12765|43.83882|
| Separate fills |0|0|6.43235|
| **All kernels** |**271.48295**|**252.19059**|**224.77529**|

Denominators5/5/7; each kernel counted once, subtotals not added twice.
FLA fills are4.29176+2.14059 us. These sums are instrumented kernel durations,
not complete public-API spans and not a measurement of host launch overhead.

Of the observed19.29236 us total decrease,16.86176 us is in the changed
state. The other2.43060 us is variation in unchanged prefix/solve/WU/output
bodies: every one of their dynamic opcode counts is identical between arms.
Do not attribute that residual to new preparation/output code.

## State: waiting falls without removing useful work

| State fact | Control | Pipeline | FLA |
|---|---:|---:|---:|
| Grid / threads |128 /128|128 /128|128 /128|
| Regs/thread, **box compiler** |234|232|256|
| Stack B/thread |0|0|0|
| Shared B |49,408|49,408|50,432|
| Register-limit blocks/CU |4|4|4|
| Achieved warps/CU |7.03|7.06|7.10|
| Dynamic instructions |11,967,104|11,858,560|10,040,320|
| BF16 MMA |524,288|524,288|524,288|
| Matrix loads |720,896|720,896|655,360|
| AIU loads |20,480|20,480|different copy family|
| v.mov.v2s |461,824|461,824|0|
| CTA barrier executions |66,048|49,664|133,120|
| s.wait executions |1,462,784|1,380,480|1,021,952|
| KVD / L2 global-store bytes |98 /98 MiB|98 /98 MiB|98 /98 MiB|

DRAM reads are88.269 /88.267 /88.264 MiB, essentially unchanged. Actual
barriers close exactly:128 CTAs x4 warps x(32 chunks x4 +1) before, and
128 x4 x(32 x3 +1) after. AIU closes128 x32 x5 in both arms: no missing K/U
or next-W loads and no extra final-chunk prefetch.

| Raw stall counter (`ws__warps_issue_stalled_*.avg`) | Control | Pipeline | FLA |
|---|---:|---:|---:|
| sync |40,902.39|21,044.39|7,993.58|
| commit_dependency |11,204.24|4,982.45|1,706.92|
| memory_dependency |51,741.73|48,256.84|43,915.76|
| tsm_ldst_dependency |30,591.43|33,647.94|37,920.95|
| compute_dependency |16,926.27|17,126.90|15,967.10|

Sync decreases48.55%, commit waiting55.53%, while instructions fall only0.91%.
Active warps and occupancy limits barely change. This supports the intended
operand-issue/wait schedule benefit, not a claim that more occupancy, less
mathematics or less HBM traffic explains the speedup. TSM dependency rises10%;
not every stall improves. These counters are averaged warp-stall accounting,
not disjoint wall-clock shares; their sum cannot be subtracted from duration.

The actual uploaded native CFG retains12 PROJECT MMAs between each current
K/U copy and its commit wait, and16 UPDATE MMAs between next-W and its next
wait. Moving the same wait immediately after next-W (unchanged opcode
multiset, diagnostic ISA text only) fails the schedule check. Source guards
and local lifetime proof remain necessary for the last iteration.

Do not substitute local resources for box measurements: localSDK2.1.1 gave
230 regs and1880 static instructions; this box compiler gives232 and1956.
Its control has234 regs and1958 static instructions. The schedule still
matches; this was not a static-footprint experiment. Prefetch rescheduling
and one fewer barrier were combined, so their individual latency savings
are not separately measured.

## Remaining gap and next bounded change

Matched mathematical phases exceed FLA by33.84765 us:

- Preparation18.14529 us (53.61%): solve7.85059, W/U10.42294, prefix-0.12824.
- State13.41353 us (39.63%).
- Output2.28883 us (6.76%).

FLA's6.43235 us fills reduce the full-call gap to27.41530 us. Preparation is
now the largest combined excess; state alone is still the largest single
kernel excess. Keep those denominators distinct.

Next candidate: **W/U conditioning and register delivery**, preserving the
now-measured state pipeline and all arithmetic/rounding boundaries. W/U does
262,144 BF16 MMAs on both sides, but executes11,707,392 vs6,823,936 instructions
(1.716x). Ours has811,008 scalar shared b32 loads and524,288 extra BF16
conversions/stores. Current path loads unweighted K/V into shared, reads them
back for beta/exp weighting, writes rounded operands to shared, then loads
MMA fragments. FLA does the weighting in its loaded operand flow. Inspect
how to condition the real PPU operand fragments without that extra shared
roundtrip, not by changing input layout or weakening the oracle. This is a
next implementation proposal, not a change made during this upload review.

C++/CUTLASS versus Triton is not the causal distinction here. Our WY kernels
use actlize MMA/load atoms inside hand-written ownership, materialization
and scheduling code; they are not a mature library-provided GDN collective.
FLA also generates native tensor-core instructions. More manual control is
an opportunity to improve that execution structure, not an automatic speed
advantage. Equal MMA work with1.716x total W/U instructions is a concrete
implementation gap. Simply changing the language or calling the MMA a
CUTLASS atom does not remove an extra shared-memory roundtrip.

Even matching FLA state alone leaves238.77706 us. Even matching *every*
mathematical phase leaves218.34294 us (our call has no separate fills), not
149.85019 us. The1.5x goal still needs102.34040 us /40.58% of current duration
removed: closing parity is necessary but not sufficient. No inflated win or
weak-gate/other-shape extrapolation follows from this profile.

Evidence: `/workspace/gdn-wy-state-pipeline-acu-analysis-20260924/`, with
verified raw import, per-PC closure, phase/counter JSON and reproducible scripts.
Future box runner now defaults to`/sim/eec/shared/junfu.qx/asight/bin/acu`;
69 host capture/ABI/benchmark contracts pass. A real local invocation with
that site executable absent fails before build/profile and keeps INCOMPLETE,
despite an SDK ACU being available. No silent SDK/PATH fallback.
