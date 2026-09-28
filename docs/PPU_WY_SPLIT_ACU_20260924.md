# Split prepare: device verdict against incumbent and FLA

**ACU all-kernel sum: 304.995 -> 269.306 us, 11.70% lower latency.**
The same capture's FLA sum is223.583 us, including both fill kernels. We
remain20.45% slower; the user's1.5x target requires149.055 us. Keep the new
path as an opt-in experimental candidate. No default/routing promotion.

## Evidence and scope

Uploaded `/root/acu.tar.gz`, SHA256
`9c1802c922dd18aa7262fb786e96b4d10b192484ee31ba3f9ead6931e541ad71`:
575 unique regular files, exact574-entry checksum denominator, all verified.
Source and measured binaries are from clean
`c2bdb5c306904a6bbcd34295dbeddf71dd165a80`, run
`/workspace/gdn-wy-split-prepare-c2bdb5c-20260924T041207Z-4029087`.

- Binding SHA256: `eb49e7e28d8f379bf6a36a5f39d5894ea907bfbb7ec7cb3c983064b34d0a9feb`.
- Device DSO SHA256: `d8a9f61a62d6cd4263acaadaeb92bad470396ae8e5272abf8327eb643781cfb4`.
- Physical UUID: `019ee024-8860-091c-0000-0000007aff6d`, PPU-ZW810,72CU.
- Every one of15 captured kernels reports1.700 GHz. Actual loaded-library
  hashes, subjects, symbols, launch grids and input/reference receipts agree.
- Box banners: HGGC2.2.0-dev (June3), ACUv2.0.2/data15000, Torch2.9.0,
  runtime12.9, Python3.12.3. Build logs are not archived. Local SDK2.1.1
  report re-import does not change the recorded box compiler identity.

The successful capture supersedes the earlier missing-hgcc build attempt.
It does not establish which SDK-path repair the operator used.

Shape is **B1/S2048/Hk16/Hv32/K128/V128/C64**, native GVA1:2, g=-1,
BF16 inputs/output, zero FP32 initial state, FP32 final state requested,
scale1/sqrt128, no QK normalization. Control is AIU13808; candidate is
split-prepare30192. FLA0.6.0/Triton3.4.0 has backend dispatch disabled and
only the recorded process-local CUDA13/PTX90 parser backport.

Device correctness:16 distinct cases x2 deliveries x8 repeats, unchanged
independent2% oracle, scalar-WY RAW-BIT equality, tails, nonzero state, GVA
and output-only checks PASS. Preceding numerical admission covers BOTH
g=-0.1 and-1 with8 repeats; actual profiled outputs match those receipts.
Control and candidate output/state fingerprints are identical. Performance
is **g=-1 only**; no weak-gate speed claim. API timing was explicitly NOT_RUN.

Native report re-import independently agrees with archived durations,
opcode totals and traffic. Per-PC execution sums close for all15 kernels;
omitting an executed PC fails closure in every kernel. No archived source
or binary was executed locally. All65 host benchmark/capture/selector
contracts were rerun PASS using the prior task-local Torch/Python environment
with GPUs hidden; the default Python lacks Torch. No system install changed.
Each arm has one full ACU capture, not a
repeated timing envelope: the following is the observed profile comparison,
not a cross-shape production performance admission.

## Complete phase ledger (us)

| Phase | AIU control13808 | Split30192 | FLA |
|---|---:|---:|---:|
| Prefix |included below|2.42118|2.70882|
| KKT + solve |included below|52.40647|44.85412|
| W/U |included below|46.73824|35.89235|
| **Prepare subtotal** |**136.34118**|**101.56589**|**83.45529**|
| State |123.11882|120.93294|90.40765|
| Output |45.53471|46.80706|43.29471|
| Separate fills |0|0|6.42529|
| **All kernels** |**304.99471**|**269.30589**|**223.58294**|

FLA fills are4.28529+2.14000 us. Kernel denominators are3/5/7; every
kernel contributes exactly once. Subtotal rows are not counted twice.
These are summed instrumented kernel durations, not complete API latency
or a measure of host launch/dispatch gaps.

Prepare saves34.77529 us (25.51%). State/output are unchanged native bodies
with identical dynamic opcode counts in control and candidate. Their small
timing shifts are not new state/output source improvements.

## What the split actually changed

| Quantity | Fused control prepare | Split prepare, all three kernels |
|---|---:|---:|
| Dynamic instructions |30,292,992|26,854,400|
| BF16 MMA |344,064|344,064|
| TF32 MMA |98,304|98,304|
| KVD global-store interface |512.25 MiB|40.25 MiB|
| L2 global-store interface |32.25 MiB|40.25 MiB|
| DRAM reads |24.269 MiB|40.907 MiB|

W/U itself now writes32 MiB at KVD AND L2, instead of32 MiB useful output
creating512 MiB KVD requests. **The16x interface amplification is removed.**
New40.25 MiB is prefix0.25 + inverse8 + W/U32; the8 MiB inverse materialization
is a real additional write/read. Extra key reads also remain. Hence the
result is not a claim of lower traffic at every hierarchy or16x HBM savings.

The independent resource budgets are measured, not inferred from code:

| Kernel | Threads | Shared B | Regs/thread | Active warps/CU |
|---|---:|---:|---:|---:|
| Control fused prepare |128|70144|86|11.77|
| Split prefix |64|256|32|23.08|
| Split solve |128|49664|84|19.01|
| Split W/U |256|41472|128|29.78|
| FLA W/U |256|25600|100|37.31|

All kernels in this resource table have zero stack; FLA's separate solve
reports8 B/thread stack, not zero. W/U residency is register-limited at4 blocks/CU,
whereas shared permits6; FLA permits5 register-limited blocks. We changed
phase/resource separation, operand ownership/reuse AND vector publication.
The aggregate gain is established; their individual latency contributions
are not isolated by this one experiment.

## Remaining excess: state first, but not state alone

Matched mathematical phases exceed FLA by52.14824 us:

- State30.52529 us:58.5%.
- Prepare18.11060 us:34.7%.
- Output3.51235 us:6.7%.

FLA's6.42529 us fills reduce the whole-call gap to45.72295 us. Do not use
the matched-phase denominator to claim a whole-call percentage.

### State: a scheduling/delivery investigation, not another grid claim

Both state kernels have grid128/block128,524,288 BF16 MMAs, about88.266 MiB
DRAM reads,98 MiB KVD/L2 writes and7.06/7.10 achieved warps/CU. Our shared
footprint49,408 B is slightly smaller than FLA50,432 B. Occupancy or missing
tensor arithmetic is not sufficient to explain120.933 vs90.408 us.

| State observation | Split candidate | FLA |
|---|---:|---:|
| Executed instructions |11,967,104|10,040,320|
| Matrix-load instructions |720,896|655,360|
| v.mov.v2s |461,824|0|
| s.blksyn.defer |66,048|133,120|
| ws__warps_issue_stalled_sync.avg |40,034.76|7,789.57|
| ws__warps_issue_stalled_commit_dependency.avg |11,008.17|1,643.62|

Our fewer barrier instructions coexist with more sync waiting. This is a
reason to inspect the generated load/wait/arrival schedule, not to remove
required barriers. These named absolute counters are not wall-time fractions;
their sum is not a latency equation. The current source already overlaps
the U load with W@H. It does NOT yet prefetch the next chunk's W/K during
the current state update. A bounded next candidate should address that
cross-chunk staging/lifetime seam while retaining current arithmetic and
the present state body as control. Additional buffering must be charged
against shared/register residency; more buffers are not free overlap.

### W/U: publication is fixed; conditioning still costs

The new W/U46.738 us versus FLA35.892 us accounts for10.846 of the18.111 us
prepare gap. Both have262,144 BF16 MMAs,32 MiB KVD/L2 output writes and
about32.39 MiB DRAM reads. Our11,707,392 executed instructions are1.716x
FLA6,823,936. Measured excess includes524,288 BF16 conversions and524,288
shared BF16 stores; scalar `tsm.ld.b32` is811,008 versus0. Our matrix loads
are196,608 versus393,216: fewer matrix loads did not make W/U faster.

Source correspondence: current W/U bulk-stages unconditioned K/V, reloads
and weights them through scalar shared accesses, rewrites conditioned BF16
shared values, then uses the same planes for accumulator publication. FLA's
source weights loaded tensors before `tl.dot`, without that explicit
shared-reload/rewrite step. A complete
producer/layout/consumer variant that conditions in registers is the next
W/U hypothesis; mapping the measured instruction excess to each source region
still needs a controlled native-code comparison. Do not label all excess as
unsupported-PTX emulation or swap SWZL for NCOM on unchanged bytes.

Solve52.406 vs44.854 us also remains open. Preserve the existing TF32
high/high + both residual terms:98,304 vs FLA32,768 TF32 MMAs is an explicit
precision cost, not a free instruction reduction.

## Decision and next boundary

Retain split-prepare as the measured opt-in candidate and AIU13808 as its
counterfactual. No source kernel or selector changed in this analysis.
Next: state operand/prefetch schedule, then W/U conditioning/delivery;
output is lower priority. Each needs independent source/native/numeric
admission and a complete same-run ACU comparison.

**1.5x is still unmet.** FLA223.58294/1.5=149.05529 us. We need another
120.25060 us (44.65%) off269.30589 us. Prepare+output alone already costs
148.37295 us, leaving just0.68234 us for state at that target. Therefore a
state-only change cannot plausibly deliver the target; further improvements
across stages are necessary. Matching FLA phase times is parity, not1.5x.

Host-only reproduction is in
`/workspace/gdn-wy-split-acu-analysis-20260924/`: `analyze.py --native`,
`reimport.sh`, `summarize.py`, verified receipts, native exports,
`benchmark.csv`, `summary.json`, `analysis-verified.log` and the extracted
reports. Raw reports/libraries stay outside source commits. No device rerun
is required to inspect this verdict. `check_host_contracts.sh` preserves the
host-test environment and `host-contracts.log` the65-test result. The measured
resource/traffic lesson is retained in the PPU iteration skill reference.
