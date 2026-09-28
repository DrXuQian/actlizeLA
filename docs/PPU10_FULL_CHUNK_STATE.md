# PPU1.0 full-C64 state experiment

Parent `c99d84bb93e32849a9eeac312809c3a4072a6c56`.
Opt-in `residual-full-chunk`; control `residual-gate-cache-solve-static`.
Default routing and SM90 are unchanged. This is a local-validated candidate,
**not a measured performance win**.

## Scope and one changed stage

Priority: B1/S2048/Hk16/Hv32/K128/V128/C64, natural-log g=-1 and -0.1,
BF16 inputs, FP32 state, zero initial state and final state enabled.
The measured control's complete ACU sums were180.070us and181.194us;
FLA224.480us and226.119us. Those were single captures, not new baselines.
About30.4us still separates that control from the1.5xFLA goal.

For positive sequence lengths divisible by64, `valid` is64 for every chunk.
The state specialization removes four redundant tail seams: runtime `valid`,
beta's row predicate, residual's zero branch and Vnew's zero branch.
Key/value AIU descriptors consequently receive a constant extent. It does
not change FP32 recurrence/FMA order, `expf`, BF16 rounding boundaries,
gate reuse, layouts, output ownership, workspace, CTA/grid or barriers.
Prefix, static solve and HV output use the same existing kernel symbols.
There are still four kernels in a complete call.

For every other sequence length the host calls the unchanged static-solve
control, which retains its own input admission and tail handling. Selecting
this delivery does not install a production policy. Negative/zero extents
remain errors. Full chunks have beta writers0..63 once; the real native
C-layout covers64x32 value cells and128x32 state cells exactly once.

The new state is in a separate CU intentionally: extracting a shared
`__forceinline__` template changed the old control's native body1166->1163
sites. That factoring experiment was rejected and the control file restored.
The source gate allows only the four listed seams between the two bodies.

## Native screen, not latency

Real local HGCC2.1.1-a5c56e, PPU1.0:

| Resource / instruction | Control | Full chunk |
|---|---:|---:|
| Static instruction sites |1166|1085|
| Registers/thread |124|120|
| Stack bytes |0|0|
| Shared bytes |46080|46080|
| BF16 MMA sites |20|20|
| AIU loads |4|4|
| `ld.swzl` nontrans / trans |28/8|28/8|
| CTA barrier sites |5|5|
| `exp2` sites |2|2|
| `s.cbr.az` |14|5|
| `v.csel.b32` |8|0|
| `v.mov.v2s` |40|36|
| `s.wait` |73|78|

Static BF16 conversions41->40 and shared16-bit stores44->40 come from removed
zero/tail paths, not a new numerical precision. Useful floating arithmetic
sites and matrix/global transfer sites are unchanged. Fewer static sites
do not imply a proportional dynamic-count or time reduction; waits increased,
and the lower register count need not change the residency limit.
Native loops rotate when predicates disappear, so loop spans are not paired
by ordinal to claim an unchanged phase.

All37 pre-existing kernels retain identical native instruction/operand
sequences and resource records under this same local compiler. The inventory
is now38, adding one state kernel. The actual previous box used HGCC2.2.0-dev;
the runner repeats native checks on its real build, not the local image.

## Admission and preregistered decision

Local checks include65537 sequence lengths0..65536,524800 complete chunks,
signed extent boundaries, real C-layout ownership, wrong admission/owner/
coverage-denominator negatives, exact source/host wiring and native negatives.
Renaming the old state under the new symbol must fail the codegen gate.
Linked host calls prove the new state, static solve, shared output and old
tail fallback exist. IBT PLT targets are resolved from actual ELF relocations,
not guessed disassembler labels; missing/wrong target negatives remain red.

Box correctness retains the unchanged30-case independent2%oracle and8 RAW-BIT
repeats against scalar residual. This candidate adds six cases: S128 with
zero/nonzero state at both gates, and S2048 with nonzero state at both gates.
Thus **36 cases**, plus existing GVA expansion/output-only/input-immutability
checks. Any failure stops timing/profile capture. No tolerance is relaxed.

Capture control/candidate/FLA sequentially on one idle PPU, one built binary
and the same fixture, at both gates. Count every kernel (4/4/7, including FLA
fills). The subject capture must contain `gdn_wy_residual_full_chunk_state`;
the control must contain `gdn_wy_residual_gate_cache_state`. Otherwise this is
not the registered experiment. Report state and complete-call ACU sums,
dynamic opcode counts, registers/spills, occupancy, memory dependencies and
shared traffic. Do not substitute API-event spans or a single selected stage.

- State and complete-call lower: observed improvement; repeat for confirmation.
- Fewer static/dynamic instructions but no lower complete-call time: no observed
  speed gain; retain control and report it, not just the favorable counter.
- Either gate regresses: keep that regression visible; no automatic promotion.
- New spill, extra data movement or changed numerical work: experiment invalid.
- The1.5x target is met only if **each** complete candidate sum is <= its
  same-run FLA sum/1.5 with numerical admission. One capture is not a robust
  selector admission, even if that ratio passes.

## Run on box

In the actlizeLA checkout of `agent/ppu10-full-chunk-state-20260928`, with the
chosen PPU idle and the SDK-matched environment already configured:

```bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_full_chunk_box.sh
```

`DEVICE` is the physical visible-device selector; use the intended device.
`PPU_SDK`/`PPU_SDK_ROOT` may select an installed SDK if not `/usr/local/PPU_SDK`.
The runner builds once, admits both gates, then uses
`/sim/eec/shared/junfu.qx/asight/bin/acu` directly. It prints the two upload
tar paths under a fresh `/workspace/actlizeLA-ppu10-full-chunk-*` directory.
Failure in the first child stops the second capture.

Local runtime import is **SKIP**: this host lacks GLIBC_2.38 required by the
SDK runtime. Device numerics and performance are NOT_RUN locally. Compilation,
linking and host/native evidence cannot substitute for those box gates.

Completed local replay:204 Python tests,25 host CTests,65 whole-image negative
plants and16 source/native/host-call negatives PASS. All37 old native bodies
and resource records are unchanged. See the hash-bound
[local evidence](../dev/ppu/results/full_chunk_local_20260928.json).
