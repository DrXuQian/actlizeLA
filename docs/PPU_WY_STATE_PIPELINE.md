# Single-buffer state operand pipeline

Device follow-up: [verified three-arm ACU result](PPU_WY_STATE_PIPELINE_ACU_20260924.md).
State122.196 ->105.335 us; all-kernel sum271.483 ->252.191 us, FLA224.775 us.
Numerics pass; keep opt-in,1.5x goal not reached. Local-only facts below are
retained as their original evidence scope, not substituted for box resources.

Parent5ec469f; explicit `delivery="state-pipeline"`, mask62960. Control is
`split-prepare`/30192. This changes only recurrent-state staging/synchronization;
prefix, solve, W/U, output, public input layouts and original auto-routing
remain unchanged. The old21 native kernel bodies remain in the same binary.

## Why this candidate

The [verified previous capture](PPU_WY_SPLIT_ACU_20260924.md) measured state
120.933 vs FLA90.408 us at B1/S2048/Hk16/Hv32/K=V128/C64, g=-1. The state
grid, active warps, BF16 MMA count and main traffic are essentially equal.
Our lower barrier count nevertheless has much more synchronization waiting.
This is a load/consume/arrival-schedule question, not another grid experiment.

The old path waits for W AND K before W@H, although K is only needed for the
later state update. It already overlaps U with W@H. Each iteration then
finishes with a CTA barrier and loads the following W at the next iteration.

## Execution and buffer lifetimes

One prologue issues W0. Every chunk then performs:

1. Write BF16 H snapshot while W is in flight; wait for W and synchronize.
   This barrier also retires every previous K/U/scaledV reader.
2. Publish H; issue current K/U and load gate rows. W@H runs before waiting
   for K/U. The following barrier also retires ALL W/H readers.
3. If another chunk exists, load its W into the now-dead W plane and commit.
   Do not wait for this unrelated next W yet.
4. Compute vnew/scaledV with unchanged FP32 subtraction and distinct BF16
   rounding boundaries; synchronize those producers with consumers.
5. Publish vnew, decay register state and compute K^T scaledV in the same
   reduction order. Return to step1 without a redundant end barrier.

The first barrier of the NEXT chunk is before every next K/U overwrite.
Snapshot storage is disjoint from K/U/scaledV; its previous W@H readers
already retired at step2. Thus snapshot writes may begin while other warps
finish the current update. The final iteration never issues W-next, so W's
union alias `final_h` can be used safely after the loop. No extra buffer,
counter, atomic, host synchronization or inter-CTA communication is introduced.

All input tiles keep the native `AIU.swzl -> ld.swzl` pair. Register-born H,
vnew and scaledV keep their proved store/consumer mapping. No load-family
swap, arithmetic approximation or state-precision change is part of this test.

## Local proof and actual native code

| State fact | Control | Candidate |
|---|---:|---:|
| Threads |128|128|
| Shared B |49408|49408|
| Local SDK2.1.1 registers/thread |234|230|
| Stack B/thread |0|0|
| Static instructions |1875|1880|
| BF16 MMA sites per recurrent body |32|32|
| Matrix-load sites |44|44|
| v.mov.v2s sites |34|34|
| CTA barriers per chunk / optional final |4 /1|3 /1|

Static instructions did NOT fall. The hypothesis is overlap, not reducing
MMA work or promising higher occupancy. Fewer barriers alone are not proof
of less time. Native compilation shows7 AIU sites:2 prologue,3 current K/U,
2 next W; dynamically the same5 loads/chunk execute without a final extra
prefetch. All21 old instruction+operand sequences are identical to the
same-SDK admitted baseline.

`check_state_pipeline.py::native_schedule` follows actual branch edges and
the recurrent strongly-connected component, not linear PC order. The compiler
places K/U's wait after12 of16 W@H MMAs (not the source's apparent16). Next-W
has all16 state-update MMAs before the next commit-group wait. The proof
records this scheduling difference explicitly. The native CFG includes a
syntactic last-iteration exit; source/Plan checks separately bind the guard
that prohibits issuing next-W on that iteration. Neither is substituted for
the other. Other scoreboard/pipe-flush operations still exist; earlier issue
is not a measurement of hardware latency successfully hidden.

Local gates:

- Shipping `Plan::next/valid`,65536 sequence/head combinations,4227072
  chunk visits and270532608 tail cells; exact1 W load per chunk.
-66 partial-order lifetime graphs (1..33 chunks, final state off/on),55242
  accesses and30275476 interval pairs. Four warps may progress independently;
  AIU issue-to-completion intervals are not silently completed by CTA barriers.
  Conflicting accesses must be ordered for every permitted interleaving.
-9 host fault plants: missing waits/barriers, outstanding final prefetch,
  wrong tail and stale group. Source binding compares unchanged arithmetic
  blocks and checks actual stage/stream/selector order;11 negative plants.
- Native gate requires the exact new body, zero stack, no register-budget
  increase, native paired delivery and real overlap. Moving the same wait
  immediately after next-W, without changing the opcode multiset, must fail.
  Missing image/link/wait/retirement and wrong-writer negatives also fail.

These are local compile/ownership/lifetime proofs, **not device numerical or
performance results**. Full handoff still runs the original16-case device
gate, scalar-WY RAW-BIT checks, independent2% oracle,8 repeats, tails,
nonzero state, native GVA and output-only. Same-input FLA admission retains
both weak and strong decay. No precision threshold is relaxed.

Final local regression:11/11 compiled host tests,68 Python capture/ABI/
benchmark contracts,7 HGCC architecture contracts,45 algebra cases and305
original-structure controls PASS. Local libraries include22 WY images and
the original15 images; new schedule checks/negative controls are additional
to the old audits, not substitutes for them.

## One box command; all-kernel ACU verdict

From GDN-QSA-sm80 after pulling `ppu-backend`, use the SDK that built the
successful previous run (override PPU_SDK if installed elsewhere):

```bash
DEVICE=0 PPU_SDK=/usr/local/PPU_SDK JOBS=16 \
  ACU=/sim/eec/shared/junfu.qx/asight/bin/acu \
  bash tools/run_ppu_wy_state_pipeline_acu_box.sh
```

The wrapper builds both libraries, checks native bodies, completes device
numerical admission and then directly invokes ACU sequentially for:
**split-prepare / state-pipeline / FLA**. No API timing, PPUProfiler or
launch-count filter. Both our arms have5 kernels; FLA's entire call includes
its fills. Capture g=-1 first; this does not measure weak-gate performance.
Upload the printed `/workspace/gdn-wy-state-pipeline-.../acu.tar.gz`.

Verdict fixed before measurement: compare complete ACU sums; report state
time and unchanged-stage variation separately. If state/total do not improve,
retain the old split path and reject the latency-hiding hypothesis for this
candidate. Do not promote it because source/native ordering looks better.
If it improves, retain it as an experimental winner for this scope only.

User goal remains **FLA sum / candidate sum >=1.5**. Previous sums were
269.306 vs223.583 us, target149.055 us. Reaching state parity alone would give
about238.781 us, still slower than FLA. This experiment alone is not a claim
to finish the target; W/U conditioning/delivery is the next separate candidate.

Local reproducibility: `/workspace/gdn-wy-state-pipeline-evidence-20260924/`
contains plan/ledger, build scripts, actual libraries/native exports and
full regression logs. Raw build artifacts are not in source commits.
