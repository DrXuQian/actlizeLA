# Eight-warp B-layout: independent UPDATE operands

Parent `db8acc8`. Experimental only: default routing, math, precision and
H/K layouts are unchanged. This is an operand-schedule test, not another BC
layout or occupancy change. The control is **residual-warps8-blayout**.

## Question being tested

The preceding same-binary ACU capture reduced BC26.3% but not time. It also
changed native operand allocation and exposed more compute/TFU-WAR waiting.
The full account is in
[the B-layout result](PPU_GDN_RESIDUAL_WARPS8_B_LAYOUT_ACU_20260924.md).
Here only `K^T @ scaledV` gets a two-slot register lookahead. The same native
AIU/SWZL pair, scalar/shared writers, V32/grid128x256, shared45568B, chunk/K
order, output publication and all BF16 boundaries stay fixed. The existing
host-proved `prefetch_atoms` helper is reused, not reimplemented.

The loop loads the next atom into the other register slot before consuming
the current atom. No extra shared plane, copy, barrier or MMA is introduced.
Both output K fragments still accumulate K atoms in their original order.
There is no register cap or new assembly primitive.

## Local SDK2.1.1 native result

| Property | Eight-warp B control | UPDATE lookahead |
|---|---:|---:|
| Registers/thread / stack bytes | 124 /0 | 124 /0 |
| Grid / block threads | 128 /256 | 128 /256 |
| Shared bytes | 45,568 | 45,568 |
| BF16 MMA / matrix-load / AIU sites | 20 /36 /4 | 20 /36 /4 |
| CTA barriers / async completion sites | 5 /1 | 5 /1 |
| Whole-body static sites | 1,313 | 1,315 |
| Recurrence backedge static sites | 516 /530 | 510 /524 |
| UPDATE: next load immediately reuses MMA inputs | 3 | 0 |
| Other phases: immediate input reuse | 10 | 10 |

UPDATE B-load lead, measured as intervening native MMA instructions before
consumption, is `[0,1,0,1,0,1,0,1] -> [0,1,2,3,2,3,2,3]`.
These are native scheduling facts, **not retirement cycles, measured WAR
elimination, a latency win or a1.5x FLA claim**. Actual clocks, BC, occupancy,
eligible warps and stall counters still need the box capture.

The bounded compile screen also tried applying two slots to all three GEMMs,
four slots, and an empty-assembly operand-liveness constraint. KH/PR native
delivery did not improve; the deeper/liveness forms produced the same native
schedule as the two-slot form. Those ineffective changes were removed. The
final source changes only UPDATE; do not describe H projection or P@R as
optimized. No empty-assembly hack remains.

## Local gates and negative controls

`l028_wy_residual_warps8_operands` exercises the actual production template
for all8 warps/32 lanes/two output fragments/four K atoms/four operand words:
2,048 MMA consumers and16,384 input-word comparisons. Independent tags bind
K/order/slot/lifetime, not only a potentially canceling accumulated sum.
Eight negatives cover wrong slot, missing/extra K, missing fragment, wrong
word, missing lane, missing warp and premature slot overwrite.

The source gate reverses only the registered UPDATE edit and requires exact
control equality everywhere else. Six source plants must fail. Native gates
compare math/transfer/barrier counts in the entire body and both recurring
backedges, track physical register words, and classify KH/PR/UPDATE using
both native input orientations. **Replacing the candidate body with the
control body—same useful work—must fail specifically for unchanged native
lookahead.** Four additional native plants remove MMA/wait/barrier or change
the transpose. A source-only double-buffer declaration cannot pass this gate.

The linked library inventory is30 images, including29 unchanged controls.
Check their actual instruction/operand sequences against the previous build;
the fixed denominator must change29->30, not silently skip an old body.
The complete rerun passed19/19CTest,87host contracts,7dialect tests,
45WY+61residual algebra cases and305original source controls. All30WY and
15original native images are linked/audited;29/29 old native instruction and
operand sequences remain identical. The initial rerun caught the stale
28/29 image denominator; after fixing it to29/30 the **entire** local suite
was rerun, not just that individual check. See `final-local-complete.log`.
No local device execution was performed; the box must still run all30 scalar
and30 cases per selected delivery, each8 repeats, plus both-gate independent
recurrence/GVA/tail/nonzero-state/output-only checks before profiling.

## Box command and preregistered interpretation

In `GDN-QSA-sm80`, branch `ppu-backend`, after the published commit is pulled:

```bash
git pull --ff-only &&
DEVICE=0 JOBS=16 CANDIDATE=residual-warps8-operands \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Keep the already working SDK, or set `PPU_SDK` explicitly. The runner writes
to a fresh `/workspace` directory and uses
`/sim/eec/shared/junfu.qx/asight/bin/acu`, not PPUProfiler or a PATH substitute.
It selects the exact B-layout control and new candidate from one library;
missing candidate symbols and wrong controls fail instead of falling back.
The final printed tar includes sequential control/new/FLA captures with
complete4/4/7-kernel inventories and source/binary/fixture receipts.

Judge the complete ACU kernel sum, then state time, at matched clocks and
correctness. Report unchanged-stage variation separately. Lower UPDATE WAR
and higher readiness would support the mechanism; unchanged time despite the
verified schedule change would reject it as an effective speed optimization.
Neither outcome changes the acceptance target or silently promotes routing.
If a gain appears, confirm against the retained non-B eight-warp incumbent
before any production promotion; the B control isolates this experiment,
not a claim that it previously beat that incumbent.

Local source worktree: `/workspace/gdn-wy-warps8-operands-20260924`.
Compile/gate artifacts and rejected native probes:
`/workspace/gdn-wy-warps8-operands-evidence-20260924`.
Local compile-only DSO SHA256:
`893a87c3f099ad47b3f4748daa4dad513d7e0b637ce47de742b69e3ec5534377`;
binding SHA256:
`ffc3999084453174edf368631d5fefa356a48df741c44cc56eefcc02b8c503ea`.
H/snapshot paired placement and K's two-consumer layout remain subsequent
independent experiments, not bundled into this change.
