# PPU1.0 full-prefix / final-tail state experiment

Parent `f967cfe87251c7559f07bb30a39c49e4163a9d1d`.
Branch `agent/ppu10-mixed-tail-20260928`.
Plan registered 2026-09-28 09:38:50 UTC before the first kernel edit.
Goal: architecture-independent SM90 migration completeness, not the suspended
1.5x FLA target. This is an explicit experiment; public/default routing stays
unchanged. The previous first-chunk candidate is not composed into it.

Device follow-up is now complete; see
[the four-cell ACU verdict](PPU10_MIXED_TAIL_ACU_20260928.md).
The registration and local-only evidence below are retained as written;
they are not a substitute for that later measured record.

## Scope and implementation

Previously `full-chunk` delegated the **entire call** to the generic state
kernel whenever `S % 64 != 0`. For S2049, all32 complete chunks therefore
paid tail checks even though only chunk32 had one valid token.

`delivery="mixed-tail"` keeps the existing four-stage call and replaces only
the state kernel for mixed lengths:

| Sequence | Selected state |
|---|---|
| Positive full C64 length, including S2048 | Exact existing full-chunk state symbol |
| Short partial length S1..63 | Exact existing generic fallback |
| S>64 with nonempty tail | Constant-full prefix loop, then one guarded tail, within the same CTA |
| Invalid extents | Existing admission rejection |

The same FP32 state registers cross the loop/tail boundary. There is no extra
launch, FP32-to-BF16 state handoff, global intermediate state, reset or
first-chunk skip. Supplied initial state and output_final_state=False retain
their contracts. Inverse storage remains disjoint from H snapshots; the
original group stride includes **all** ceil(S/64) chunks for every batch/head.

One templated iteration body has Full=true/false instantiations. Restoring
only its compile-time validity choices gives byte-normalized equality with
the admitted full/generic iteration bodies. Initialization, shared owner
maps, staging, barriers, publication and final state are separately bound
to the unchanged control. A shared helper does not replace this proof.

Prefix, static inverse and HV-layout output keep their original symbols.
This closes the **state** interior/tail mechanism, not a claim that the
prepare/output full variants have gained admission on mixed lengths.
Their full-C64 candidates did not establish a stable performance benefit.

## Local evidence, not a device verdict

Native HGCC2.1.1/PPU1.0 compilation plus Python extension link passed.
All42 old native instruction/operand sequences and resource/ABI records are
identical to the same-SDK parent build. No shared build directory was edited.

| State body | Generic control | Full-only control | Mixed-tail candidate |
|---|---:|---:|---:|
| Total static native sites | 1,166 | 1,085 | 1,909 |
| Recurring full-prefix component sites | 485 | 411 | 415 |
| Prefix s.min / v.csel sites | 1 / 8 | 0 / 0 | 0 / 0 |
| Registers/thread | 124 | 120 | 122 |
| Stack bytes | 0 | 0 | 0 |
| Shared allocation bytes | 46,080 | 46,080 | 46,080 |

Actual native CFG has one cyclic20-MMA prefix component and an acyclic20-MMA
tail. Each retains4 AIU load sites,28 normal+8 transposed ld.swzl sites and
the same math. Four retirement/ready barriers per iteration plus final
publication remain. Native control flow cannot return from a tail MMA to
the prefix. The larger static body is an explicit possible cost, not free
specialization. Static sites do not predict dynamic instruction savings or
latency. The box compiler/resources are authoritative for its own build.

Local checks include:

- 65,537 lengths [0,65536], signed extent boundaries, 64,449 mixed lengths,
  32,997,888 full-prefix steps and64,449 tails, exact once. Actual actlize
  C-layout owners checked for all63 nonempty tail extents; five plants reject.
- 276 CPU cases: all63 tail extents x both gates x None/nonzero state, plus
  full/short fallback and S2049/S2111. RAW-BIT residual algebra and unchanged
  independent2% recurrent oracle. Reset, extra state rounding and missing
  tail negatives reject.
- Source/native/linked-ABI plants reject missing prefix/tail, wrong tail
  count, changed rounding, lost retirement, aliased scratch, stale launch,
  wrong binding, missing math/copies and an ignored shape request.
- Shape travels through admission, child CLI, fixture and receipt. An old
  receipt without shape only admits historical S2048; it cannot certify a
  mixed-tail request. Same-gate different-sequence cases remain distinct.

Compilation/host gates are not device execution. The SDK runtime needs
GLIBCXX_3.4.32; the available newer C++ library in turn needs GLIBC_2.38,
unavailable on this host. Device numerics and performance remain NOT_RUN,
with runtime-import separately SKIP. No H800/PPU job was launched.

## Fixed device admission and performance inventory

Keep the existing36-case x8 gate for both control/candidate. Add261 cases x8:
all63 tails at B1/Hk1/Hv2 with both gates and both initial-state modes;
S2049/2111 at Hk16/Hv32 with both gates/state modes; one B2/Hk2/Hv4/S191
variable-beta/FP32-gate stress case. Reuse the independent2% oracle and
residual RAW-BIT check, GVA expansion, output-only and input immutability.
No tolerance changes. Any failure stops before capture.

Four performance cells, fixed before device data:

- B1, S in {2049,2111}, Hk16/Hv32, K128/V128, C64, g in {-1,-0.1}.
- BF16 inputs, FP32 final state, initial=None. Native FLA head mapping.
- Same binary/input/device control=`residual-full-chunk`, candidate=
  `residual-mixed-tail`, reference=FLA. No API-event speed claim.
- Full-call ACU kernel sums. Our calls must contain4 kernels each. FLA's
  existing full-length call has7 including two fills; count **every** actual
  tail-shape helper/fill if its inventory differs, never truncate to7.
- Verify 128 state CTAs x256 threads,33 iterations for both priority
  shapes,32 full+1 tail, and identical useful matrix work versus control.
  Predicted state BF16 warp-MMA count is675,840 for each arm/shape/gate.

A lower complete-call time in all four captures is only an observed gain
pending repetition/non-regression before routing. Mixed/slower signs retain
the control and are reported individually. Missing cells, incorrect results,
spills, compatibility repair or absence of the constant prefix invalidate
that candidate. No substantial speedup is promised; S2048 is unchanged.

## Run and upload one file

From this branch on an idle PPU with FLA installed:

```bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_mixed_tail_box.sh
```

The script builds once, runs admission, invokes
`/sim/eec/shared/junfu.qx/asight/bin/acu` for the four paired cells, then prints
one upload path: `/workspace/actlizeLA-ppu10-mixed-tail-.../mixed-tail.tar.gz`.
The tar contains all four raw-report bundles, sources, binaries/hashes,
device/SDK identity, host/native/numerical receipts and measurement SHA.
No manual CSV copy/paste is needed. It rejects incomplete evidence and never
overwrites an existing archive. Repack a completed run without rerunning:

```bash
bash tools/pack_ppu10_mixed_tail.sh /workspace/ACTUAL_RUN_DIRECTORY
```

SDK precedence is PPU_SDK, PPU_SDK_ROOT, then /usr/local/PPU_SDK. Use a fresh
OUT directory (mkdir, not mktemp). Invoke scripts with bash, not source.

After this item: resolve inverse register retention and paired-conversion
applicability, then shape-driven PPU selection. This work does not make
Hopper-specific TMA/WGMMA/warpgroup features portable to PPU1.0.
