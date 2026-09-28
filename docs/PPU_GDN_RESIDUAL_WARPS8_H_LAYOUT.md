# Paired H layout: state, snapshot publication and output

Experimental delivery `residual-warps8-hlayout`, based on `27e47c7`.
Performance control: **residual-warps8-blayout**, not the ineffective
UPDATE-only lookahead and not a different warp geometry. Default routing
and every existing kernel body remain unchanged.

Device follow-up at14c7268 is now recorded in
[the ACU report](PPU_GDN_RESIDUAL_WARPS8_H_LAYOUT_ACU_20260925.md):
all-kernel205.732→194.466us,FLA224.350us; both changed stages improve.
The local-only claims below remain historical compile/proof scope, not
retroactively relabelled device measurements. Default routing is unchanged.

## Contract and why both kernels change

H is not like the internal-only residual/scaledV planes. It has a state-MMA
reader and a globally published snapshot consumed by the output kernel.
Simply placing H in B orientation makes canonical `[K,V]` vector publication
noncontiguous. This experiment instead changes the **private** snapshot
workspace to `[V,K]` and pairs its output reader with that format:

```
FP32 H[K,V] accumulators
  -> same BF16 rounding, B-oriented16x16 shared microcubes
       -> native nontransposed KH matrix reader
       -> contiguous16B stores to private snapshots[group,V,K]
            -> output's matched AIU.swzl / nontransposed ld.swzl reader
```

No second H view, extra transpose kernel, lane shuffle or scalar-gather
publication is introduced. The private workspace allocation size is unchanged.
Public Q/K/V, BF16 output, FP32 initial/final H layouts remain unchanged.
The residual rounding boundaries and all K accumulation orders are unchanged;
RAW-BIT equality with scalar residual remains required, not just tolerance.

State still uses V32/grid128x256 at B1/S2048/Hk16/Hv32/K=V128/C64.
Output remains grid1024x256. K, inverse, residual/scaledV, Vnew, gates,
barriers, initial/final-state publication, prefix and solve are unchanged.
The new state and output live in a separate TU, with one paired launcher.
Using the old output with the new private snapshots is forbidden and tested.

Offsets and strides are BF16 **elements**, never bytes:
`state_offset(group) + v*128 + k`. The caller offsets each state's V slice
by `v0*128`, and each output panel by `panel*128`. All slices/groups publish
exactly once. A dedicated64-bit offset check covers large group indices.

## Cost, not a claim of a free transpose

Native matrix loads remain one-for-one and the shared/global byte extents
are unchanged. The output's physical H tile changes128x64 to64x128; the
native128B cube-width constraint makes it **two64x64 AIU cubes instead of
one128x64 cube**. That adds one bulk-copy instruction per output panel
(two per CTA at this shape), not extra logical bytes. Actual transaction
bytes and cache/request behavior still need ACU measurement.

Local SDK2.1.1/PPU1.0 exact-body compilation:

| Property | B-layout control | Paired H layout |
|---|---:|---:|
| State registers/thread; stack | 124;0 | 122;0 |
| Output registers/thread; stack | 94;0 | 78;0 |
| State shared bytes | 45,568 | 45,568 |
| Output shared bytes | 49,408 | 49,408 |
| State static instructions | 1,313 | 1,215 |
| State recurring backedges | 516/530 | 512/526 |
| Output static instructions | 1,442 | 1,418 |
| Output panel backedges | 251/258 | 247/254 |
| State matrix-load sites, normal/trans | 20/16 | 28/8 |
| Output matrix-load sites, normal/trans | 36/24 | 52/8 |
| State BF16 MMA /AIU /barrier sites | 20/4/5 | same |
| Output BF16 MMA /AIU /barrier sites | 40/6/6 | 40/7/6 |

The entire static reduction is not all recurring savings. Registers alone
do not prove an occupancy increase; account for shared limits, grid supply
and actual eligible warps. These facts do **not** establish lower BC,
latency improvement, or the1.5x FLA target.

## Local proof and negative controls

`l029_wy_residual_warps8_hlayout` uses actual native C/B MMA traits, the
production writer/publication coordinates, actual output AIU descriptors
and actlize's independent native matrix-load simulator. Full coverage:
32,768 H producer values,131,072 state-reader values,4,096 contiguous16B
global vectors,131,072 output-reader values, all8warps/32lanes/4V-slices/
2groups and both output panels. Tags uniquely encode group/K/V. Global
exact-once and unchanged snapshot byte extent are checked independently.

Ten negatives cover stale shared placement, one-bit swizzle, wrong state
cube, **old canonical publisher**, **old output address**, wrong output
cube, omitted warp/slice/output-K/group. Missing output K is rejected by
the fixed denominator even when all remaining read values are correct.
The pre-existing AIU gate separately proves every valid row tail for the
same64x128 writer/reader geometry.

The source gate reverses only the registered state/output H edits and
requires exact old-body equality elsewhere. Eight source negatives include
old output dispatch, changed rounding and missing retirement. During gate
development the old-output-launch plant initially escaped a permissive
normalization: replacing new names with old names also accepted already-old
names. The checker now requires the new paired call **before** normalization;
this meaningful negative remains part of the suite.

Native checks cover both whole bodies and actual recurring/panel backedges,
all useful math, ordinary shared/global loads and stores, waits, barriers,
the deliberate AIU+1 and exact normal/trans load replacement. Six native
plants remove/change the reader, vector publication, AIU, wait or barrier.
The library inventory is32 images:30 old controls plus the new state/output.
All30 old native instruction/operand sequences must compare identical.

Complete local regression finished with exit0:20/20CTest,89host contracts,
7compiler-dialect tests,45WY+61residual algebra cases and305original source
checks.32WY and15original native images are linked/audited;30/30 old WY
instruction-and-operand sequences are identical to the prior build. The
original experimental C32 image retains its separately reported144B stack;
neither new H kernel nor any production C16 image spills. Device numerics,
counters and performance are **NOT_RUN** locally and must not be relabelled
by the host tests. The final complete rerun also exited0; authority:
`final-local-complete.log`, not isolated reruns.

Compile-only DSO SHA256:
`0d2633c415be8d54a8de1d29dd65be10f476db3f26ccac0350e85fc0fe3ce170`;
binding SHA256:
`a5cc49852044345411a78ef911079ba58910fd65b137cd7573439bc68f9ec131`.

## Box command and fixed interpretation

After pulling the published change in `GDN-QSA-sm80`:

```bash
git pull --ff-only &&
DEVICE=0 JOBS=16 CANDIDATE=residual-warps8-hlayout \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Keep the working SDK environment (or set `PPU_SDK` explicitly). The runner
uses `/sim/eec/shared/junfu.qx/asight/bin/acu` and creates a fresh
`/workspace` bundle; it never uses PPUProfiler or API-event timing here.
It first builds/gates the exact library, then runs30 scalar and30 cases per
selected delivery with8 repeats, tails/GVA/nonzero-state/output-only and
independent2% recurrence checks. Both gates are admitted before default g=-1
capture. A missing candidate backend cannot fall back to an old kernel.

Three same-binary/fixture/device captures run sequentially: B-layout
control, paired-H candidate, FLA. Count **all4/4/7 kernels** including FLA
fills. Candidate state/output symbols must contain `warps8_hlayout`; the
control retains its old state/output symbols. Report state and output
separately because both changed; prefix/solve variation is not their gain.

Judge the complete ACU kernel sum at matched clocks and correctness. Lower
H conflicts plus lower time support the proposed mechanism; lower BC or
registers without lower time reject the performance hypothesis. Extra AIU
or publication/cache costs count against the candidate. Neither result
changes the1.5x target or automatically promotes a default. If it wins,
confirm against the retained non-B eight-warp incumbent before routing.

Upload the final printed tar; it contains native/resource, numeric and
source/binary/fixture receipts. Source worktree:
`/workspace/gdn-wy-warps8-hlayout-20260925`; local evidence:
`/workspace/gdn-wy-warps8-hlayout-evidence-20260925`.
