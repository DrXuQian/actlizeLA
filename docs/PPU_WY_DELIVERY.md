# Coalesced WY delivery candidate — all three stages

Parent: `bae4ac7`. Scope and diagnosis are in
[the verified ACU analysis](PPU_WY_ACU_20260921.md). This is a forward-only
delivery change, not a new GDN algorithm, lower-precision state, reset policy,
larger grid or automatic route promotion. The subsequent device verdict is
[traffic target met, speed target not met](PPU_WY_DELIVERY_VERDICT_20260921.md):
both gates passed numerical admission; no delivery ablation established a
speed advantage, and all lost to FLA. The candidate stays opt-in, not default.
The subsequent [compute-tile experiment](PPU_WY_COMPUTE_TILES.md) preserves
these controls and introduces a separate `tiled-*` candidate family.

## Changes and invariants

| Stage | Candidate change | Explicitly unchanged |
|---|---|---|
| Prepare | Native C fragments exchange through warp-private shared scratch; 16-byte W/U stores. Reuse the dead inverse-merge scratch. | Prefix, triangular solve, TF32 high/residual products, W/U arithmetic and BF16 boundaries; 1024 x 128 launch |
| State | Pack incoming H snapshots and Vnew across the whole V32 CTA; cp.async U while computing W@H; vector final-state store. Reuse W scratch only after the final chunk. | 32 sequential chunks, C64/V32, FP32 register state, two MMA products, 128 x 64 launch |
| Output | Warp-private shared exchange and 16-byte output stores, with whole-vector row predicates. | QH + causal PV arithmetic, scale/rounding, 1024 x 128 launch |

State exchange union lifetime is explicit: snapshot production -> CTA barrier
-> vector reads -> CTA barrier -> U prefetch -> W@H -> wait/CTA barrier ->
Vnew + scaled V production -> CTA barrier -> Vnew writeback + KTV -> CTA
barrier. Final FP32 output reuses W only after that loop. Prepare/output
exchanges are warp-private, with warp barriers before reading and before reuse.
There is no cross-CTA barrier or reduction.

Both scalar and packed instantiations are compiled in the same library.
`gdn_chunk_wy(..., delivery=...)` accepts `scalar` (default), `prepare`,
`state`, `output`, and `all`. Each non-scalar mode explicitly selects its
corresponding kernel; there is no fallback to scalar if it fails. The existing
7-argument scalar Python call and C ABI are preserved. Original `gdn_chunk`
and all strong/reset/scan routing stay untouched.

The scalar control is **recompiled**, not presented as the archived dd70e5d
binary. Its prepare machine bytes match the archive. State/output have minor
compiler move-instruction differences (state +2 vector moves / -31 scalar
moves; output -12 scalar moves), with identical MMA counts and register/stack
resources. Thus timing attribution uses the *same-run recompiled scalar*, not
subtraction from the historical 715 us. The archived binary remains immutable.

## Local admission

Real PPU SDK 2.1.1 hgcc compilation and full link, including Python bindings:

| Stage | Scalar registers | Packed registers | Scalar / packed dynamic shared bytes | Stack |
|---|---:|---:|---:|---:|
| Prepare | 84 | 78 | 70,144 / 70,144 | 0 / 0 |
| State | 244 | 240 | 37,120 / 41,216 | 0 / 0 |
| Output | 80 | 60 | 73,984 / 76,032 | 0 / 0 |

Generated code, not source intention: all packed stages have
`vmem.st.b32x4` and no `vmem.st.b16`; packed state has no scalar BF16 global
U loads. Static BF16/TF32 MMA bodies match the scalar control at each stage.
Resource gates reject spills, missing MMA/vector stores, scalar-store
regressions and missing linked launchers.

The new compiled host proof checks 4,080 combinations of complete row-tail
ranges, padded strides and base offsets. Native MMA traits independently
anchor producer fragment ownership; every valid output is owned once and
every invalid/padding cell stays untouched. Twelve negative tests plant
wrong word permutation, one missing transaction, and a tail write. Existing
algebra/reference, native layout, API and original-source preservation gates
remain required. At the original local handoff device tests were NOT_RUN;
the linked device verdict now records the completed box results.

Complete post-edit rerun: **4/4 CTests, 47/47 Python contracts, 45 algebra
cases plus five negatives PASS**; 305 original control expressions unchanged;
original 15 and WY six device images compile/link. Local evidence:
`/workspace/gdn-wy-delivery-20260921/sealed-local.log`.

The useful target for the next ACU ledger is state writes 784 -> 98 MiB;
prepare W/U+gates 512.25 -> 128.25 MiB; output 256 -> 64 MiB at the KVD
interface, using the footprint calibrated by the incumbent capture.
These were **address-footprint predictions** at handoff; the subsequent
all-delivery ACU confirmed all three counts exactly, without a speed win.
Prepare/output still store warp-sized N16 tiles; this first candidate
does not claim their global coalescing is already identical to FLA's full tile.

## One-command box experiment

```bash
git pull --ff-only
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 DELIVERY_AB=1 \
  bash tools/run_ppu_wy_fla_box.sh
```

The existing runner builds for box PyTorch/SDK, keeps original admission,
and checks all five WY deliveries on all 16 numerical cases. Every candidate
must match scalar output **and state bits**, including eight repeats, native
versus expanded GVA, tails, nonzero initial state and output-only mode.
Failure stops timing; the 2% independent recurrent-oracle gate is not weakened.

Timing roles: original, scalar WY, prepare-only, state-only, output-only,
all-packed, FLA. Fourteen balanced sample orders, ten complete calls per
sample, five warmups, gates -0.1 and -1.0; never concurrent. Defaults are
14 samples only when `DELIVERY_AB=1`; overrides must complete that balanced
cycle. Results retain every sample and all winner/loser/unresolved outcomes.
Decisions use disjoint observed envelopes; no trimming or median-only wins.
The pre-existing <=1.10x FLA goal remains unchanged.

Artifacts: printed `/workspace/gdn-wy-fla-<sha>-<UTC>/`, including
`comparison.log/json`, codegen, raw-bit admission, source/diff/submodule and
binding/device-library hashes. Look for `[WY delivery verdict]`, especially
`candidate=wy-all control=wy` and `candidate=wy-all control=fla`.
Terminal PASS means numerics/measurement completed, not a performance win.

For the subsequent counter capture, explicitly select the new path:

```bash
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 \
  bash tools/run_ppu_gdn_fla_acu_box.sh --wy-run /workspace/<completed-run> --wy-delivery all
```

This reuses that run's library without rebuilding, binds the selected delivery
to its admitted comparison arm and records the delivery in both receipts.
Omitting `--wy-delivery` intentionally selects scalar; never compare that
capture as if it were the optimized path. ACU durations and API event spans
are different protocols and remain separate.
