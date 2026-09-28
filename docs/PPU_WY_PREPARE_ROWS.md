# Exact row-factor reuse in prepare

## Incumbent and scope

The latest **user-pasted**, strong-gate (`g=-1.0`) full-public-API result is:

| Role | Median us | Observed range us |
|---|---:|---:|
| Original | 425.572 | 419.268–434.096 |
| Scalar WY | 705.004 | 703.772–709.496 |
| Old tiled state/output, mask48 | 437.840 | 436.820–439.808 |
| State address + row factors, mask240 | 379.088 | 377.868–381.072 |
| Prepare address, mask496 | 356.760 | 355.468–360.424 |
| Output address, mask752 | 380.480 | 379.156–383.756 |
| Both stage addresses, mask1008 | 357.674 | 356.396–361.764 |
| FLA | 487.472 | 475.668–520.584 |

Mask496 lowers latency by 5.89% against mask240 with disjoint envelopes.
The output-only change and the added output change on mask496 establish no
gain: their respective control envelopes overlap. The next incumbent is
therefore **prepare-address + state-both + old tiled output**, not mask1008.
The 1.366x median ratio to FLA describes complete API calls, not kernel-only
speed. Full raw JSON/binary/device identities and weak-gate results have not
been received locally; do not infer them. Original final state is BF16,
WY/FLA final state is FP32. No default or automatic routing is promoted.

This task changes only `expf(prefix[row])` reuse during prepare's K
conditioning. It does not change prefix scan order, pairwise
`expf(prefix[r]-prefix[c])`, the TF32 high/high + two residual inverse,
BF16 boundaries, W/U arithmetic, state or output. In particular,
`(float(K) * beta) * factor` remains left-associated: caching `beta * factor`
instead would not preserve FP32 bits.

## Two mechanisms, one unchanged incumbent

| Delivery | Full-call mask | Mechanism | Tradeoff |
|---|---:|---|---|
| `stage-address-prepare` | 496 | Same row exponent evaluated for all 128 columns | Existing control |
| `prepare-rows-shared` | 1520 | 64 row owners cache FP32 factors in dead inverse scratch | One additional CTA barrier and shared reads |
| `prepare-rows-warp` | 2544 | Two FP32 factors/lane in each warp; full-mask indexed shuffle | Redundant computation across four warps and one shuffle/iteration |

Shared lifetime: the inverse merge's last CTA barrier retires all `temp`
readers, then threads 0–63 write `temp[0][row]`. The independent inverse
BF16 conversion runs before a new CTA publication barrier. All K columns
read their row factor after that barrier. The existing conditioning-complete
barrier retires the cache before W/U. No additional shared allocation.

Warp lifetime: every lane computes rows `lane` and `lane+32`. All 32 lanes
participate in all 64 conditioning iterations. Choose a register before the
shuffle; there is no dynamically indexed register array and no new CTA
barrier. The factor stays FP32 through either delivery method.

For this conditioning term alone, the source and native loop imply **256
warp exponent executions/CTA before, 2 shared-cache or 8 warp-cache** after.
These are decomposition predictions, **not measured ACU counters**. Pairwise
gate exponents are unchanged, so these are not whole-kernel reduction ratios.
Warp-cache adds 256 warp shuffle executions/CTA. Neither mechanism has a
device speed result yet; the barrier/shared or shuffle cost can erase a gain.

Modes live in `gdn_wy_prepare_rows_ppu.cu`; the arithmetic authority remains
`prepare_inverse<Address, RowCache>`, default `RowCache=0`. Bits1024/2048
are mutually exclusive and require PrepareAddress256. One typed visitor
selects both the resource-attribute target and actual kernel launch.
The public unsigned C ABI is unchanged. Independent Cartesian census:
6 prepare ×6 state ×4 output =144 valid masks among8192 tested values,
including unchanged old27/54/96 subsets. Missing/conflicting options fail.

## Local evidence and limits

PPU SDK2.1.1-a5c56e compiles and links all16 WY native images. All **14 prior
instruction and operand sequences** match the immutable same-SDK parent.

| Prepare body | Registers | Stack B | Static instructions | Static exp sites | CTA barrier sites | Indexed shuffle sites |
|---|---:|---:|---:|---:|---:|---:|
| Address control | 86 | 0 | 2431 | 9 | 6 | 0 |
| Shared rows | 86 | 0 | 2440 | 9 | 7 | 0 |
| Warp rows | 84 | 0 | 2447 | 10 | 6 | 1 |

Each retains 16 BF16 MMA, 12 TF32 MMA and 16 async-copy **static sites**.
Do not mistake unchanged/increased static exponent sites for failure: the
optimization removes repeated evaluation from the rolled conditioning loop.
The audit uses actual native backward-branch targets between the last TF32
inverse product and first BF16 W/U product, not a hardcoded PC or BB number.
It finds one conditioning loop, zero cache-mode exponents inside it, and
one/two exponent sites outside **all** backward loops. Shared's store and
publication barrier must precede consumption; warp's shuffle must be inside
conditioning. Parser negatives move the actual exponent or publication
barrier across that boundary **without changing opcode totals**; both fail.
Those mutations are ISA-text validator tests, not executed device images.

Host gates cover 512 chunks (8 distinct fixtures × all valid tails1–64),
32,768 shared producers,131,072 warp producers and8,388,608 consumers.
Factors and left-associated products must match FP32 bits and the BF16
boundary. They reuse the production ownership helper, with independent
ownership/denominator expectations. Nine plants cover wrong row/half, missing
warp participant, late publication, beta reassociation, ignored/conflicting
mode, omitted denominator and rounding the factor to BF16. Native CFG checks
complement the host ownership model; neither claims to prove device races.

The complete seal includes eight CTests,63 Python contracts,45 CPU algebra
cases + five negatives,305 preserved original controls and21 native-validator
negatives. Local logs: `/workspace/gdn-wy-prepare-rows-evidence-20260922/`.
The full unchanged Torch binding is **SKIP/environment locally** (CPU-only
Torch lacks CUDA generated headers/libraries), separately from completed
native device-library compilation/linking. It is built and admitted on box.
New device raw equality, race/replay and performance remain **NOT_RUN**.

## Run and decide

From `GDN-QSA-sm80`, on the same physical PPU as the control:

```bash
git pull --ff-only origin ppu-backend &&
env -u OUT PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 PERF=1 \
  PREPARE_ROWS_AB=1 STAGE_AB=0 STATE_AB=0 TILE_AB=0 DELIVERY_AB=0 \
  SAMPLES=16 LAUNCHES=10 WARMUP=5 bash tools/run_ppu_wy_fla_box.sh
```

Eight sequential roles: original, scalar WY, mask48, mask240, incumbent496,
shared1520, warp2544 and FLA. Both gates `-0.1/-1.0`,
B1/S2048/Hk16/Hv32/K128/V128/C64, zero initial state and returned final state.
Sixteen balanced samples,10 complete API calls/sample,5 warmups, no trimming
or overlapping workloads. Old experiment inventories remain available.

Before timing:16 device cases, independent2% numerical oracle, candidate
output **and state raw equality** to scalar WY,8 repeated launches, tails,
GVA, nonzero initial state, FP32 gates and output-only. A failed admission is
FAIL, not an unresolved timing cell. Each cache mode compares directly with
incumbent496 and the other cache mode. The unchanged disjoint-envelope rule
determines candidate/control wins; overlap means UNRESOLVED. If both lose or
are unresolved, retain496. Do not promote from only one strong-gate result.

Results and identities are in the printed workspace's `comparison.log`,
`comparison.json`, codegen/correctness logs, source diff and binary hashes.
`[WY verdict] subject=wy` is still the old scalar comparison; use
`[WY delivery verdict] candidate=wy-prepare-rows-* control=wy-stage-address-prepare`
for this experiment. ACU reuse accepts the exact new role without rebuilding;
equal output fingerprints never authorize rebinding a different role.

## User-reported strong-gate result, 2026-09-22

Superseded evidence status: the [verified ACU/comparison bundle](PPU_WY_SHARED_ACU_20260922.md)
now includes both gates, complete numeric admission and exact binary/device
bindings. The paragraph below preserves the earlier excerpt-only review.

The returned excerpt contains all eight roles with16 finite samples each.
Recomputed medians match the printed values to0.001 us. It does not include
the complete comparison JSON, numerical admission log, source/binary/device
identities or weak-gate result. The pasted warp label says
`wy-prepare-row-warp` (singular), whereas this source emits
`wy-prepare-rows-warp`; it is not treated as a verified raw result bundle.

| Full-API role, g=-1.0 | Median us | Range us | Against incumbent496 |
|---|---:|---:|---|
| Prepare address,496 | 356.590 | 355.552–357.508 | Control |
| Shared rows,1520 | 352.192 | 351.240–353.872 | CANDIDATE-WINS |
| Warp rows,2544 | 354.350 | 352.660–357.468 | UNRESOLVED |
| FLA | 494.490 | 478.012–522.352 | Reference |

Shared saves4.398 us (1.233% latency reduction), with a1.680 us separation
between its maximum and the incumbent's minimum. Shared versus warp is
also **UNRESOLVED** because their ranges overlap: this is evidence for a
gain against496, not proof that shared delivery beats warp delivery.
Shared versus FLA has disjoint ranges; its1.404x median speedup is a full-API
comparison (28.78% lower latency), not a kernel-only throughput claim.

The older controls remain consistent in this cohort: original426.348,
scalar WY705.266, mask48=437.158 and mask240=379.174 us. The final
`subject=wy control=original ORIGINAL-WINS` describes scalar WY, not the
new candidate. Shared is retained as a **strong-gate experimental** incumbent;
default routing remains unchanged, and no weak-gate or global admission is
inferred. Do not subtract older ACU stage durations from these API spans.
The net4.4 us gain alone cannot attribute barrier, shared-read or shuffle
costs; a current same-binary stage profile is needed for the next bottleneck.
