# WY delivery device verdict: traffic target met, speed target not met

## Evidence and limits

Uploaded archive: `gdn-qsa-acu-20260921T123213Z-3755721.tar.gz`.
SHA256: `02b13c62c4a1a1683a90f24fff20e0ace404984158c9476daf55f6df1f108e77`.
All **537 members**, including the complete checksum denominator, validated.
No uploaded code or binary was executed locally. Reproducible local analysis:
`/workspace/gdn-wy-delivery-acu-20260921/analyze_bundle.py` and `parsed.json`.

The capture is **all-delivery, g=-1.0**, not scalar. All three actual kernel
names contain `<true>`. Loaded extension/library hashes match the archived
comparison binaries and the source origin `1b1f508`; preceding source diff is
empty. WY and FLA have identical input/reference identities and GPU UUID
`019ee024-8860-091c-0000-0000007aff6d`. All stages report 1.700 GHz. There is
no same-run scalar ACU archive in this upload. The older scalar archive uses
`dd70e5d`, g=-0.1; it is structural context, **not a controlled latency A/B**.

The included preceding comparison now supplies **both** gates. Its medians,
all 14 samples per role, envelope verdicts and output fingerprints were
independently checked. The numerical log has 16 cases, four candidates per
case (64 admissions), five total WY modes and eight repeats. No denominator
was inferred from a single final PASS line.

## Full-API timing verdict (unprofiled)

Times in microseconds. No outlier removal; disjoint observed envelopes remain
the rule. Every delivery ablation is UNRESOLVED versus scalar and loses to FLA.

| Gate | Original | Scalar WY | Prepare | State | Output | All | FLA |
|---|---:|---:|---:|---:|---:|---:|---:|
| -0.1 | 918.446 | 717.626 | 715.980 | 715.266 | 720.648 | 724.098 | 496.814 |
| -1.0 | 425.240 | 710.058 | 708.476 | 714.356 | 713.394 | 719.990 | 495.610 |

Do not promote the candidate. Preserve scalar WY and the original route.
Original beats FLA for the strong fixture with disjoint envelopes, but not
for the weak fixture. This remains the existing numerical-error contract:
original final state is BF16, WY/FLA final state is FP32; it is not an equal
final-state-precision comparison or permission to narrow WY state precision.
Terminal PASS means numerical admission and measurement completion, not speed.

## Matched all-delivery versus FLA ACU

These are **profiled kernel durations**, not full-API event spans. The FLA
prepare counterpart sums prefix + KKT/solve + W/U. Its two allocation-fill
kernels are separately 4.36706 and 2.18059 us.

| Stage | All-delivery us | FLA us | Duration ratio | All executed instructions | FLA executed instructions | Instruction ratio |
|---|---:|---:|---:|---:|---:|---:|
| Prepare | 164.10647 | 82.14588 | 2.00x | 46,330,880 | 17,228,800 | 2.69x |
| State | 443.36706 | 91.87941 | 4.83x | 38,941,056 | 10,040,320 | 3.88x |
| Output | 86.58000 | 44.78647 | 1.93x | 19,608,576 | 9,166,848 | 2.14x |

Kernel sums: 694.05353 versus 225.35941 us (including the two FLA fills).
Do not subtract these from the API medians to claim a measured host-overhead
component; they are different capture/timing protocols. All three stages
remain optimization targets, not just state.

### The write-traffic prediction was correct

Unit is MiB of **KVD-interface global-store request bytes**, not DRAM bytes.
Metric: `kvd__bytes_pipe_lsu_mem_global_op_st.sum`.

| Stage | Archived scalar, different gate/build | Predicted candidate | Measured all candidate |
|---|---:|---:|---:|
| Prepare | 512.25 | 128.25 | 128.25 |
| State | 784 | 98 | 98 |
| Output | 256 | 64 | 64 |

These are structural counts; the table does not establish a latency delta
between the old archive and this candidate. Current state writes equal FLA
exactly: **102,760,448 bytes and 102,400 warp store instructions** on each.
State BF16 MMA also matches exactly: **524,288** on each. Yet duration is
4.83x and total instructions 3.88x. Thus matching this write-traffic quantity
is demonstrably **insufficient** to match FLA's performance.

The predicted global U load removal is real: state ordinary global reads are
1 MiB (gates), versus the archived scalar's 257 MiB (256 U + 1 gates).
Asynchronous global-to-shared reads increase from 128 to 144 MiB; FLA has
164.56 MiB on that path and 2 MiB on the ordinary path. Never count only one
path and call it the total input bandwidth.

### The exchange has a cost, and the original computational mapping remains

Compared with the archived scalar's structural counts, candidate state TSM
load instructions rise **1,056,768 -> 1,421,312**, stores
**270,336 -> 1,073,152**, and total instructions
**34,556,288 -> 38,941,056** (+12.69%). Output total instructions rise
18,479,104 -> 19,608,576 (+6.11%); prepare falls 2.08%.
These are not a same-run time attribution. New shared exchanges replace the
scalar global epilogues; they do not remove the rest of the MMA delivery work.

In particular, the state kernel writes its BF16 H snapshot to shared for
publication, then **still** uses `state_to_b` register shuffle/repack for W@H.
That shared snapshot is deliberately overwritten by U before it could serve
W@H. Source sites: `gdn_wy_ppu.cu` snapshot/exchange lifetime and W@H loop;
`wy_mma.cuh::state_to_b`. Snapshot coalescing did not optimize this consumer.
Per source, each 16x16 H conversion has 16 shuffle calls; eight H fragments,
32 chunks, two warps and 128 CTAs imply 1,048,576 warp-level shuffle calls.
That is a **source operation count**, not an ACU opcode count or an attributed
number of microseconds. The export does not contain a dynamic per-opcode table.

State still launches 128 x 64 threads versus FLA's 128 x 128, with achieved
3.55 versus 7.10 active warps/CU. Our resource limits allow six shared-memory
blocks and eight register-limited blocks/CU; only 1.78 CTAs/CU are launched
on average. Therefore 240 registers is not evidence that this grid is
occupancy-limited by registers. Doubling threads without redistributing work
would not fix that computational mapping.

Output still makes small warp-private N16 products in an eight-iteration N
loop, reloading Q fragments per output column tile. FLA expresses a larger
distributed output tile and reuses loaded Q for QH and QK. Current output has
64 versus 16 MiB KVD write requests, 1,294,336 versus 491,520 TSM load
instructions, 128 versus 256 threads/CTA, and 76,032 versus 49,152 shared
bytes. Its active warps/CU are 11.74 versus 22.86; the difference is broader
than the global-store epilogue.

Prepare is not yet an identical arithmetic workload: BF16 MMA matches
344,064, but TF32 MMA is **98,304 versus 32,768**. Our `tf32_product` computes
high/high plus two high/residual terms; the captured FLA solve requests plain
`input_precision='tf32'`. This precision distinction is explicit: do not
silently discard the residual terms and call it a pure delivery optimization.
Our fused prepare reserves 70,144 shared bytes (3 blocks/CU); FLA's split
solve and W/U reserve 6,144 and 25,600 with different warp distributions.

## Next bounded changes, not performance claims

1. **State consumer mapping:** distribute H across four useful warps, retain
   FP32 state, and make the existing BF16 snapshot usable by the following
   W@H AIU load instead of retaining both publication exchange and register
   shuffle conversion. Prove shared lifetimes and per-output K order before
   implementation; do not merely change the launch thread count.
2. **Output compute tile:** reuse Q across QH/QK and several N fragments;
   align the CTA-wide output ownership and coalescing. Require unchanged
   arithmetic/rounding and isolated output-stage timing.
3. **Prepare decomposition:** distinguish solve cost from W/U tiled GEMMs;
   reduce fragment reloads and excessive shared lifetimes. Any plain-TF32
   comparison must be an explicit separate precision candidate with the
   unchanged independent numerical gate, not a hidden relaxation of raw
   equality to the current high/residual path.

These steps attack producer/consumer tile layout, reuse and instruction work,
not just the final stores. Keep all three ablations and complete-API timing.
No implementation/default/routing change is made by this evidence review.
