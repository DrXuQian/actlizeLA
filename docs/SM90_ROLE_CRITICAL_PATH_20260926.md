# SM90 role timelines: locate the inverse chain, retain S24

**No new speed winner. Keep S24 (`bc3c154`); beating both libraries remains
NOT MET.** D28 is an isolated diagnostic, S29 is unresolved, and S30 fails
its native asynchronous-protocol requirement before device timing. No default
route, actual SM80 backend, numerical tolerance or production binary changes.

The unchanged target is B1/T2048/Hqk16/Hv32/K128/V128/C64, BF16 Q/K/V/beta,
natural-log scalar gates, FP32 final state, no QK normalization. Same exclusive
H800 PCIe/114 SMs, CUDA12.8.93, pinned FlashInfer
`5d9f8c8d97fa53e22952ce8672f475d235f07478`. All performance numbers below are
**nsys sums of every GPU kernel per complete forward**, twelve interleaved
calls per role, unchanged disjoint-observed-range rule. No foreign workloads,
clock/power changes or counter-permission workarounds. H800 is not PPU1.7.

## Device verdict

| Experiment | Subject median [min,max], us | Paired S24, us | Fastest FI no-CP, us | Use / verdict |
|---|---:|---:|---:|---|
| D28 probe, g=-0.1 | 130.6395 [129.119,132.000] | 120.7675 | 112.8315 | Diagnostic overhead +8.174%; NOT a speed candidate |
| D28 probe, g=-1 | 129.872 [128.320,130.752] | 120.208 | 112.065 | Diagnostic overhead +8.039%; NOT a speed candidate |
| S29 warp coordinates, g=-0.1 | 122.0485 [120.608,122.881] | 120.464 [119.360,122.208] | 112.720 | UNRESOLVED versus S24; loses to fastest FI |
| S30 register redistribution | NOT RUN | — | — | Native protocol gate FAIL; rejected |

Do not promote S29 from its lower static instruction count. Strong-gate and
FlashQLA speed confirmations were not run for an unresolved screening result.
The prior S24 wins over fastest FlashQLA auto (1.36–1.39x) remain historical,
not fresh measurements here; see [the S24 report](SM90_AUXILIARY_SASS_20260926.md).
The current fastest-FI gap is still roughly7%, not closed by choosing its
slower automatic dispatch as the comparison.

## D28: what the role timestamps establish

An opt-in `GDN_SM90_ROLE_TRACE` build records `%globaltimer` from one leader
of the existing auxiliary WG and each of the two state WGs. No added barrier,
atomic, modified arithmetic, operand layout, pipeline count or production
argument. Reset/readback occur outside the measured forward. Storage is a
bounded diagnostic-only device symbol:64 CTAs ×64 chunks ×3 roles ×16 slots.
Normal builds do not expose the debug methods. `%globaltimer` is target-specific;
its H800 nanosecond semantics are not a PPU clock contract. See the
[official PTX timer specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer).

Eight independent calls per gate produce **35,840 valid stamps per call**:
32 CTAs ×32 chunks ×(9 auxiliary +13 state0 +13 state1). Every expected slot
is present, all unused slots are zero, and all per-role/chunk timestamps are
ordered. Output and final-state bytes match uninstrumented S24 on every call.
The two raw timestamp files reproduce all16 analysis records exactly.

The registered perturbation ceiling was20%; actual overhead is about8%.
That admits directional investigation, **not** exact uninstrumented costs.
Probe spills grow: no-initial-state stack24->72B, initial-state32->48B;
the observer changes scheduling/liveness despite preserving data protocols.

Below, each entry is the median across8 calls of the per-CTA median of an
interval summed across its32 chunks. Intervals in different roles overlap;
they must not be added together or subtracted from the separate nsys runs.

| Instrumented inclusive interval | g=-0.1, us | g=-1, us |
|---|---:|---:|
| Auxiliary inverse + its synchronization | 58.304 | 58.296 |
| Auxiliary scalar epilogue + output-slot acquire | 25.680 | 25.776 |
| Auxiliary QK issue + both matrix products retiring | 9.824 | 9.792 |
| Auxiliary final cast/commit/release | 7.432 | 7.408 |
| State WG0 inverse/beta wait | 17.048 | 16.968 |
| State WG1 inverse/beta wait | 13.112 | 13.016 |

Steady-state chunk cadence is about3.712us in the instrumented image. This
makes the auxiliary inverse/publication chain a concrete target, and shows
state waiting on it. **It does not prove that58us is excess over FlashInfer,**
that every inverse instruction is on the uninstrumented critical path, or
that deleting17us of waits would yield17us of speedup. These waits protect
actual data; they cannot simply be removed.

Native validation examines all four real gate-dtype/initial-state bodies:
70 CS2R timer sites per body (two auxiliary full/tail clones ×9, four state
clones ×13); HGMMA/HMMA/LDSM/STSM/BAR/SYNCS/WARPGROUP counts stay unchanged.
For the BF16/no-initial body, auxiliary timer0 atPC1700 precedes the K wait,
timer1 at17c0 follows it; timer2 at1b20 follows Q readiness; timer3 at1e50
follows WARPGROUP.DEPBAR at1e40. Inverse endpoints at3c30/5440 follow the
pre-inverse/post-inverse named barriers at3c20/5430. Register-only operations
can still be scheduled across timer sites: do not assign their isolated
latency from adjacent stamps. Disabled instrumentation is checked against
all four complete S24 instruction streams, including operands/order.

The first diagnostic build (`24563df`) incorrectly combined an extern device
declaration with a separately defined symbol and failed compilation. That
was an implementation error, not an environment SKIP. Only corrected
`bd6f6a6` / `trace-build-r2` is measured; failed artifacts are retained.

## S29: a real simplification, not a speed improvement

Only the FP16 inverse32->64 helper's WG-local warp ID changes:

`canonical_warp_idx() - 4*canonical_warp_group_idx()`
becomes `(unsigned(threadIdx.x) >> 5) & 3`.

This is in the **SM90-owned vendored** `cula/kerutils/device/sm80/` instruction
helper, not our actual SM80 backend. TF32 and all layouts/math remain unchanged.
The compiled helper is exhaustively checked for all1024 legal CUDA thread
positions, independently using warp/group quotient arithmetic; a wrong-bit
shift causes768 mismatches. Owners are256/256/256/256.

Native full/tail inverse intervals lose one SHFL each (22->21); matrix,
conversion and barrier work is unchanged in all four specializations.
BF16/no-initial full interval358->350 static sites, tail329->323. However,
the full interval gains a local-memory load (2->3), and whole-kernel spill
loads/stores20/28->24/36B, stack24/32->24/40B. The total four-body footprint
27856->27816 sites does not translate to a full-forward win. Keep the negative
result rather than merging an aesthetically simpler expression as a speed fix.

## S30: unchanged total register pool still serializes WGMMA

The independent experiment reallocates registers per thread:

| Role | S24 | S30 |
|---|---:|---:|
| Loader | 24 | 24 |
| Auxiliary | 104 | 120 |
| Each state WG | 192 | 184 |

Both use `(loader + auxiliary +2*state)*128 = 65,536` registers. No extra CTA,
changed math or changed state source. Nevertheless, ptxas emits **C7512 in all
four bodies: WGMMA serialized for insufficient register resources**.

| Native static work per body | No initial state, S24 -> S30 | Initial state, S24 -> S30 |
|---|---:|---:|
| HGMMA | 112 ->112 | 144 ->144 |
| WARPGROUP.ARRIVE | 20 ->112 | 24 ->144 |
| WARPGROUP.DEPBAR.LE | 18 ->112 | 22 ->144 |

Exact allocation operands do reach SASS (24/120/184); the experiment is not
an ignored knob. The compiler inserts a completion boundary after essentially
each matrix operation, defeating the async-path premise. The pre-registered
native protocol gate stays **FAIL**, not converted to PASS/SKIP. Device numerics
and timing are NOT_RUN for this rejected candidate.192 is the current image's
known working allocation, not an architectural minimum for every kernel.

## Validation and evidence

D28 and S29 each pass all14 independent CPU output/state cases under the
unchanged<2% gate, with identical parent errors and input/output fingerprints.
S29 also passes two separate g=-8/-10000 direct O/state byte comparisons;
these do not broaden the ordinary timing harness's input range. All admitted
timing roles pass8 repeats and every captured output. Host suites41/38 PASS.
Timestamp negatives reject a missing stamp, omitted CTA denominator,
nonmonotonic stamp and unowned slot. Native negatives reject the old image,
missing timer, missing data barrier/matrix work, missing specialization,
changed operands and reordered instructions.

Actual remote/local native identity matches for all three images:

| Image | Four-body static sites | Normalized native SHA256 |
|---|---:|---|
| D28 | 28,576 | `fd3f6fe9dfa83d1cf29cec2dd9cbf1acbdc599a72b503707dd9661d51b0529db` |
| S29 | 27,816 | `338379a6603597c92f334c64809360f902968bb0c79c6369a5640c1b62c70d22` |
| S30 | 28,584 | `f322894151a52bf9ef2bdd94fd86d41983b23749e8419f85a2875ffc62ef2eb9` |

S29/S30 compile all four real bodies against PPU CUTLASS3.6 in CUDA
source-check mode. That is a compile result, not a reversal of S30's native
protocol rejection. **Native PPU1.7 SDK/model validation remains SKIP.**

Sources are pushed separately: D28 `sm90-role-trace-20260926` (kernel
`bd6f6a6`, tests `6c05068`), S29 `sm90-inverse-warp-coords-20260926` (kernel
`c342ff9`, tests `adc1e3a`), S30 `sm90-aux-registers-20260926` (kernel
`effb618`, tests `f9163e4`). The
[manifest](../dev/backends/sm90_role_campaign_20260926.json) binds source and
binary identities, three captures/216 complete forwards, both timestamp files,
cases/stresses and exact SQLite/raw-timestamp reanalysis. Diagnostic overhead
rows are explicitly separated from speed rankings in the experiment CSV.

Raw evidence is retained locally and remotely at
`/workspace/gdn-sm90-win-20260926/remote-evidence-roles-20260926T1246Z.tar.gz`:
24,674,763 bytes, SHA256
`10a8924f342fcf79823044069564187ca383b045c50525e04bc4580d20b268fa`.
It includes the failed first compile, three measured/native-rejected builds,
source bundle, raw traces, cases/stresses, native gates, host tests and
PPU-fork CUDA source checks. Local/remote archive hashes match; member hashes
for all12 profiler records, all3 binaries, native inputs and both timestamp
pairs are verified. Final disabled-probe recompilation retains all27,856 S24
instruction sites with normalized SHA256
`23907fbf1ac595040d9622440188c4a59c53ffe8e8b0aaf1df8dcc0a6340e075`.

## Next bounded direction

Keep the uninstrumented S24 parent. FlashInfer's pinned
`delta_rule_dsl/collective_inverse_hmma.py` uses the same8->16->32->64 blocked
inverse structure; a different algorithm is not established as the missing
advantage. Next compare the **matched inverse phases' native operand lifetimes
and dependency chains**, with the8x8 diagonal elimination and32->64 HMMA/FP16
reduction boundaries explicit. Preserve the admitted rounding and shared
producer/consumer layouts. A change must reduce the actual critical path and
win complete-forward timing; fewer address sites or more auxiliary registers
alone have now failed that test. No S31 or composition was added to this turn.
