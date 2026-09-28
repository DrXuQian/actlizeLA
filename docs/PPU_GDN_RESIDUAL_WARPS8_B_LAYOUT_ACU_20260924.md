# Eight-warp B-layout ACU: fewer conflicts, same latency

## Verdict and scope

**Numerics PASS; no observed speed gain; keep the eight-warp control.**
The lower-BC variant remains an experimental building block, not a default
promotion. This single capture does not establish a statistically stable
regression either. The 1.5x FLA target remains unmet.

Captured source `0273ad3d9cad0f4b61e7a2429ac414935d3f787e`, same binary and
fixture, B1/S2048/Hk16/Hv32/K=V128/C64, V32, g=-1, zero initial state,
BF16 inputs and FP32 final state. All15 kernels ran at1.700GHz on the same
72-CU PPU-ZW810. This analysis executes no device code and changes no kernel,
runner, numerical criterion or default routing.

## Full-call accounting

Times are microseconds from one full site-ACU capture per arm, including
every child/helper kernel. They are not API event timing.

| Phase | Eight-warp control | Eight-warp B layout |
|---|---:|---:|
| Prefix | 2.39941 | 2.39647 |
| Solve | 52.63353 | 52.41235 |
| Residual state | 104.31353 | 105.14588 |
| Output | 46.01235 | 45.89882 |
| **All4 kernels** | **205.35882** | **205.85352** |

State changes by+0.83235us (+0.798%), the whole call by+0.49470us (+0.241%).
Unmodified phases contribute-0.33765us; their opcode histograms are identical
and that timing variation is not credited to this state-layout change.

FLA all7 kernels sum223.91471us: cumsum2.75765, BF16 fill4.31235,
KKT solve44.59235, W/U35.43000, FP32 fill2.41765, state89.52471,
output44.88000. The candidate is1.08774x FLA, versus a1.5x target of
149.27647us: another56.57705us would have to go. Our residual state includes
P@R and eliminates a separate W/U stage; state-only comparisons do not have
equal work. Full-call comparison remains the acceptance scope.

## BC is reduced, not eliminated

| State metric | Control | B layout |
|---|---:|---:|
| Shared read conflicts | 3,932,160 | 2,883,584 |
| Shared write conflicts | 4,046,848 | 2,998,272 |
| **Total conflicts** | **7,979,008** | **5,881,856** |
| Nontransposed matrix loads | 393,216 | 655,360 |
| Transposed matrix loads | 786,432 | 524,288 |
| All matrix loads | 1,179,648 | 1,179,648 |
| All shared load instructions | 1,708,032 | 1,708,032 |
| All shared store instructions | 1,343,488 | 1,343,488 |
| KVD-to-TSM bytes | 117,440,512 | 117,440,512 |
| Ordinary global KVD load bytes | 17,825,792 | 17,825,792 |
| Ordinary global KVD store bytes | 102,760,448 | 102,760,448 |
| BF16 MMA instructions | 655,360 | 655,360 |
| AIU instructions | 16,384 | 16,384 |
| BF16 conversions | 1,311,744 | 1,311,744 |

Both read and write BC lose exactly1,048,576 events: total-26.283%.
The intended262,144 transposed R/scaledV loads really changed; this is not
an unused/stale branch. The earlier empirical bank-event account predicts
these deltas, but remains empirical, not independent per-PC hardware proof.

H, K, Vnew, V scalar reads, gates and snapshot/final publication are unchanged.
The remaining5.882M conflicts are substantial. They do not all disappear
by changing one matrix reader: stores account for2.998M of them. Consult
[the native map analysis](PPU_GDN_RESIDUAL_BC_STATIC_20260924.md) before
attributing exact remaining events to individual PCs. Neither `trans` nor
an aggregate conflict count proves a particular instruction's latency.

## Occupancy: capacity, work supply, and issue readiness

| State resource/supply metric | Control | B layout |
|---|---:|---:|
| Grid CTAs / threads per CTA | 128 /256 | 128 /256 |
| Warps per CTA | 8 | 8 |
| Registers/thread / stack bytes | 162 /0 | 124 /0 |
| Shared bytes per CTA | 45,568 | 45,568 |
| Register-limited CTAs/CU | 3 | 4 |
| Shared-limited CTAs/CU | 5 | 5 |
| Warp-limited CTAs/CU | 8 | 8 |
| Architectural CTA limit/CU | 64 | 64 |
| Theoretical resident warps/CU | 24 | 32 |
| Theoretical occupancy | 37.50% | 50.00% |
| Actual active warps/CU | 14.12 | 14.11 |
| Actual occupancy | 22.07% | 22.04% |
| Eligible warps/WE | 0.36 | 0.34 |

Capacity is the minimum of register/shared/warp/block limits. Actual supply
also depends on the grid and task lifetimes:128/72=1.7778 CTAs/CU, or about
14.22 warps/CU averaged over the grid. Both capacities exceed that supply.
A fourth register slot therefore does not produce an extra ready block in
this workload. Reducing shared alone has the same limitation. The report's
resource-normalized waves/CU0.59->0.44 is not a useful-wave speedup.

Even resident warps need independent ready instructions. Fewer registers
can change operand prefetch distance and reuse, while more CTAs can duplicate
data or change synchronization. Consider all these axes together; neither
`registers<128` nor maximum occupancy is a performance admission rule.

## Compute dependency: a concrete new register-reuse clue

| Per-issue ratio (NOT time share) | Control | B layout |
|---|---:|---:|
| Memory dependency | 1.74 | 1.71 |
| TSM load/store dependency | 1.40 | 1.36 |
| TSM pipe busy | 0.19 | 0.06 |
| Compute dependency | 0.55 | 0.79 |
| Compute RAW | 0.40 | 0.26 |
| Compute WAR | 0.04 | 0.42 |
| Compute TFU WAR | 0.00 | 0.39 |
| Sync | 1.12 | 1.17 |
| Instruction fetch | 0.23 | 0.28 |

Native execution falls16,107,648->15,885,440 instructions (-1.380%).
`s.wait` falls2,078,720->1,777,664, but `v.mov.v2s` rises
1,153,024->1,218,560. These are counts, not savings in elapsed cycles.

The captured native body and per-PC export agree on the following pair:

```text
0x5480dd88: v.mma.f32.bf16.m16n16k16 vreg[16:23], vreg[110:113], vreg[102:105], vreg[16:23]
0x5480dd90: tsm.ld.swzl.b32x4.s0.t1.trans0 vreg[102:105], [sreg21] @sreg41
```

The next load reuses registers the preceding MMA reads. At that load the
profiler records137 compute-dependency samples across32,768 executions.
Three other verified pairs show the same input-overwrite pattern; one is
still a transposed reader. The control uses more separate operand registers
and has a different prefetch schedule. This supports a register-lifetime/WAR
hypothesis in addition to the limited supply of ready warps; it is not proof
that124 registers alone caused the timing outcome.

Per-PC compute samples increase2,586->3,476; memory samples7,801->7,947.
These samples are not elapsed cycles and do not share the aggregate ratio's
denominator. In particular the small memory-dependency ratio decrease is
not proof of saved wall time exactly canceled by compute waiting. The public
per-PC export gives compute dependency, not its TFU-WAR subreason. Do not
assign every compute sample to TFU WAR or invent a precise cancellation.

The user's interpretation—removing one wait exposes another, with too few
ready warps to hide it—is plausible. It is not the only explanation because
native operand allocation/scheduling also changed. The next test must
separate these two mechanisms rather than assuming only a hidden old stall.

## Next optimization order (design, not implemented here)

1. **Keep B placement; independently improve operand lifetimes.** Test a
   small independent-register/prefetch variant and interleave independent
   accumulator work. Prove from native CFG that the next operand no longer
   overwrites an in-flight MMA source and that useful overlap actually grows.
   Do not impose a128-register cap; account for all resource limits, spills,
   grid supply and eligible warps. Keep old8warp and current8warp-B controls.
   Same BC but lower WAR/greater readiness and lower time would distinguish
   this from a bank-layout win. The old four-warp prefetch experiment is not
   proof about this new eight-warp schedule.
2. **H/snapshot paired layout.** H contributes262,144 transposed loads and
   524,288 scalar BF16 stores under current ownership, recurring every chunk.
   This makes it a concrete remaining target, not a promise that all those
   events are conflicts. H also feeds canonical16B global publication: prove
   and account for both readers. Do not trade a cleaner MMA load for scalar
   global-store amplification or an uncounted rearrangement pass.
3. **K's two consumers.** K is already AIU.swzl plus matching ld.swzl. Both
   normal K@H and transposed Kt@scaledV use262,144 matrix loads; changing the
   shared orientation alone swaps which consumer needs trans. Search a native
   paired layout/ownership usable by both, including writer/reader proofs.
   A second materialized K view is an explicit traffic/shared-budget tradeoff,
   not a free swizzle. Preserve the global staging budget unless deliberately
   testing such a tradeoff.

Reducing H/K conflicts can shorten a critical load or release shared-pipe
pressure, so it remains worthwhile even with compute waiting. But if the
dependent MMA/register reuse or barriers remain on the critical path, the
kernel may still not get faster. Compare layout-only, lifetime-only and their
combination without changing geometry/math simultaneously; judge full-call
ACU time, not one counter. The earlier V16 experiment already demonstrated
that doubling CTA supply while duplicating K/P traffic is not a free fix.

## Evidence, integrity, and reproducibility

Upload SHA256:
`6b47378313f59748a5d34177f3f3391bddfbf3e6df2ea2f0a20cf520a0a07125`.
Verified616 files/615 hashes,140 source snapshots exactly matching captured
git SHA, clean tracked-source diff, same-binary/fixture/device receipts,
all4/4/7 kernels and15/15 native opcode sums. Omitted-PC negatives fail.
Uploaded compile gates retain6 source and4 native negative controls.
30 scalar,30 eight-warp and30 eight-warp-B-layout cases each repeat8 times
with RAW-BIT equality; independent2% recurrence/GVA/tail/output-only tests
and both gate admissions pass. Only g=-1 performance was captured.

Binding SHA256:
`e677544050e28c7dec4602416ff9eb62861f4f6c57eccec6e8c8deab98f6a7cc`.
DSO SHA256:
`0c55c2a01530d6111c85819aa8eb1e233830329c1b0eea6739f2f410b57e991f`.
Physical UUID `019ee024-8860-091c-0000-0000007aff6d`;
SDK2.1.1-a5c56e; capture uses exactly
`/sim/eec/shared/junfu.qx/asight/bin/acu`.

[Condensed machine-readable evidence](../dev/ppu/results/residual_warps8_blayout_acu_20260924.json)
retains raw metric names, kernel symbols/times, opcode histograms, per-PC
native pairs and provenance. Local replay is under
`/workspace/gdn-warps8-blayout-acu-analysis-20260924`:
`reimport.sh`, `analyze.py`, `export_stalls.py`, `import_stalls.sh`,
`summarize.py`, raw exports and the immutable archive. Only host-side report
imports ran; no uploaded Python, shell or binary was executed.
