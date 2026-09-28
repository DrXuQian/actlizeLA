# Residual metadata lookahead on the measured HV control

This is an **opt-in timing experiment**, not an auto-routing change or a speed
claim. Parent: published `69e6bfb` (local `aa7aefa`, identical tree).

## Question and fixed boundary

Priority workload: B1/S2048/Hk16/Hv32/K128/V128/C64, g=-1, PPU1.0.
The preceding same-device HV capture had state88.40706us and complete
four-kernel186.15942us, versus FLA224.85000us across all seven kernels.
Those are historical context, **not this candidate's measurements**.
The overall1.5xFLA goal remains unmet.

Most sampled state synchronization was after INPUTS_READY; gate/beta global
loads also had visible waits. This motivates testing overlap, but does not
establish that metadata is the sole cause of that synchronization time.

Only gate/beta issue timing changes:

1. Fetch chunk0 metadata into private registers before the loop.
2. Publish those registers to the existing shared arrays at chunk entry.
3. After current Vnew publication, fetch next chunk's metadata into registers,
   before current UPDATE. Skip this fetch on the last chunk.
4. The unchanged RETIRE barrier protects shared reuse. Do not prefetch into
   shared, reuse the current arrays early, or add another shared buffer.

Beta remains raw BF16 in the lookahead register until its original conversion
at publication. Prefix is FP32. `MetadataPlan` uses element offsets, **not byte
offsets**. Padded prefix rows still load; padded beta rows do not.

K/P/V/H layouts, matrix delivery, arithmetic order, rounding, grid128,
256threads,45568B state shared storage, all barriers and solve stay unchanged.
The candidate calls the **existing HV output kernel**, not a cloned or new
output implementation. No K-layout or solve optimization is combined here.

## Local evidence

`l031_wy_residual_metadata` exercises the production constexpr address plan
against independent canonical indexes for every sequence1..2048, both batches,
and all heads in GVA1:1/1:2/2:4:

| Quantity | Exact denominator |
|---|---:|
| Batch/head/sequence contexts | 28,672 |
| Chunks | 473,088 |
| Prefix load/publication rows | 30,277,632 |
| Valid beta load/publication rows | 29,374,464 |
| Inactive metadata-thread visits | 90,832,896 |

This is an address/lifecycle proof, not a host simulation of GDN numerics.
It covers prologue, every tail, single/last chunk,64-bit addresses and an
eight-warp partial-order publication/retirement proof. Eight planted defects
must fail, including missing coverage context and early shared overwrite.

Source binding removes **only the three registered metadata moves** and
compares the remaining state body to HV. It also binds variant10 to its exact
launcher, keeps variant9 on HV and requires the old HV output call. Nine
source/binding negative controls reject stale chunks, missing prologue,
lost retirement, wrong beta address, changed rounding and stale dispatch.

SDK2.1.1, native PPU1.0 compilation:

| State property | HV control | Metadata candidate |
|---|---:|---:|
| Vector registers | 122 | 120 |
| Stack bytes | 0 | 0 |
| Static native instruction sites | 1,194 | 1,309 |
| BF16 MMA sites | 20 | 20 |
| AIU copy sites | 4 | 4 |
| SWZL load sites, normal/transposed | 28/8 | 28/8 |
| Recurring CTA barriers | 4 | 4 |
| State shared bytes | 45,568 | 45,568 |

The extra static control/address footprint is an explicit cost. Fewer registers
do not by themselves prove greater occupancy or lower latency.

Native CFG, not source order: both recurring metadata load sites precede
**eight UPDATE MMA instructions and zero KH/PR MMAs** before their next
vector-load completion wait. The compiler places those load blocks at higher
linear PCs and branches back to UPDATE; reading disassembly linearly would
misidentify this as a late load. The syntactic final-chunk exit also follows
all eight UPDATEs; the production guard/host proof, not a guessed native
predicate equivalence, establishes that no next-chunk load occurs there.

Recurring useful arithmetic, ordinary shared/global accesses, AIU copies and
barriers have identical opcode inventories. Whole-body work also matches,
apart from two extra static load sites for the prologue. Total dynamic metadata
loads are unchanged: one prologue plus chunks-1 lookaheads. Whole-body checking
is necessary: a loop-only check initially missed a planted deletion of final-H
publication, so that checker gap was closed before handoff.

Seven native negatives include the actual old late-load schedule and two
serialized-load plants preserving **both the complete instruction/operand
multiset and the CFG**. All must fail. This proves an emitted overlap window,
not that eight MMA instructions completely hide global-memory latency.

Local evidence directory:
`/workspace/gdn-wy-residual-metadata-evidence-20260925`.
Verification entrypoint there: `verify_local.sh`; final log: `final-local.log`.
The linked-library gate also requires all34 previous kernel instruction/operand
sequences to be identical to the same-SDK parent build; there is only one new
device image (35 total).

Full local suite PASS:22/22CTest,93interface/capture contracts,7compiler-dialect
tests,45WY+61residual algebra cases,305original control expressions and35WY+
15original linked device images. The pre-existing C32 experimental original
kernel still has144Bstack and is not performance-admitted; zero-stack claims
above apply to the metadata/HV state pair, not every archived experiment.

Local DSO SHA256:
`395da41db7c84a4a22bb6a3f96bb6616972967d7aa7a99e3a5d55b8f2712bb38`.
Python binding SHA256:
`e04c4785ad6c07cfd42c8edbd35467a802cf6bee818f3a9082a79dd8443727dc`.
These are local compile evidence, not a claim that the box uses those binaries.
The runner rebuilds and records its actual source/compiler/binary identities.

## Box handoff and fixed interpretation

From the updated GDN-QSA-sm80 `ppu-backend` checkout:

```bash
DEVICE=0 JOBS=16 CANDIDATE=residual-warps8-metadata \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Set `PPU_SDK` only if the SDK is not at `/usr/local/PPU_SDK`. `OUT` defaults
to a fresh `/workspace` directory. The runner uses
`/sim/eec/shared/junfu.qx/asight/bin/acu`, performs numerical admission before
profiling, and prints one tar.gz to upload. No PPUProfiler or pasted CSV.

Admission:30scalar+30HV+30candidate cases, eight repeats, exact bytes versus
scalar residual and the independent2% recurrence gate, including tails,
nonzero state, variable metadata, GVA and output-only. Device admission is
**NOT_RUN locally**. Capture control/candidate/FLA sequentially on the same
device and fixture; compare all4/4/7 kernel durations, not public-API event spans.
Expected subject kernels are prefix, split-solve, new metadata state and old
HV output. A stale state entry is not a valid experiment.

If state and full-call latency improve with unchanged numerical results and
traffic, retain the candidate for confirmation. If waits fall but full-call
time does not, it is **no observed speed gain**, not a win. A slowdown also
remains visible: added control/address instructions can outweigh overlap.
Inspect input-ready samples, metadata waits, register/occupancy and unchanged
MMA/AIU traffic, but do not turn one counter into a causal latency attribution.
No default promotion from a single capture. Solve static indexing remains the
next separate experiment after this one is measured.
