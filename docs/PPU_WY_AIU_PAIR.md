# Matched AIU/SWZL state and output delivery

Scope: forward WY only. Parent2698276; incumbent `prepare-rows-shared`
(mask1520), B1/S2048/Hk16/Hv32/K=V128/C64, both g=-0.1 and-1.0.
This implements the native delivery candidate, **not** the complete planned
five-stage FLA rewrite. No default route or arithmetic criterion is changed.

## What changes

`include/gdn_qsa/ppu/wy_aiu.cuh::Tile` owns both endpoints: full-height AIU
SWZL cubes, up to64 BF16 columns (128 bytes), and matching SWZL matrix reads.
One issuing thread copies a whole cube; commit/wait plus the existing CTA
barrier publish it. Head-interleaved row stride is in BF16 elements; hardware
padz stops at the exact valid tail height. Strides outside the descriptor's
32-bit range fail explicitly before allocation/launch.

The same layout defines shared offsets for register-produced snapshots,
conditioned values, causal attention and output exchange. Those values do
not acquire fake AIU global loads. Every aligned8-BF16 publication remains
a16-byte vector store. Cross-slice rotation is tested, not assumed XOR.

`csrc/gdn_chunk/gdn_wy_aiu_ppu.cu` contains separate state/output bodies.
Prepare remains the admitted shared-row kernel. State stays FP32 in
registers, recurrence and MMA reduction order stay unchanged, and output
retains its two-panel arithmetic schedule. This intentionally isolates the
delivery change; joint QH/QK reuse and vector W/U prepare remain future work.
Old bodies remain verbatim because refactoring them can change native code.

| Delivery | Mask | Prepare | State | Output |
|---|---:|---|---|---|
| prepare-rows-shared |1520| incumbent | incumbent | incumbent |
| aiu-state |5616| incumbent | native pair | incumbent |
| aiu-output |9712| incumbent | incumbent | native pair |
| aiu-state-output |13808| incumbent | native pair | native pair |

## Local evidence (not a device speed claim)

Real SDK2.1.1 hgcc, ppu0010 target. The same-SDK parent has16 images; the
candidate has18. All16 old native instruction/operand sequences are identical.
`l016_wy_aiu_pair` uses the actual actlize SWZL simulator and native MMA traits:
8 tile geometries,28,416 fragment values,17,633,280 descriptor/tail elements
(including64-head materialized GVA expansion).
Six negatives reject wrong swizzle, old16x16 placement, omitted cube,
byte/element stride mix-up, tail overread, and ignored dispatch option.
Full local closure:9/9 CTests,58 Python contracts,7 compiler-dialect contracts,
45 algebra cases and305 preserved upstream source controls. Both original
and WY Python3.12/Torch2.9 bindings compile and link against the actual SDK
device libraries. This is not a PPU runtime import or device-numerical verdict.

| Static native property | State incumbent | State AIU | Output incumbent | Output AIU |
|---|---:|---:|---:|---:|
| Body instructions |2300|1875|1725|1442|
| v.mov.v2s sites |52|34|68|52|
| v.madl.i32 sites |297|112|212|34|
| BF16 MMA sites |32|32|40|40|
| Registers/thread |242|234|98|94|
| Stack bytes |0|0|0|0|

These are static sites, not dynamic ACU counts or timing predictions. v2s
does not vanish: native descriptors/coordinates still need scalar operands.
Global/shared bytes and mathematical work have not been reduced. The actual
AIU writer compiles to `vmem.aiu.ld.tsm.l0...b16.kp1`; the same-source linear
writer control produces `.l1`, and the binary gate rejects that mismatch.
The source control is `dev/ppu/l017_aiu_writer_compile.cu`.
Normal and transposed matching SWZL readers are required; any NCOM consumer
in these two bodies is rejected. A negative that replaces only one read must
fail even when many correct reads remain.

## Box handoff

From the updated repository, use the existing runner:

```bash
AIU_AB=1 DEVICE=0 PPU_SDK=/usr/local/PPU_SDK JOBS=16 \
  bash tools/run_ppu_wy_fla_box.sh
```

The runner gates the unchanged16 numerical cases, including tails, GVA,
nonzero initial state and output-only calls. All four family entries must
match scalar WY raw bits over8 repeated calls, in addition to the original
independent2% oracle. Only then it times original/scalar/incumbent/AIU-state/
AIU-output/AIU-both/FLA on the same inputs with14 balanced samples per arm.
Both decay cases run. It records masks, fingerprints, raw samples, binary
hashes and source under its printed `/workspace/gdn-wy-fla-...` directory.

Compare each candidate to the contemporaneous incumbent, not only scalar
WY. Preserve losses and overlapping envelopes as UNRESOLVED. A terminal
PASS means numerics and measurement completed, not an admitted speedup.

### Initial user report: 2026-09-24 UTC (superseded by verified upload)

The [three-arm ACU and full-sample verification](PPU_WY_AIU_ACU_20260924.md)
now closes binary identity, both-gate API envelope wins and real state/output
kernel improvement. The initial report below is retained with its original
evidence limits, not the current final verdict.

User reports correctness PASS following the8340fc7 box handoff. Record this
as **USER-REPORTED/PASS**; detailed per-arm logs and loaded binary identity
have not yet been received, so no specific case/repeat counts are claimed
as verified results. No device code was run locally.

The subsequent user table reports these complete-public-API event medians:

| Role | Median (us) |
|---|---:|
| wy-aiu-state-output | 318.564 |
| wy-aiu-state | 332.100 |
| wy-aiu-output | 339.696 |
| wy-prepare-rows-shared | 354.364 |
| original | 422.000 |
| fla | 492.188 |
| wy (scalar) | 716.770 |

The supplied labels spell `ailu`; this table normalizes that apparent typo
to the source's `aiu` labels, not a hash-verified runtime identity. The pasted
table does not include gate value, shape/device receipt, source/binary hash,
or sample envelopes. Do not assign these medians to both decay cases.

Descriptively, both reduces latency by35.800 us /10.103% against shared,
24.511% against original, and35.276% against FLA (1.545x median speedup).
These are **USER-REPORTED/DESCRIPTIVE_MEDIANS_ONLY**, not an observed-envelope
speed admission and not per-kernel ACU timings. Routing remains unchanged.

`[WY verdict] subject=wy control=fla verdict=FLA-WINS` compares the old scalar
716.770 us arm with FLA492.188 us; it says nothing about the AIU candidate.
The runner also emits `[WY delivery verdict]` for each candidate against
FLA, original, the incumbent and other candidates. It does not have a single
all-candidate winner verdict. Raw samples are required to check those verdicts.

Next capture incumbent/AIU-both/FLA under matched ACU settings and compare
prepare/state/output durations and dynamic instruction/traffic counts. Keep
full-call timing as a separate integration metric: subtracting ACU replay
sums from these API medians does not measure dispatch overhead.

The [three-arm ACU command](PPU_GDN_ACU.md) reuses the completed AIU comparison
binary and captures the two WY deliveries plus one FLA reference in a single
archive. No rebuild or new API timing sweep is required.
