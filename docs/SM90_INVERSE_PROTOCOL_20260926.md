# S31–S33: numerical closure, no new speed winner

Retain S24 (`bc3c154`, binary `43d62a8...`). The fastest FlashInfer no-CP
gap remains~7%; slower auto-CP is not the goal. H800 only, not native PPU1.7.
Fixed B1/T2048/Hqk16/Hv32/D128/C64, BF16 inputs, scalar gates, FP32 final state.
Every result is an exclusive, paired nsys sum of all forward GPU kernels,
12interleaved/reversed calls per role; no threshold changes.

| Candidate, g=-0.1 | Candidate median [range],us | Paired S24,us | Fastest FI,us | Decision |
|---|---:|---:|---:|---|
| S31 unit-diagonal in KK publication |124.3525 [121.985,125.120]|120.448 [119.520,121.664]|112.656|CONTROL-WINS|
| S32 packed-half inverse reduction |121.9365 [120.800,122.624]|120.592 [119.776,121.536]|113.0565|UNRESOLVED|
| S33 remove unused alpha-last channel |120.497 [119.457,121.281]|120.625 [119.905,121.313]|113.025|UNRESOLVED|

No strong-gate or FlashQLA speed confirmation was spent on a candidate that
did not win screening. Prior S24 wins over fastest FlashQLA remain historical,
not freshly measured here. Default routing and actual SM80 code are unchanged.

## Mechanisms and limitations

- S31 moves unit-diagonal initialization into the existing KK writer. The
  solver receives exactly its former normalized input, including padded tail
  diagonals. Actual MMA/STSM mappings cover64tails×2stages×4096cells. Five
  negatives reject diagonal garbage, upper contamination, omitted tail diagonal,
  missing owner and a wrong lane-local zip of a cross-lane STSM. Native inverse
  FSEL17/18→7, but whole-body grows80sites and spills20/28→24/32B. It loses.
- S32 recasts the actual16-half accumulator fragment into8pairs and uses
  CUTLASS pairwise RNE additions, no reassociation.33,554,432 actual device
  scalar/packed comparisons have zero raw differences, including all half bit
  patterns against32boundary/special operands. Swapped partner and omitted
  pair produce18,842,224 and2,740,094 mismatches. Native repeated full inverse
  reduction36→20/21sites,16HADD2+8PRMT→8HADD2. **The parent tail already used
  packed adds**; its21→23sites is a cost, not a saving. Stack24/32→16/24B
  still does not establish a speedup.
- S33 deletes a duplicate scalar channel that is written/waited but never
  read by state. Real alpha arrivals become384 rather than416, while the
  vector-KDA count stays416. Actual4type instantiation and55,008 reachable
  bounded ring states pass; stale416/deletedconsumer negatives reject deadlock
  and premature reuse. Native drops4LDS/4STS and6wait sites per body, preserves
  matrix/completion, and retains initialization/storage. Spills rise20/28→44/56B.
  This repeats part of old S13 in a later S24 context; it is not a newly
  discovered algorithm or a speed win.

## Validation and transparent failed checks

All3candidates pass14CPU-oracle cases with unchanged<2% output/state limits,
identical S24 input/output/error fingerprints, plus two direct-byte overflow
stress pairs each. Every measured role passes8repeats and every captured output.
Host suites38/38/39 PASS after correcting literal-source tests to their new
seams while retaining/adding removal negatives. Initial failing logs remain.

The first S31 host map zipped STSM source/destination in one lane; the actual
cross-lane copy trait rejected it, and it is retained as a negative. S32's
first native gate incorrectly expected scalarized adds in the parent's tail;
the correction above was recorded before timing and the candidate8/0 rule was
not relaxed. Initial S32 compile errors were a nonconstant parameter expression
in static_assert and an Array reference-proxy test access, not environment SKIPs.

Local and actual measured remote native streams match across all4specializations:

| Candidate | Sites | Normalized native SHA256 |
|---|---:|---|
|S31|28176|`732f6f43ee1c9e41b94436d388ce927d3d429ee91b696c04c1eb3976cac5a937`|
|S32|27792|`2dd031c67632c84f385d5e0a1e390717d9a243c369a6e2a08b4bbdc593d4f121`|
|S33|27408|`dc45dcabdb1fea991b0ce9c62b2a69129807185bbdad72851bce8206294fe9ba`|

All compile against the PPU CUTLASS3.6 dependency in CUDA source-check mode.
Native PPU1.7 SDK/model execution remains **SKIP: unavailable**.

Sources are pushed on `sm90-inverse-normalized-input-20260926` (kernel0e3cbfa,
tests51f3921), `sm90-inverse-packed-half-20260926` (kernel6ee9b52,gate675983d),
and `sm90-scalar-alpha-single-channel-20260926` (kernel0cb0091,tests2bfb74f).
The [raw-derived manifest](../dev/backends/sm90_inverse_campaign_20260926.json)
exactly re-extracts3captures/216complete forwards and binds all3binaries,
42case results,6stress pairs and host logs. It does not select a new winner.

The user's methodological correction is accepted: close the **whole reference
pipeline** differences before more micro-edits. The subsequent source/SASS
ledger and coupled reference profile are separate work in
[SM90_FLASHINFER_ALIGNMENT_20260926.md](SM90_FLASHINFER_ALIGNMENT_20260926.md).
There has not yet been a complete configuration sweep.

All S31–S33 raw evidence is sealed together with S34/S35; the archive path,
hashes and explicit no-promotion conclusions are recorded in that alignment
report's Evidence archive section. This closes the inventory, not the
remaining fastest-FlashInfer performance gap.
