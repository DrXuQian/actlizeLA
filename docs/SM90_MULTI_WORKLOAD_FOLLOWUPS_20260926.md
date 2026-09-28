# Expanded-case followups: keep negative results and distinct protocols

Physical H800 PCIe, unchanged scalar-GDN contract and immutable S24/S38
parents. No default routing or SM80 edits. Native PPU1.7 is still SKIP.
The expanded goal is **not met**, so the H800 remains on.

## S39/S40: output stash rejected by screening

Both add a32KiB private FP32 O1 stash; S40 also moves register budgets from
state192/aux104 to state160/aux168. All14 CPU/parent cases and2 direct-byte
stress pairs pass, all4 native bodies remain asynchronous. Neither wins:

| Candidate / workload | Parent weak / candidate | Parent strong / candidate |
|---|---:|---:|
| S39 B1/T2048 | 115.686 /126.848 | 117.088 /127.772 |
| S39 B2/T2048 | 116.546 /127.956 | 116.366 /127.772 |
| S40 B1/T2048 | 115.296 /122.992 | 117.300 /125.060 |
| S40 B2/T2048 | 115.314 /123.688 | 117.112 /125.594 |

Microseconds, paired graph screening, **not nsys admission**. Every cell
has disjoint ranges favoring the parent. Extra shared traffic does not become
free just because it removes accumulator liveness. No promotion.

## S41: eliminating loader spill is not a universal speed improvement

Loader24→32 on S38, state192/aux232 unchanged; all4 actual bodies have zero
spill with unchanged matrix/retirement counts. All numerical gates pass.
Six registered nsys captures/408 full forwards were re-extracted exactly
locally from SQLite. Same12 interleaved samples and disjoint-range rule.

| Workload / gate | S38 | S41 | Parent comparison | Fastest reference | S41 vs reference |
|---|---:|---:|---|---:|---|
| T2048 / -.1 | 99.985 | 101.872 | UNRESOLVED | FI113.169 | wins |
| T2048 / -1 | 100.145 | 101.488 | UNRESOLVED | FI112.721 | wins |
| T8192 / -.1 | 360.729 | 358.729 | S41 wins | FI354.376 | UNRESOLVED |
| T8192 / -1 | 359.354 | 355.674 | UNRESOLVED | FI352.987 | UNRESOLVED |
| Hv16 / -.1 | 94.111 | 95.071 | UNRESOLVED | QLA92.255 | loses |
| Hv16 / -1 | 93.375 | 94.063 | UNRESOLVED | QLA91.503 | loses |

The T8192 weak-gate parent envelopes are separated by only0.001us:
S38[360.025,362.232], S41[357.752,360.024]. Report the registered verdict,
but do not turn that edge into a general selector or claim a reference win.
The graph screen's tiny T2048 gain was not reproduced as an nsys parent win.
Different protocols remain separate. S38 stays the admitted primary path.

## S42: one state WG rejected before GPU

V128/one state WG and shared BF16 H passed the actual CuTe STSM→SS reader
map:16384/16384 cells, transpose/alias/missing-owner negatives red. It does
not pass native execution-structure admission: all4 bodies emit C7512,
stack920/1016B, spill stores2580/2852B and loads2630/3310B (without/with
initial state). Actual state248/aux232/loader24 options are emitted.

Removing the converted H operand did not remove H128+O64+SK64 FP32 live
accumulators: already256 registers before control. No timing was run.
Retain this failed branch, do not describe its generic compile PASS as a
valid asynchronous candidate. S43 retains two state WGs with state160/aux168,
but all4 bodies still emit C7512 (stack192/248B); rejected before GPU.
The last registered resource point S44 state176/aux136 removes C7512, yet
stack120/96B and spill128/272B stores,192/300B loads violate its predeclared
parent-spill ceiling. No GPU timing or measured-slowdown claim for S42–S44.
All actual maps pass16384cells and3 negatives; map correctness is not
resource admission. The shared-H direction is closed under those gates.

## S45/S46: late O1 also fails to establish an improvement

S45 moves the unchanged O1 block after NewV, before O2, preserving every
dot-product, conversion and retirement. Source-bound movement plus three
removed-wait/release/unchanged-body negatives pass; the changed QKV/QKKK
subgraph terminates at every reachable state for1..8 chunks (not a CUDA
memory-order proof). All4 native matrix/TMA/retirement counts match S24.
Stack96/112B and spills increase. S46's state168/aux152 redistribution emits
C7512 in all4 bodies and is rejected before GPU.

S45 passes14 parent-fingerprint CPU cases,2 direct-byte overflow stress
pairs and the screen's8-repeat/captured-output checks. Paired graph results:

| Workload / gate | S24 | S45 | Verdict |
|---|---:|---:|---|
| B1/T2048 / -.1 | 116.564 | 127.166 | parent wins |
| B1/T2048 / -1 | 116.888 | 127.182 | parent wins |
| B2/T2048 / -.1 | 117.302 | 126.818 | parent wins |
| B2/T2048 / -1 | 116.982 | 126.802 | parent wins |

All four envelopes are disjoint. These are screens, not nsys/reference
admission. No promotion. First numeric-admission wrapper attempt retained
as FAIL: it used a same-process monitor around subprocess cases. Fresh r2
validates actual PID ancestry/NSpid; unrelated, vanished and recycled PIDs
remain rejected. Timing's same-process monitor was not changed.

Next direction is [chunk-parallel auxiliary preparation](SM90_AUX_PRECOMPUTE_PLAN_20260926.md),
not another blind register point; design only at this checkpoint.

## Archived evidence

Frozen56-attempt matrix archive is local and remote, SHA256
`7940961bd67962ca6cca1db3c0b9f27551cc397e82d3838fa0a7c05f14835792`:
`/workspace/gdn-sm90-multishape-evidence-20260926T1712Z.tar.gz`.
Contains55 valid captures plus3 superseded captures, original failure
receipts, exact reference images, numerical diagnosis, binaries and sources.
The failed reference remains failed; it is not silently removed from coverage.

Followup evidence root `/workspace/gdn-sm90-multishape-20260926/`:
`followup-admission-r2`, `followup-screen`, `s41-nsys` (all locally copied).
Their binary/raw-report archive is local and remote:
`/workspace/gdn-sm90-followup-evidence-20260926T1753Z-r2.tar.gz`, SHA256
`dad91434fd02f52209c22f0e8c92b82f8ae2ec63ea02ef9b3c7e97d401f120c5`.
The earlier non-r2 tar is explicitly INCOMPLETE (missing source-bundle name),
not a complete receipt. The final stash gate source is additionally preserved
in `final-native-check.bundle` at81dac54. S41's34 reference image hashes
are locally verified. The late-O1 root retains numeric r1 failure/r2 PASS
and four-cell screen separately from these earlier archives.
S42 `/workspace/gdn-sm90-shared-h-20260926/`, source branch
`sm90-shared-h-20260926` commit`bbddadd`. No performance data for S42.
