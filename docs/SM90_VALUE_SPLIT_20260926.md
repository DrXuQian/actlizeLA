# V64 ownership: primary-shape win, broader admission still pending

On the **H800 PCIe**, S38 beats the fastest measured FlashInfer and FlashQLA
paths for B1/T2048/Hqk16/Hv32/K128/V128/C64, in both decay regimes. This does
**not** establish a universal selector or native PPU1.7 performance.
The user subsequently required broader cases. The new 14-workload inventory
must be measured before concluding the task; the H800 remains on.

## What changed, and what it costs

S37 partitions public V128 into two independent V64 CTAs. Each CTA has one
state warpgroup, one auxiliary warpgroup, and one loader warpgroup: 384
threads, versus the parent's 512. Each output/state coordinate still has
exactly one owner; recurrence order, K work and rounding are unchanged.
There is no partial-output reduction or new workspace.

This doubles the primary grid from 32 to 64 CTAs, but also duplicates Q/K
loads and QK/KK/inverse work. It is a **parallelism/resource tradeoff**, not
free extra occupancy. Shared storage is 152,576 B, still one CTA per SM.

S38 changes only the auxiliary budget104→232 after that ownership change.
State keeps192 registers; loader keeps24. The former two-state redistribution
experiment had reduced state192→184 and triggered ptxas C7512 serialization;
S38 does not. The four compiled bodies preserve their matrix/completion
families, and native static spill goes16→8B relative to S37. Static site
counts are not dynamic work counts: the duplicated auxiliary work remains.

## Full-forward nsys results

Twelve interleaved/reversed calls per role on an otherwise idle, identity-bound
device. All reference helpers are included. The unchanged rule requires
disjoint observed ranges; medians alone do not admit a win.

| gate | comparison capture | S38 median [range], µs | paired fastest reference, µs | result |
|---|---|---:|---:|---|
| -0.1 | FlashInfer confirmation |101.457 [99.104,102.305]|112.737 [111.457,113.441], no-CP|S38 wins|
| -1 | FlashInfer confirmation |101.329 [97.889,102.465]|112.7375 [109.697,113.153], no-CP|S38 wins|
| -0.1 | FlashQLA |99.3445 [97.856,101.185]|165.330 [163.264,166.849], auto|S38 wins|
| -1 | FlashQLA |99.4565 [97.505,100.641]|163.4415 [161.121,165.249], auto|S38 wins|

The first paired FlashInfer captures independently won101.3765 vs113.1045
and100.689 vs112.018µs. The immutable S24 parent remains approximately120µs
in the same captures. S37 alone won103.969 vs112.129µs on the weak-gate cell.
Across7 captures, all480 complete forwards re-extract exactly from SQLite.
Every candidate comparison wins against **all** measured reference paths;
the slower FlashInfer auto path is not used to inflate the conclusion.

## Correctness and code generation

- Both candidates pass all14 existing CPU O/state cases under the unchanged
  2% max/max gate, with identical parent input/output/error fingerprints.
- Two direct-byte overflow stress pairs per candidate match the parent.
  Every reference/candidate passes8 raw-stable repeats, and every captured
  result is checked. No tolerance, fast-math, fixture or output cast changed.
- Actual compiled type maps prove16,384 state /8,192 output owners exact-once,
  parent per-lane coordinates, and96 global ABI maps. Four mapping negatives
  plus three native-body/wait negatives are retained. Host suite38 PASS.
- Local and measured-device builds have identical full native SASS hashes for
  each candidate. Four bodies compile; no C7512/C7510 warning.
- PPU CUTLASS3.6 CUDA **source-check** passes. Native PPU1.7 SDK/model unavailable:
  **SKIP**, not a native correctness or speed claim.

One failed check remains visible: recompiling the experiment with the switch
off produced27,840 native sites, not S24's27,856. All14 numerical cases match,
but this is **not** native-code identity. Timing always used the immutable
old S24 image, not that recompile. No shipping/main/SM80 routing was changed.

## Evidence / source bindings

Experiment branch `sm90-value-split-20260926`: kernel `c490c8d`, native tests
`c352173`, primary recorder `2b51b2f`. The multi-workload branch's `388f212`
renames its success flag to `primary_shape_admitted` and explicitly sets
`shutdown_authorized=false`, reflecting the user's later scope expansion.

S24 measured image SHA256:
`43d62a8cabcf072b0e19b787d0c5ac9fd6aff2855ab1fcc5c96c7b73decbbb3c`.
S38 measured image SHA256:
`78993bafb1b09c2eba5e99d7c1f923dde9af94a26b2e170620a0c620102c714a`.

Machine-readable results, full samples and hashes:
`/workspace/gdn-sm90-value-split-20260926/primary-campaign.json`.
Local and remote archive:
`/workspace/gdn-sm90-value-primary-evidence-20260926T1537Z.tar.gz`
SHA256 `792bd85452be5a326fafe48486d1d03819009fdc0635f1502994a70ec1d702d4`.
It retains7 traces, all binaries/build receipts/native code, all CPU/stress
captures, tests and source bundles. A missing confirmation, losing comparator,
or mixed candidate binary makes the primary admission function reject.

## Expanded acceptance, registered before measurements

Authority: `tools/sm90_workloads.py` on `sm90-multishape-20260926`.
T={512,1024,2048,4096,8192} plus tail2051; B={1,2,4}; Hv={16,32,64};
GVA={1,2,4}; distinct FP32 gates and nonzero initial state. Fourteen workloads,
each at g=-0.1/-1 and against both libraries: **28 numerical scenarios and
56 nsys captures**. These are12 distinct shapes plus2 input/state variants,
not a full Cartesian product. All K/V remain128 (current public kernel scope).

The official Qwen35B head geometry16/32 and Qwen122B16/64 are covered; other
rows are deliberate axis probes. Both libraries retain their auto and no-CP
paths; select the faster reference independently per cell. Never drop a losing
large-grid cell or hide it with a geometric-mean headline. The old winning
shape must reproduce unchanged through the generalized harness.

Batch adapters preserve recurrence boundaries: FlashInfer packed tokens have
`cu_seqlens=[0,T,...,B*T]` and VK state; FlashQLA retains its fixed-batch API
and KV state. Actual CPU tests reject omitted boundaries, wrong GVA and a
square-state transpose, as well as a missing workload denominator. All9 CPU
tests passed on the remote CPU; the local no-Torch environment explicitly
skips4 tensor tests and passes5 metadata tests.
