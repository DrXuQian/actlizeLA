# S47/S48: independent auxiliary preparation, complete-forward accounting

S47 is **not promoted**. Its B2 counterexample survives same-input nsys.
S48 is also rejected: it produces no new winner in the registered four-cell
screen against S24/S38/S47. Both unsuccessful candidates remain archived.
The expanded14-workload goal remains unmet; H800 remains on. This is CUDA
SM90a/H800 evidence, not native PPU1.7 performance.

## Immutable identities and arithmetic

S47 source8bd9fe4, measured binary in
`/workspace/gdn-sm90-precomputed-aux-20260926/s47-build`.
Local/remote CUDA native SASS SHA256:
`03cef08c1506ba6014a5331722ad95b0bfb96ab04801a842b06fa00d9a5c9f10`.
S24 and S38 remain the previous immutable controls; no selector/default or
SM80 change. S47 reuses byte-identical scalar auxiliary/state/gate arithmetic
headers from c5c0584. BF16 dots, FP16 triangular inverse, BF16 conditioned
operands, FP32 state and final-state output are retained.

Preparation launches one128-thread CTA per B/Hv/chunk; QK and beta-conditioned
inverse are stored as two private physical BF16 operand images. A V64 state
CTA reloads them without a second swizzle. This is not an affine state scan
or FlashInfer CP: state still recurs through chunks, and both V slices repeat
Q/K loads. The complete call includes preparation and state.

## Admission before timing

- Six actual CUDA bodies (two prepare gate types/four state gate+initial).
  Asynchronous WGMMA, zero stack/spill; prepare95 registers, state128 static
  with explicit state192/load24 role budgets. Dynamic shared50560/102144B,
  plus1024B static per kernel. CUDA resource API: prepare4/state2 CTA slots
  per SM in all four gate/initial variants, not a claim of achieved residency.
- Actual STSM cross-lane writer -> global physical image -> raw cp.async
  reload -> O2/NewV reader:8192cells exact-once, wrong-halfword and missing
  owner negatives red. Scratch-offset checks cover all14 performance shapes;
  omitting one shape fails the independent inventory denominator.
- Source-bound data-pipeline graph explores every reachable state for1..8
  chunks; missing TMA/cp.async waits and phantom producers are red. This is
  a dependency/progress proof, **not** a CUDA memory-order proof.
- Device numerical suite:14 fixed admission cases (tails1..257, GVA,batch,
  both gate dtypes, varying gates, nonzero initial state, output-only, and
  target2048), plus two extreme-gate byte stresses. Exact parent fingerprints
  and CPU oracle pass. These14 admission cases are NOT the separate14
  performance-workload matrix. Screen repeats8direct launches and checks
  every captured output; reference captures independently check all outputs.
- Harness69 CPU tests pass with CUDA hidden on the remote host. Local Python
  lacks Torch: its attempted full suite has environment import errors, not
  a claimed full PASS. The original identity-only resume gate also correctly
  rejects this changed execution-contract epoch; its immutable old repair
  remains tested, and new captures use a fresh root.
- PPU CUTLASS3.6 CUDA source-check passes. Native PPU1.7 SDK/model unavailable:
  **SKIP**, not device PASS.

Initial prepare build had1072B per-thread parameter copies/generic stores.
Using CUTLASS's existing grid-constant kernel-Params contract eliminated
them. The failed native gate remains preserved as r1; it was not relabelled.

## Four-cell graph screen (complete two-launch spans)

| Shape | Gate | S24 us | S47 us | Fixed observed-envelope verdict |
|---|---:|---:|---:|---|
| B1,T2048,Hq16,Hv32 | -0.1 |116.740|104.304|S47 wins S24|
| same | -1 |116.754|103.802|S47 wins S24|
| B2,T2048,Hq16,Hv32 | -0.1 |115.470|157.762|S24 wins|
| same | -1 |117.378|157.512|S24 wins|

Graph speed relative to S24 does not establish a primary-shape win over S38.
Because B2 loses, do not spend56captures pretending S47 is a broad finalist.
Two separately registered diagnostic nsys captures resolve the two-child cost.

## Same-input nsys diagnostic, weak gate

Medians in us. Each row has12 interleaved/reversed complete forwards per arm,
the same pinned FI auto/no-CP references, original oracle and parent raw gate.

| Workload | S47 prepare | S47 state | S47 all-kernel sum | Best old control | Fastest FI |
|---|---:|---:|---:|---:|---:|
| B1/T2048/Hv32 |26.7035|80.1105|106.654|S38 99.454|112.862|
| B2/T2048/Hv32 |51.375|110.014|161.277|S24 127.598|114.958|

Component medians need not add to the median of sums. S47 loses its old
incumbent in both rows with disjoint ranges. It beats FI at B1 but loses FI
at B2. Gap-between-child medians1.040/1.536us are listed separately, never
subtracted as preparation cost. No complete-call win or explanation follows
from resource capacity alone.

Raw reports/SQLite/receipts/binaries are retained locally and remotely under
the task root. Two captures cover144full forwards. Local Python3.12 reimports
both SQLite files with the new contract: exactly one prepare followed by one
state on the same stream, no unrecorded helper. Old single-kernel arms still
reject extra kernels. Missing prepare, missing build binding, wrong ordering/
stream or a reduced denominator fails. A constructed fast-state/slower-total
trace is correctly a loss, not an apparent speed win.

## S48 hypothesis and closed screen

Registered before edits in
`/workspace/gdn-sm90-precomputed-v128-20260926/docs/plan.md`; source164560e.
Keep the prepared kernel's actual two native instruction sequences identical
and change the state to the **existing V128/two-state-WG geometry**. Q/K
staging per head/chunk halves64->32KiB and prepared reads halve32->16KiB;
prepare writes16KiB and reads32KiB Q/K unchanged. These are logical/issued
bytes, not measured DRAM. State grid halves and resource capacity returns
to one CTA/SM: an explicit underfill tradeoff, not free sharing.

Local CUDA all six bodies pass with zero stack/spill. State168static registers,
explicit state192/state192/load24,384threads,118528Bdynamicshared. Actual
two-WG operand maps and source-bound pipeline progress pass. All14 fixed
parent-raw/CPU numerical cases and2 extreme-gate stresses pass. Actual resource
queries confirm one state CTA/SM and four prepare CTAs/SM. Local and remote
CUDA SASS are identical, SHA256
`97138b7088da794c020118c178b2bda8701e75e952aab120e4735bde85150fcb`.
PPU CUTLASS3.6 source-check passes; native PPU1.7 remains SKIP.

The registered graph screen counts both launches, checks8 direct repeats and
all64 captured outputs per cell, and alternates/reverses the four arms:

| Workload/gate | S24 us | S48 us | S47 us | S38 us |
|---|---:|---:|---:|---:|
| B1,-0.1 |115.750|114.252|104.710|91.302|
| B1,-1 |116.104|114.188|104.392|91.060|
| B2,-0.1 |117.570|164.126|157.700|191.546|
| B2,-1 |117.662|163.986|157.612|191.490|

S48 loses S47 and the best immutable incumbent in all four cells. Lower
logical duplicate traffic is real but did not produce a speed improvement.
No full reference-matrix rerun is justified for this rejected candidate.
These graph values are screening spans, not replacements for the frozen
matrix's nsys kernel sums. No default routing or SM80 change.
