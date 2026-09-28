# Exact FlashInfer pipeline alignment, before another speed claim

Control: S24 `bc3c154` / `06b7471`, H800 binary `43d62a8...`. Reference:
FlashInfer `5d9f8c8d97fa53e22952ce8672f475d235f07478`, the actual loaded
no-CP SM90 cubin, not its slower auto-CP dispatch. Same B1/T2048/Hqk16/Hv32,
D128/C64. No configuration sweep has been completed: S1–S33 are implementation
experiments on one geometry, not an enumerated optimum.

## Source and native difference ledger

Pinned reference paths below are relative to
`flashinfer/gdn_kernels/delta_rule_dsl/`. C++ paths are relative to
`csrc/backends/sm90/`. The source proves scheduling dependencies; static SASS
counts do not prove their elapsed cost.

| Axis | S24 C++ | Fastest measured FlashInfer | Evidence / disposition |
|---|---|---|---|
| Work ownership | One CTA per B×Hv;512threads, loader+aux+2state WGs | Same | `delta_rule_sm90.py:51–60`; kernel launch/types match |
| Registers per role | 24/104/192/192 | Same | Actual USETMAXREG; lowering state to184 serialized WGMMA in S30, rejected |
| Chunk / operand / accumulator | C64,D128; BF16 products, FP32 accumulator/state | Same | Actual HMMA/HGMMA types, independent CPU O/state gates |
| Inverse | FP32 8×8 diagonal, FP16 blocked8→16→32→64 | Same | `collective_inverse_hmma.py`; no different mathematical algorithm established |
| Q/K/V/O stages | **2/2/1/1** | **2/3/2/2** | C++ mainloop traits versus `delta_rule_sm90.py:164–173`; not previously aligned or swept |
| alpha/beta stages | **2/2** | **5/5** | Same source comparison; changing issue order alone did not test this topology |
| Loader issue order | Q,K,V | K,Q,V | `load_qkv_tma:486–589`; S17 alone unresolved on older shallow stages |
| Auxiliary publication | Acquire KK+QK; store both; inverse; commit both | Inverse+commit KK, then acquire/store/commit QK | `run_aux_loop_body:1519–1607`; S20 alone lost on older shallow stages |
| Scalar alpha consumers | aux128+state256+own producer32=416 | aux128+state256=384 | Reference `kernel:2149`; S33 removal alone unresolved |
| Beta consumers | aux128+state256=384, although scalar state no longer reads beta | aux128 only | Reference `kernel:2154`; part of old S13, not a new hypothesis |
| State matrix schedule | O1→SK→NewV→O2→KV, ordered WGs, commit/wait0 each | Same | `compute_loop_body:862–1008`; do not call FlashInfer wait-free |
| First-chunk specialization | Separate single-chunk and multi-chunk first bodies;4state clones | One first body with B=min(T,64);3state clones | `run_state_math_role:1257–1378`; whole no-initial kernel112vs100HGMMA sites includes an unexecuted clone, not12extra dynamic MMAs |
| NewV FP32→BF16 operand | Scalar conversion during retile | Vector conversion before operand use | `helpers.py:400`; actual no-initial body128 scalar F2F sites vs0. Old S9 removed the sites but was unresolved on~180us S7; not proof it cannot matter after S24 |
| Output retirement | TMA commit then wait0 before slot release | Same | Both `collective_store_tma`; an O2-stage ring still allows producer lookahead, not unsafe reuse |
| Gate input/math | Natural-log g, standard exp2f; cached relative factor | alpha multiplier, log2(alpha+1e-10), fastmath exp2; relative factor in state | Deliberately retained numerical contract; direct port is not raw-bit equivalent. Adapter excluded and separately measured as already registered |
| Shared storage not consumed | Original vector-KDA scaled-Q/K and alpha-last allocations retained | No such duplicate scalar allocation | Capacity cost, not repeated traffic; do not claim byte traffic without accesses |
| Tail ABI | Fixed-batch guarded tail; four dtype/initial bodies | Varlen/index/checkpoint capable | Not the full-chunk target's arithmetic bottleneck; preserve admitted tail behavior |

The stage-depth mismatch should have been checked before repeated local inverse
edits. S31 lowered diagonal selectors but lost; S32 removed16 full-chunk sites
but was unresolved; S33 duplicated part of the earlier S13 direction in a new
S24 context and was unresolved. None establishes the remaining~8us cause.

## Next modification: one coherent reference-delivery profile (S34)

User explicitly requested implementation after closing this ledger. Match
Q/K/V/O=2/3/2/2, alpha/beta=5/5, K→Q→V loading, KK-ready-before-QK-acquire,
scalar alpha384 / beta128 consumers. This is a **coupled reference profile**,
not attribution of any speedup to one knob and not a configuration sweep.
Keep S24 arithmetic, prefix/exp2 semantics, FP16/BF16 boundaries, inverse,
state matrix order, register allocation, and output-store retirement. Keep
unused storage in this first comparison to avoid mixing allocation reclamation.

Add explicit O/alpha/beta stage options alongside existing Q/K/V options so
future enumeration instantiates actual types. Print actual stage counts,
shared bytes and participant counts from those types; compilation alone is not
proof of legal resource occupancy. Native matrix/math/completion work, all14
CPU/parent fingerprints, overflow byte checks,8repeats/everycapturedoutput and
complete-forward nsys remain required. Same immutable S24 and fastest library
paths are controls. No SM80/default-route/native-PPU performance change.

After this comparison, a bounded configuration sweep should enumerate only
implemented stage axes, report the entire legal/rejected denominator and price
shared/register occupancy together. Chunk size and state-WG count require new
layout/math admission first; they are not currently legal sweep axes.

13:49 result: S34 is raw-admitted but **loses**128.8165[127.425,129.601]us
versus S24120.704[120.129,121.761]; fastestFI113.089us. Deeper rings and
reference publication/order together are not a speed admission. No defaults
change. Whole no-initial footprint6656->6560, but spill bytes20->48; do not
ascribe the loss to any one part without an ablation. This complete profile
is retained for the future config sweep, not called the optimum.

The remaining state-loop clone and conversion differences above are native
code-generation differences in **writing the same computation**, not a new
GDN algorithm. The next bounded source-alignment arm S35 tests these together
on S24, without adopting losing S34's delivery profile. It must preserve each
chunk's valid rows (first min(T,64), final tail), RNE and exact parent output.

## Measured closure, 2026-09-26 14:05 UTC

Neither alignment profile establishes a new winner. Keep the admitted S24
binary immutable. These are complete-forward **nsys kernel sums**, not API
wall time, not estimates from static instructions. Each row has12interleaved
calls per role under the same idle-device/watch contract as prior experiments.

| Profile, g=-0.1 | Candidate median [range],us | Paired S24,us | Fastest FI no-CP,us | Decision |
|---|---:|---:|---:|---|
|S34 reference stages and publication|128.8165 [127.425,129.601]|120.704 [120.129,121.761]|113.089|CONTROL-WINS|
|S35 reference state-body writing|119.9985 [118.847,120.863]|120.831 [119.967,121.375]|112.927|UNRESOLVED|

S35 is numerically and structurally admitted, but a0.7% median improvement
with overlapping ranges is not a speed admission. It remains6.3% slower than
the fastest FI median in this paired capture. We did not spend strong-gate or
FlashQLA confirmation on either screened-out profile. The goal of beating both
fastest libraries remains **NOT MET**. Previous QLA wins are historical here.

S35's actual BF16/no-initial body shrinks6656→5904sites, removes128scalar F2F
casts, and has100instead of112static HGMMA sites because one first-chunk clone
disappears. The actual chunk iterator covers266240visits (T1..4096, initial
state yes/no), including min(T,64) on the first body; missedfirst/unclampedfirst
plants fail. Every remaining native matrix batch retains wait0. Four negative
native plants reject old clones/casts, missing specialization, lost wait and
relaxed wait. This is **not fewer runtime GEMMs**.

Resources expose a real tradeoff: target/no-initial stack24→16B and spill
stores/loads20→12B, but initial-state stack32→80B and spills28/28→92/144B.
Do not promote the initial-state specialization from the no-initial timing.
Its correctness passed; its speed has not been admitted. S34 instead uses
224256B shared and48B spill stores/loads, versus S24's shallower stage profile.
Neither resource change by itself proves the causal cost of the timing delta.

Both profiles pass14CPU<2% O/state cases with exact S24 input/output/error
fingerprints; two direct-byte stress pairs at g=-8/-10000;8raw repeat launches
and every captured output. Host suites41(S34)/38(S35) PASS. Local and measured
H800 native streams match all4gate/initial-state specializations:

- S34 kernel1736ecb, final head154925f;27424sites,
  normalized SHA `22a3da861b62788300d426a02409d1769d8e8d923e1d52bbe6f14a94122addc9`.
- S35 kerneld5fda8b, final head9a39fce;24352sites,
  normalized SHA `c0cea0400e0fe639e4bc97206801d88aa2f6630eb95181dc502102f4ee6235ee`.

PPU CUTLASS3.6 CUDA source-check compiles both profiles; **native PPU1.7 is
SKIP because its SDK/model is unavailable**, not PASS. Actual SM80 and default
shipping dispatch remain unchanged. Source branches are
`sm90-flashinfer-pipeline-20260926` and `sm90-flashinfer-state-form-20260926`.
The [raw-derived manifest](../dev/backends/sm90_alignment_campaign_20260926.json)
re-extracts both SQLite captures (144complete forwards), without inventing a
second threshold or claiming that a smaller SASS listing must be faster.

## Configuration sweep: missing, and separate from source alignment

**No complete configuration sweep has been performed.** S1–S35 are an
implementation inventory, not35configurations or proof of an optimum. S34
adds real options for O/alpha/beta stages, but compiling one reference profile
is not enumerating the space. Its coupled ordering/participant changes must
not be mixed into a claimed stage-only sweep.

Next bounded inventory should retain S24's admitted computation, publication
order and register allocation; expose only stage options independently:
Q=2, K∈{2,3}, V∈{1,2}, O∈{1,2}, alpha∈{2,5}, beta∈{2,5}: **32prospective
tuples before admission**. This is a proposed bounded search, **NOT32built or
legal cells**. Compile the actual4types, record shared/register resources,
reject oversize or serialized-WGMMA codegen explicitly, then keep numeric
and measured/rejected counts with a denominator of32. The S24 tuple must
reproduce its native image; otherwise the sweep harness changed the control.
Measure weak and strong gates; use cheap screening first and complete-forward
nsys confirmation for finalists. Keep every existing per-shape winner.

Chunk64, head-dim128 and512threads/two state WGs are still mathematical/layout
constraints in this implementation, not implemented sweep axes. Extending
them requires a separately proved decomposition and consumer layout. Do not
emit unsupported rows or silently call a reference's one setting optimal.

## Evidence archive

`/workspace/gdn-sm90-win-20260926/remote-evidence-alignment-20260926T1410Z.tar.gz`
is retained on both the local host and the authorized H800. It includes the
S31–S35 raw builds/binaries, all five nsys captures and SQLite exports, numerical
cases/stress captures/parent controls, source snapshots and the re-extraction
programs. Initial failing build/check logs are preserved, not relabeled SKIP.
Its nested `local-alignment-evidence-20260926T1408Z.tar.gz` contains actual
native listings/identity gates, compiler proofs, CPU negatives and PPU3.6 CUDA
source-check builds. No credentials or GPU-control changes are included.

Outer archive:50,259,809bytes; SHA256
`637488522cfff5e1b8b64e47d9d89fa79dd52169afff3300e4a127be01fa14f5`.
Nested local evidence SHA256
`fb5ae28956e7b7596c61976e48db212acf480bed64f8c69fcb251ec39f389a9b`.
The two campaign manifests re-extract5captures/360complete forwards;
`benchmark.csv` matches all5raw-derived records exactly. Strong-gate and QLA
confirmation are intentionally NOT_RUN for these non-winning screenings.
