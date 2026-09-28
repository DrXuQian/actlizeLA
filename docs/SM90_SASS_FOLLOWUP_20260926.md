# SM90 follow-up: producer ownership and full-chunk specialization

Continuation of [the first SASS campaign](SM90_FLASHINFER_SASS_20260926.md).
Same physical H800, CUDA12.8, B1/T2048/Hqk16/Hv32/D128/C64, scalar gates
g=-0.1/-1.0, unchanged BF16/FP32 boundaries and independent 2% O/state gate.
Numbers below are **nsys sums of every GPU kernel in one full forward**,
12 interleaved/reversed-order calls per role. They are not API/event times.
No default routing or SM80/PPU1.0 changes. Native PPU1.7 remains unmeasured.

Final incumbent: **S19, kernel64691d1**, branch`sm90-aux-full-tail-20260926`.
It wins against S16 and fastest FlashQLA in both gate regimes, but still loses
to fastest FlashInfer. **The requested two-library goal is NOT MET.**

## S16: first confirmed improvement in this continuation

S16 `e799c55`, branch `sm90-factor-owner-20260926`, combines cached identical
gate coefficients with publication by the existing alpha-last warp, leaving
the Q/K/V TMA warp free to issue copies. Relative to its cached S12 parent,
state/auxiliary arithmetic and storage are unchanged: no extra warp, no new
shared allocation, no fast-math flag. The producer publishes all four factor
channels before alpha commit, consumes/releases that same alpha stage while
publishing alpha-last, and retains all416 alpha consumers.

This is an interaction test, not a reclassification of the earlier cache-only
loss/overlap or producer-only overlap as wins. Native state coefficient work
is retained at the lower count. Delivery correctness passes14 independent
cases and parent input/output-state fingerprints/errors,8 stable repeats and
all captured outputs. Missing-channel/producer/callback/release negatives fail.

| Paired capture | S11 control | S16 | Fastest reference | Verdict |
|---|---:|---:|---:|---|
| FI, g=-0.1 |147.6165 |142.9600 |112.8005 no-CP |S16 improves; FI wins |
| FI, g=-1.0 |147.2800 |142.8805 |112.8800 no-CP |S16 improves; FI wins |
| QLA, g=-0.1 |146.3530 |140.6090 |165.3610 auto-CP |S16 wins |
| QLA, g=-1.0 |146.4320 |140.2405 |163.6005 auto-CP |S16 wins |

All stated wins/losses use disjoint observed envelopes in that SAME row.
Units:microseconds. S16 improves S11 by3.0–4.2% and beats QLA1.176x/1.167x;
it is still about1.267x as slow as FI. **The two-library target is NOT MET.**
Do not pair the best QLA-window candidate time with another window's FI time.

Measured binary SHA256:
`a88a07e8c48c53a860e5cc94a936737be508e487314acbc0d197cbbf34970b08`.
Remote binary:`/workspace/gdn-sm90-win-20260926/factor-owner-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so`.

## Rejected or unresolved experiments remain visible

- **S14** `107c0de`: immediate predicated named-barrier IDs remove the intended
  runtime-ID spill path. Other spills grow; stack64→96B. Paired148.689 versus
  147.2805us overlaps:UNRESOLVED. Actual old S11 native image fails the new
  immediate-ID gate. No protocol waits were deleted.
- **S15** `5ab062b`: an empty-assembly H operand fence intended to force
  recomputation is optimized away. All29,768 normalized native instructions
  equal S11, SHA`96c0bf23a280c0cdf48d1a12368e2546bb3e125ad21303d6674806bbaa1e9ab5`.
  Rejected before device timing; do not retain this inert asm as a fix.
- **S17** `daabe62`: K,Q,V TMA order instead of Q,K,V, independently on S11.
  Actual native destinations change from Q(base0),K(base0x8000) to K,Q in all
  four bodies; six TMA load sites and descriptor/storage binding remain.
  All14 numeric/fingerprint cases pass.147.9845[147.041,148.961] versus
  146.865[145.792,148.384]us overlaps:UNRESOLVED, no composition/promotion.

## S18: remove issue ordering, not data dependencies

Experimental kernel`642d353`, native-gate follow-up`f529083`, parentS16.
Since inverse ownership moved to the auxiliary WG, state WG0 owns V0..63
and WG1 V64..127. The actual instantiated CuTe types prove disjoint ownership
of16,384 H elements,8,192 O elements and8,192 physical shared output elements
for each of four gate/initial-state specializations. A stale WG0 writer fails.

Only the two-WG alternating issue preference is removed, and only for the
aux-owned inverse specialization. The original state-owned inverse keeps its
ordered policy. TMA stage completion/reuse, all arrival counts, O publication,
async shared fences and every WGMMA fence/commit/wait remain. Source hashes
bind the state body and pipeline sources to S16; deleting a real completion
wait or release fails. The PTX contract scopes WGMMA groups/completion to the
executing warpgroup; that is necessary background, NOT by itself proof that
arbitrary inter-WG barriers may be removed. See
[NVIDIA PTX8.7 WGMMA completion](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-wait-group).

All four native bodies retain identical HGMMA/HMMA, TMA and SYNCS/WARPGROUP
families while256-thread issue-barrier sites33/41 become0. Old native code
and removed data-completion/arrival negatives fail. Local spills decrease,
but stack88/80B remains; this is NOT a spill-free implementation.37 host
tests pass in the Torch-capable environment. The local default Python lacks
Torch; its four import errors were not reported as passes.

Device14/14 CPU cases and S16 output/state fingerprints/errors pass.
First timing attempt`independent-fi-weak` is **INVALID**, stopped before
capture by the idle guard observing foreign PID143838. No timing is inferred
from that attempt; no foreign job was stopped. Fresh exclusive retry
`independent-fi-weak-r2` gives142.016[140.257,143.009]us versus S16
142.321[140.993,143.616]us: **UNRESOLVED, do not promote**. FI no-CP
114.4165[113.025,114.881]us remains faster. Eight-repeat/every-captured-call
checks passed. This rejects the claim that the inter-WG issue preference
alone explains the remaining gap; it does not price other dependency waits.

S16 also passes four actual bodies against the PPU CUTLASS3.6 dependency in
CUDA source-check mode. Native PPU1.7 compile/execution/performance are still
unavailable, not inferred from the H800 result.

## S19: specialize full auxiliary chunks, retain the dynamic final chunk

Branch `sm90-aux-full-tail-20260926`, kernel`64691d1`, parentS16. This follows
the selected FlashInfer auxiliary structure, not a new gate algorithm. The
same host/device visitor covers every length1..4096 exactly once; omitting
its dynamic final block fails. Actual TiledMmaQK coordinates cover4096 cells
once, all within64x64. Thus a nonfinal full chunk needs only the causal mask;
the final chunk retains both row/column bounds and the safe exponent mask.
The count uses `1+(length-1)/64` for positive int32 lengths; INT_MAX is a
compile-time boundary check, not an overflow-prone `length+63` expression.
The preliminary local build before that integer-boundary correction is NOT
the admitted source/binary; `full-tail-local-r2` is the final local build.

Native full-aux PC interval changes from S16`0x1500..0x62b0` to
S19`0x15b0..0x5ce0`:1244→1140 static sites, ISETP84→69, same16 WGMMA,
14 HMMA and21 LDS sites. It also changes compiler allocation: whole-body
stack88/80→24/32B, while full-aux LDL0→3. State source is unchanged, but do
not claim the eventual timing difference is solely the removed mask checks.
The extra final-block body increases total code; static sites are not executed
work. All four local/native remote bodies match,32,064 normalized instructions,
SHA`782a2f0cb34b84b2d0528881d95a64c2901261d1d4964f6ec8caad60bccc61e8`.

35 host tests and14 independent device CPU/fingerprint cases pass with
unchanged errors. `full-tail-fi-weak` and `full-tail-fi-weak-r2` are INVALID:
idle guards saw foreign PIDs150614 and153467 before capture. No foreign jobs
were interrupted. Exclusive retry `full-tail-fi-weak-r3` is admitted:
135.2965[134.880,136.641]us versus S16
141.521[140.576,142.080]us, disjoint CANDIDATE-WINS (1.046x).
FI no-CP113.185[112.737,113.664]us still wins. Final confirmations:

| Paired capture | S16 control | S19 | Fastest reference | Verdict |
|---|---:|---:|---:|---|
| FI, g=-0.1 |141.5210 |135.2965 |113.1850 no-CP |S19 improves; FI wins |
| FI, g=-1.0 |140.2720 |134.7690 |112.0000 no-CP |S19 improves; FI wins |
| QLA, g=-0.1 |140.5445 |133.4560 |165.6320 auto-CP |S19 wins |
| QLA, g=-1.0 |140.4480 |133.3600 |163.4405 auto-CP |S19 wins |

Units:microseconds. Every stated win/loss has disjoint observed envelopes.
Parent speedup1.041–1.053x; QLA speedup1.241x/1.226x. S19 remains
1.195x/1.203x as slow as FI. Eight stable repeats and every captured output
pass the fixed gates. Both S16 and S19 pass four-body PPU CUTLASS3.6 CUDA
source checks; native PPU1.7 performance remains SKIP.
Binary SHA`e9e2355f6b5d5a7f760c3b6b64d198c9b6607ea93de5621896ae2e5da0b6f0b7`.

No new default selector is installed. The experimental branch keeps the full
build/test/profile tools; the admitted H800 binary is under
`/workspace/gdn-sm90-win-20260926/full-tail-build/`. Next investigation is
the remaining **matched auxiliary full-body** instruction/operand lifetimes
versus FI (1140 versus920 static sites, same matrix work), not another blind
barrier deletion or register cap. Static differences alone do not locate the
whole-call critical path. Keep native PPU and physical H800 conclusions apart.

## Evidence packaging

The eleven admitted follow-up captures contain744 complete forwards;
all SQLite results were exactly re-extracted with the original Python3.12
analyzer. Source, binary, receipt, SQLite and nsys hashes are in
`dev/backends/sm90_sass_followup_20260926.json`. Numerical admissions are
additional to that timing denominator; rejected idle attempts are excluded.
Archive through09:09UTC:
`/workspace/gdn-sm90-win-20260926/remote-evidence-followup-20260926T0909Z.tar.gz`,
SHA`a0b24ed39f551c93f5335c17e44fe1198115a169fe58856d4ced3e883411fa8c`.
All28 receipt/result/SQLite/nsys member hashes verify for the first archive. The source/binary
archive includes the rejected attempts and S19's numerically admitted build;
later S19 timing captures are in the separate supplement
`remote-evidence-s19-confirmed-20260926T0914Z.tar.gz`, SHA
`658786472f2c328eb2269b7a3eb4776fe3f638edb04952a083dd65b52b473d02`.
Its16 member hashes also verify. The original campaign archive is retained
unchanged. All experiment branches are pushed; no GPU job remains for this
bounded inventory. Rejected and compile-only candidates are not promoted.
