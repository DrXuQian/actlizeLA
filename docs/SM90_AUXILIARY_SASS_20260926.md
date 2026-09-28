# SM90 continuation: auxiliary predicates and relative-decay ownership

Best confirmed experimental candidate: **S24, kernel `bc3c154`**, branch
`sm90-relative-gate-cache-20260926` (tests/native gates at `06b7471`).
It improves the previous confirmed S21 by 4.0–5.9% and beats fastest
FlashQLA by 1.36–1.39x, but is still about **8.1% slower than fastest
FlashInfer**. The requested **beat-both target is NOT MET**.

This follows [the previous campaign](SM90_SASS_FOLLOWUP_20260926.md).
All times below are **nsys sums of every GPU kernel in a complete forward**,
not API/event times. Each row is its own same-input, interleaved/reversed-order
12-call-per-role comparison on the exclusive H800 PCIe, 114 SMs, CUDA 12.8.
Shape: B1/T2048/Hqk16/Hv32/K128/V128/C64, BF16 Q/K/V/beta and scalar natural-log
gate; FP32 state. No Q/K normalization, fast-math change or relaxed tolerance.
FlashInfer `5d9f8c8d97fa53e22952ce8672f475d235f07478` and FlashQLA
`a97c9783bbcc42fa8fbfe895dfc674131e376b5c` stay pinned.

## Latest paired result

| Capture / gate | S21 control, us | S24, us | Fastest reference, us | Verdict |
|---|---:|---:|---:|---|
| FI / -0.1 | 128.6410 | 123.6810 | 114.3370, no-CP | S24 improves; FI wins |
| FI / -1.0 | 126.7210 | 121.2810 | 112.1765, no-CP | S24 improves; FI wins |
| QLA / -0.1 | 126.2090 | 119.1690 | 165.0570, auto-CP | S24 wins |
| QLA / -1.0 | 126.5765 | 120.0965 | 163.1370, auto-CP | S24 wins |

Every stated win/loss has disjoint observed min/max envelopes. Do not combine
119.169 us from the QLA window with 112.1765 us from a different FI window.
For FI the remaining same-window differences are 9.3440 and 9.1045 us.
All reference paths, including adapters and slower CP/no-CP alternatives, stay
in the raw records. Winning against an automatic but slower reference path is
not winning against that library's fastest path.

## S24: compute one coefficient once, without changing its arithmetic

S21 already caches `exp2(prefix)` and `exp2(prefix)*output_scale`. It does
**not** cache `exp2(last_prefix-prefix[t])`: both state warpgroups still
recompute that relative decay for their V rows. These are distinct channels.

The actual CuTe operand map covers 8,192 V-by-token elements exactly once.
Counting unique token positions per thread gives 4,096 thread/token evaluations
for only 64 distinct coefficients per chunk. This is a source ownership count,
not a claim that the compiler executes 4,096 independent scalar instructions.
S24 adds 64 floats per alpha stage (two stages, **512 B**) and lets the existing
independent gate warp publish the relative coefficient alongside the other
channels, before the **same alpha-pipeline commit**. Both state WGs load it
before their existing release. All 416 alpha consumers remain accounted for.
No helper launch, new warp, changed stage count or removed synchronization.

The last live prefix is selected with a warp shuffle, including every tail
length 1..64. An explicit `__fsub_rn` preserves the rounded subtraction from
the previous shared-read path instead of letting it contract into LOG2E
scaling. `exp2f` remains standard, including its subnormal handling. Invalid
tail positions publish zero. This is delivery reuse, not approximation.

Four exact native specializations are checked. Their state role is bounded
by the verified `USETMAXREG` allocation/deallocation transitions, not guessed
from source line labels:

- State static `MUFU.EX2` sites: **96 -> 0** across that role's cloned bodies;
  the producer gains two sites. The recurring full body loses 16, not 96 per
  iteration. Static whole-symbol EX2 sites: 158 -> 64.
- BF16/no-initial state interval: 3,471 -> 2,607 static sites; initial-state
  variant: 4,041 -> 3,197. These are not dynamic executed-instruction counts.
- Matrix, TMA, LDSM/STSM and data-completion/barrier opcode families unchanged.
- Whole-image stack remains 24/32 B and spill loads/stores 20/28 B by variant;
  **not spill-free**. The unchanged compiled register figure is 128, with
  role-specific register redistribution retained.
- Actual remote and local four-body native instruction streams are identical:
  27,856 normalized instructions, SHA256
  `23907fbf1ac595040d9622440188c4a59c53ffe8e8b0aaf1df8dcc0a6340e075`.

The native gate rejects the old uncached image and a planted lost barrier.
The real-map checker rejects a wrong last-prefix source and an omitted owner;
source/lifetime checks reject an omitted factor channel/callback. A first
local compile saw source edits in flight and was correctly rejected; only
`relative-local-r2` and the immutable remote binary are admitted.

## Other candidates, including losses

| Candidate | Change | Same-window candidate / control, us | Decision |
|---|---|---:|---|
| S20 `973c23b` | KK ready before QK empty-slot acquisition | 136.1445 / 133.4570 | LOSE |
| S21 `c5907c6` | Keep only final aux masks; unmask initialized exponent input | 129.0085 / 133.5365 | Confirmed WIN |
| S22 `db16d80` | Predicated in-place diagonal inverse FMA | 139.5525 / 135.4085 | LOSE |
| S23 `9af20e1` | S22 composed with S21 | 135.2965 / 126.7680 | LOSE |

S21 strong FI window: 128.9450 / 133.6005 us. Its QLA confirmations are
125.8410 / 132.5290 versus QLA 165.2505 us, and 125.6970 / 132.4810 versus
QLA 162.7215 us. It remains the parent of S24, not of a silently retained S22.

S20's actual shared buffers are disjoint and its publication interleavings
pass, but retaining QK while finishing inverse does not win full-call timing.
S21's metadata addresses/initialization were proved even for inactive lanes;
the final product selects discard upper-triangle Inf/NaN before publication.
Native full pre-inverse sites fall 621 -> 547, final-tail 725 -> 610. Tail
ISETP sites actually rise 63 -> 64; do not claim every predicate count drops.
The matrix and arithmetic counts stay fixed. Independent overflow stress
g=-8/-10000 passes CPU numerics and direct parent O/state byte comparison;
these are diagnostic stress cases, **not** an expansion of the ordinary
performance harness's admitted [-1,0] range.

S22 removes 35 full-body row copies (38 in the tail) while retaining all
21 diagonal shuffles and 21 predicated FP32 FMAs. Shuffles must stay
unconditional so source lanes participate. It raises whole stack 24/32 ->
40/48 B and **slows** the complete forward. S23 retains copy removal and
S21's mask saving, but also loses; stack 32/40 B. These results disprove
"fewer copies implies faster", not that a specific spill alone caused every
microsecond. Neither losing inverse variant enters S24.

## Correctness and evidence tiers

All five candidates pass the unchanged 14-case independent CPU O/state
criterion (<2%), with parent input/output-state fingerprints and identical
errors. Cases cover tails, GVA, FP32/varying gates, initial state, output-only,
multi-stage wrap and both target gates. S21/S23/S24 also directly compare
both overflow stress output/state byte arrays against their parent. Each
admitted trace repeats eight times bit-stably and verifies every captured
output. S24 has 38 host tests plus the compiled actual CuTe map and native
postconditions. No default selector, actual SM80 backend or PPU1.0 path changed.

PPU CUTLASS 3.6 **CUDA source-check** compiles all four actual S21/S22/S23/S24
bodies. Native PPU1.7 SDK/model execution and performance remain **SKIP:
unavailable**. The historical `ours-ppu-source-check` timing control in the
trace is the earlier immutable source-check image, not S24 running on a PPU.
Do not use any H800 timing as a native PPU1.7 result.

The hash-bound manifest is
[`sm90_aux_campaign_20260926.json`](../dev/backends/sm90_aux_campaign_20260926.json).
Its 11 admitted captures contain **744 complete forwards**, each GPU activity
assigned exactly once. All results were exactly re-extracted with the original
Python 3.12 analyzer, including every float/sample/count/verdict. Two S22
attempts were rejected before capture for foreign GPU PIDs 175958/176303;
they are retained as INVALID and excluded from the timing denominator.
No foreign task or device operating limit was changed.

Raw source/build/numeric/native/nsys evidence, including the invalid attempts,
is archived locally and remotely at
`/workspace/gdn-sm90-win-20260926/remote-evidence-aux-20260926T1054Z.tar.gz`,
SHA256 `72ab926d3ce8ab3ec98086fa3ddba3d167d5cb342c06078e2ed9748d94f15b40`.
The manifest/CSV agree on all 11 captures, and all 44 receipt/result/SQLite/nsys
member hashes are checked. Earlier immutable archives remain unchanged.

Measured S24 binary SHA256:
`43d62a8cabcf072b0e19b787d0c5ac9fd6aff2855ab1fcc5c96c7b73decbbb3c`.
Remote binary:
`/workspace/gdn-sm90-win-20260926/relative-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so`.
The candidate branch contains build, admission and profiler tools; use its
immutable binary with S21 as the explicit control, `CANDIDATE_RAW_BIT=1`, and
a fresh output directory for any future paired run. Production routing is
unchanged. Next investigation should price the remaining auxiliary inverse /
epilogue and pipeline critical path against fastest FI, not assume that
removing more waits or matching another static opcode count is sufficient.
