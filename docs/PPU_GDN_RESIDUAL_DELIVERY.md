# Residual delivery experiments (forward only, opt-in)

Parent9df5675; unchanged mathematical contract gated-inverse-residual-bf16-v1.
Before: all-kernel230.58412us / residual state129.64353us, SDK2.1.1, 72CU,
B1/S2048/Hk16/Hv32/D128/C64/g=-1. FLA224.30883us; goal FLA1.5x unchanged.
Measured Memory Dependency~1.125, TSM LD/ST~.73, VMEM~.31, commit~.055.
These ratios are not disjoint wall-time shares. No device win is inferred.

Each candidate starts from the OLD residual, not from another candidate:

- inverse-prefetch: single-buffer next P after current P readers retire;
  overlap UPDATE, unchanged128CTA/128threads/45568B, no added barrier.
- operands: two compile-time register slots for shared operands in KH/PR/
  UPDATE, load next atom before multiplying current; unchanged scalar math,
  rounding, shared layout and grid. Native overlap/zero spill required.
- V16: eight independent value slices/head instead of four;256CTA at target,
  still128threads, no inter-CTA reduction. More duplicated K/P loads is a
  declared tradeoff. Native traits and load simulation must cover CubeWidth16.

Old residual and23 existing kernel images remain the counterfactual. Local
gates and device RAW-BIT vs residual precede timing. No default routing change.
Candidate selection uses ALL-kernel ACU sum; FLA is captured in the same run.

## Local compile findings and first box arm

SDK2.1.1, native PPU0010, no device execution:

| Arm | Registers | Stack B | Shared B | Static instructions | Static BF16 MMA sites |
|---|---:|---:|---:|---:|---:|
| residual control | 242 | 0 | 45568 | 2092 | 40 |
| inverse-prefetch | 244 | 0 | 45568 | 2101 | 40 |
| shared operands | 244 | 0 | 45568 | 2071 | 40 |
| V16 | 98 | 0 | 35328 | 1172 | 20 |

Native inverse-prefetch retains16 UPDATE MMAs between next-P issue and its
wait; the same-opcode early-wait negative fails. Native control already
pipelines shared loads: operand-buffering changes KH/PR scheduling, not the
UPDATE lookahead. This is NOT evidence of reduced latency or a new pipeline
where none existed. All23 old native instruction/operand sequences are equal.

First arm is V16, selected before any device timing: smaller per-CTA resource
footprint and more independent work are directly verified. Target grid doubles
128->256, while thread count stays128. Static MMA sites per CTA halve; dynamic
useful MMA work must remain unchanged across the doubled grid. K/P staging is
duplicated more, so a regression remains possible. Neither higher occupancy
nor the historical1.5x goal is a promised result.

After pulling `ppu-backend`, run from the repository:

```bash
DEVICE=0 JOBS=16 CANDIDATE=residual-v16 \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Use the installed matching SDK (`PPU_SDK=/path/to/PPU_SDK` only if needed).
The runner builds locally on box, checks old gates, then30 same-math cases
with8 repetitions, GVA/tails/initial-state/output-only checks, and both-gate
FLA admission. It captures residual/V16/FLA sequentially with
`/sim/eec/shared/junfu.qx/asight/bin/acu --set full`; no PPUProfiler and no
API timing verdict. Upload the printed `/workspace/.../acu/acu.tar.gz`.
`GATE=-0.1` explicitly selects the weak-gate capture; default is-1.0.

The other two independent arms use `CANDIDATE=residual-operands` or
`CANDIDATE=residual-prefetch`. Do not combine their changes or run captures
concurrently. A failed numerical/RAW-BIT admission stops profiling. Otherwise
compare ALL four candidate kernels against all four residual kernels and the
complete FLA inventory; also report state time, achieved warps, dependency
counters, bytes and clocks. A faster state but slower total is not a win.

BC-related counters remain diagnostic in this round: no bank-layout rewrite
is included. Record both improvements and regressions without changing the
registered numerical criterion or production routing.

## Local regression closure

Final complete rerun2026-09-24:15/15 compiled host tests,80 Python contracts,
7 compiler-dialect tests,45 old WY and61 residual algebra cases PASS. Original
structure305 controls,26 WY native images and15 original images pass; all23
preexisting WY bodies are native instruction+operand identical. Wrong fragment,
wrong grid, reversed K, missing wait/barrier and serialized-prefetch plants
are rejected. Missing ACU/unknown candidate stops the runner before build;
an old backend with only `residual` still supports the unchanged default API.

Evidence `/workspace/gdn-wy-residual-prefetch-evidence-20260924`, final authority
`final-local-complete.log` and `verify_local.sh`. Device-library SHA256:
`f0d918f03877d79806531f52258881ad21e8599459ecc9c438d702e22fed93cc`.
This is local compile/layout/lifetime/algebra evidence, NOT a PPU numerical
or performance verdict. New-device RAW-BIT and all latency cells are NOT_RUN.

## Next planned work: shared bank-conflict elimination

Requested2026-09-24, queued AFTER these independent scheduling/geometry arms.
Do not mix a bank-layout rewrite into the current comparison.

1. Attribute conflicts to exact load/store PCs and the H snapshot, residual,
   scaledV or inverse plane. Reconstruct per-bank addresses from real layouts;
   aggregate6.93M vs FLA8.55M is not a causal diagnosis by itself.
2. Rework the responsible producer AND consumer together. Preserve native
   AIU.swzl + matching ld.swzl; register-produced tiles need their own proved
   mapping. No unsupported PTX compatibility workaround or one-endpoint swap.
3. Prove coverage/rounding/ownership, use a wrong-bank/permutation negative,
   compile native resources, then compare the relevant conflict/dependency
   counters and complete kernel latency. Lower conflict count with unchanged
   time is a valid result, not an automatic performance win.
