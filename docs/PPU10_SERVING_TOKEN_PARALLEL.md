# PPU1.0 serving prefill: token-parallel transport experiment

## 当前源码基线：ZW610 与 H1 direct-pack

本性能分支的 PPU10 源码构建默认启用 token-parallel 与 direct-pack；非PPU10默认不变。
`ALA_PREFILL_TOKEN_PARALLEL=OFF` 保留串行control，token-parallel ON 时可用
`ALA_PREFILL_DIRECT_PACK=OFF` 关闭H1并使用当前token-parallel基线。

H1只在 B=1、T=2048或8192、Hk=16、Hv=32/48/64 的已验证容量几何上默认选中；
其他shape安全回退到token-parallel prepare+pack。选择只依据immutable descriptor，
不读device starts/live/slot或tensor内容；准备阶段可查询
`ala_prefill_direct_pack_route` 与 `PreparedGDN.transport_route`。

融合prepare→packed保留原warp归一化树、epsilon、beta的BF16边界及FP32 natural-log gate；
不增加caller-owned workspace对齐要求。旧queries/keys/decays/betas scratch不再写，
不声称这些废弃中间值逐bit相同；packed/output/FP32 state/errors均需RAW-BIT一致。
Core仍是保留的full-chunk/guarded-tail四kernel；未晋级的core实验不作为基线提交或默认选择。

ZW610/SDK2.2.1/CUDA12.9：H1登记12格配对ACU helpers降39.28%–44.69%，完整serving
降17.14%–19.88%，基线始终为当前token-parallel，不用旧3ms串行数字作收益锚点。
54格core vs FLA 0.6.0只有24格达严格20%耗时门、28未达、2包络重叠；总目标未完成。
这些是synthetic算子完整ACU kernel-sum，不是API latency、真实模型或ZW810新结果。

公共默认6个H1几何及3个未登记shape回退图验证通过，实际8/9 kernel、0host/alloc、8次重放稳定。
完整serving55/aux42/native graph、mutable metadata RAW-BIT与单bit/错误route负控已通过。
可在匹配SDK下复现公共路由门：

```bash
python tests/prefill_parallel_device.py --direct-pack \
  --baseline-run <当前token-parallel-ON且direct-pack-OFF的已准入run> \
  --candidate <本次H1默认构建的libactlize_la_serving.so> --results <新结果路径>
```

以下保留此前token-parallel实验与ZW810交接的历史记录，其默认OFF/未晋级描述不代表上述新基线状态。

## 历史实验检查点

Status: candidate device numerical and ACU results returned for T2048/Hv32/48;
still opt-in, **not promoted to release/default**. Main, v0.2.2, installed
wheels and the GDN core selector are unchanged. The special old/new22-cell
parity receipt is in the separate candidate bundle, not the uploaded profile
bundle; do not claim that receipt was independently inspected.

Local checkpoint:265 host tests +38 subtests PASS;2 native ABI tests SKIP
(no runtime library loaded in that host test process, not counted as PASS).
Both OFF/ON transport device bodies compile with SDK2.2.1 for PPU1.0. Candidate
DSO links against byte-unchanged released core objects with `--no-undefined`;
this is not a fresh full CMake rebuild or device execution. No-device runner
returns3/SKIP. Box script rebuilds the full exact candidate and requires all
device gates. Local linked DSO SHA256:
`39357f8d76583ed2f5bdef15da5bfaf06965ebb3e5748096be3ad5b241544eab`.

## Observed problem

The 2026-11-06 upload has one complete ACU serving capture for B1/T2048,
Hk16/Hv32,K=V128,BF16 operands, FP32 asymmetric state, g approximately -0.1.
It contains nine kernels, not just GDN math:

| Kernel | Duration (us) |
|---|---:|
| validate_gdn | 5.58882 |
| prepare_gdn | 1633.47118 |
| pack_prefill | 1061.14529 |
| gather_state | 14.08529 |
| GDN prefix + solve + state + output | 169.76353 |
| scatter_prefill | 419.04412 |
| Complete serving sum | 3303.09823 |

Source mechanism: prepare launches only B*(2Hk+Hv) **one-warp** CTAs and
serially traverses T tokens. Pack and scatter also expose only B*heads CTAs.
These are independent-token operations; they are not the recurrent state scan.
The core already uses the retained full-chunk residual implementation.

Evidence: uploaded results.tar.gz SHA256
`f31c5788f57057c246a030e0d11e666551e21c2930f94987a5c47e2061c06b7a`;
binary SHA256 `ef2dbfd30af2f095e0b01c749560cfabf9fdddab2814eeb6e181d1d793a2db83`.
The upload is incomplete only because the collector expected a wide CSV rather
than the site's long metric rows. FLA/default capture **did not run**. This is
one unpaired observation, not a resolved speedup or a whole-model measurement.
The reduced real Duration/Active Cycles rows are a parser regression fixture in
`tests/data/serving_acu_long_duration.csv`. Original reports stay on the box.

## One change: expose token parallelism

Build `-DALA_PREFILL_TOKEN_PARALLEL=ON` (default OFF, PPU10 only).
Launch z=`min(ceil(width/32),65535)`, using fixed capacity, not CPU-read lengths.
Prepare uses128 threads: one complete warp per head/token, four FP32 values
per lane and the **same XOR sum tree/epsilon**. Independent tokens stride over
the added warps. Pack/scatter distribute their original element intervals over
z as well. Final-state writes are partitioned too, never duplicated per block.
Grid z is capped and loops cover additional work without truncation. Scalar
address intervals use64-bit arithmetic where a grid stride could overflow int.

No new buffers/allocations, no kernel-count change, no state-layout change,
no changed BF16 boundary, no extra query scale, no gate approximation or fusion.
Gather, GDN core and decode/verify/commit scheduling remain unchanged. The
preparation-time `ala_prefill_token_parallel()` query binds the compiled arm;
it does not inspect a device or execute a kernel. Both old/new launch ABI and
workspace extents are unchanged.

## Admission and execution

1. Host coverage proof plus missing/duplicate ownership negatives. SDK2.2.1
   PPU10 compilation of both changed device translation units.
2. Existing full serving + auxiliary + native-graph device gates unchanged.
3. Same-input old/new parity for meaningful scratch, output and state, Hk16,
   Hv32/48, T31/32/33/63/64/65/127/128/129/2048/8192. Eight graph replays change
   live counts, lengths and initial/final slots; include null final and empty
   requests. Exact bit comparison and planted one-bit failure, no tolerance
   relaxation. Independent recurrence remains in the full device gates.
4. ACU capture of candidate serving, public default core and FLA, repeated
   serially with identical processed inputs and unchanged numerical admission.
   Count every helper. Kernel sums are not uninstrumented API latency. Final
   production routing waits for this evidence; no speedup has been measured
   locally. An improvement over the uploaded n=1 baseline is descriptive,
   not a same-session paired confidence claim.

On an idle PPU1.0, in the diagnostic branch checkout:

```bash
BASELINE_SERVING_RUN=/workspace/actlize-la-flare-ppu10-226355a5-20261106T031921Z-3421332 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 JOBS=16 bash tools/run_ppu_prefill_parallel_box.sh
```

To profile **only the existing default core** at Flash-Next Hk16/Hv48, without
building or waiting for this experiment (numerical preflight is outside ACU):

```bash
SERVING_RUN=/workspace/actlize-la-flare-ppu10-226355a5-20261106T031921Z-3421332 \
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 bash tools/run_ppu_flashnext_core_acu_box.sh
```

Its default is B1/T2048, BF16 QKV/beta, FP32 asymmetric state and synthetic
g~-0.1. Use `TOKENS=8192` for the longer cell. It prints all four core durations
and their sum; this one capture does not establish a repeated speed verdict.

To capture FLA on the **identical saved operands** from the T8192 Flash-Next
core run (no rebuild, no new fixture and no repeated normalization):

```bash
PPU_SDK=/sim/eec/shared/junfu.qx/ppu-sdk-2.2.1-184ee6/PPU_SDK \
DEVICE=0 TOKENS=8192 bash tools/run_ppu_flashnext_fla_acu_box.sh
```

The script selects exactly one successful `/workspace/flashnext-gdn-core-*`
run matching T8192/Hk16/Hv48. If several exist, it lists candidates and requires
`INPUT_RUN=/workspace/<the-specific-core-run>`; it never guesses the newest.
It checks saved operands/library hashes and the original numerical receipt;
the subject also verifies device UUID and unchanged FLA source identity.
Original inputs/reports remain untouched, new FLA artifacts go to a new
`/workspace/flashnext-gdn-fla-*` directory, with a small `results.tar.gz`.
All FLA kernels, including its zero-fill kernels, contribute to the sum.

This builds into a fresh `/workspace` directory, validates before profiling,
uses `/sim/eec/shared/junfu.qx/asight/bin/acu` with its own sections, and prints
small numerical and ACU `results.tar.gz` files. It never installs or overwrites
a binary. Defaults profile T2048/Hv32 and **Flash-Next Hv48**; add
`--lengths 2048 8192` for the extension. Missing device, incorrect arm, numerical
failure or profiler/parser failure cannot be reported as a performance PASS.

## Returned ACU evidence, 2026-11-06

Upload SHA256 `2aa2df873b3461ac0e0b4560e1b4ece3f07c036167cc5dbec8149b7b69eb0b3f`.
Source9286a2a; box binary SHA256
`b357179b1048df9f53ec4edc6b408044164485304c8c8bdc22959721e8690f59`.
PPU-ZW81072CU, SDK2.2.1, site ACU unchanged. All captured CE frequencies1.7GHz.
Reimported18 CSVs: per-kernel values/identities and numerical receipts match
the bundle; aggregate floating summation differences are below1e-9us.
Full serving51 cases and auxiliary42 cases PASS receipts bind this binary;
native graph receipt included. Each performance arm separately passed8 repeats
and serving/default output+state RAW-BIT equality.

B1/T2048/Hk16/K=V128, BF16 QKV/beta, FP32 asymmetric state, g~-0.1.
All table values are **median of three ACU kernel sums**, not API latency:

| Hv | Complete serving | Serving helpers only | Our default core | FLA core incl. fills | Core result |
|---|---:|---:|---:|---:|---|
|32|455.83999us|209.94824us|246.43647us|206.84763us|ours19.14% slower|
|48|566.22412us|260.93235us|305.51176us|288.78118us|ours5.79% slower|

Both core observed envelopes are disjoint: FLA wins. This upload contains
**no T8192 performance cell**; do not label these results as the user's1ms run.
For Hv32 the earlier unpaired serving total was3303.09823us, helpers3133.33470us.
The large helper reduction supports the token-parallel hypothesis, but the
before was n=1 and a different binary: do not call the ratio a paired win.
The unchanged core is now ~246us versus the earlier ~170us; frequency alone
does not explain this because both captures reported1.7GHz. A matched old/new
core comparison is needed before attributing that difference to code or cache.

Hv48 phase accounting (per-phase medians, not exactly additive medians):
our output70.79941us vs FLA56.92647us is a13.87us difference. Our residual
state153.36235us includes inverse-applied residual/value work; FLA's separate
`recompute_w_u` plus state must be combined within each call, median149.60412us.
Comparing only FLA state104.13118us would charge the same work to the wrong
phase. Our solve78.33941us vs FLA70.69235us is another7.65us; FLA's extra fills
remain included in its complete-call denominator.

Serving still spends~261us in helpers at Hv48: pack125.80176us,
scatter66.86706us, prepare48.12765us, gather14.61471us, validation5.26118us
(individual medians). These remain a separate optimization target from core.
