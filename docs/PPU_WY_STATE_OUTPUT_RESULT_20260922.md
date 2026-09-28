# State/output combination: reported result and next capture

Update: the requested archive has now been received and independently
verified, including the weak case and all three profiled stages. See
[the verified ACU verdict](PPU_WY_STATE_OUTPUT_ACU_20260922.md). The text
below preserves the pre-capture record and interpretation criteria; its
"not yet supplied/verified" statements describe that earlier checkpoint.

Evidence: user-pasted 16 samples per role from
`/workspace/gdn-wy-fla-90eafeb-20260922T013928Z/comparison.json`.
Medians and envelope decisions were checked against those samples. This is
not yet a locally received/hash-verified bundle. Shape B1/S2048/Hk16/Hv32/
K128/V128, g=-1.0, BF16 inputs, zero initial state, final state returned.
Timings are complete public-API event spans, including allocation/submission.

| Role | Median us | Observed range us | State+output versus this role |
|---|---:|---:|---|
| original | 478.898 | 425.924–517.936 | UNRESOLVED |
| scalar WY | 710.210 | 705.772–722.768 | CANDIDATE-WINS |
| tiled prepare | 767.866 | 758.656–779.024 | not a registered direct control |
| tiled state | 458.998 | 453.540–465.760 | CANDIDATE-WINS |
| tiled output | 693.508 | 688.492–699.056 | not a registered direct control |
| tiled state+output | 439.486 | 437.684–444.076 | subject |
| tiled all | 446.700 | 443.052–454.620 | UNRESOLVED |
| FLA | 730.812 | 486.836–968.548 | CANDIDATE-WINS |

The unchanged rule is disjoint observed envelopes; no outlier removal.
Raw output/state equality to scalar is reported PASS. Original final state
is BF16, whereas WY and FLA return FP32; retain this precision boundary.

State+output lowers same-run scalar median by 38.12%. Adding tiled output
to tiled state saves 19.512 us and wins with disjoint envelopes. Adding tiled
prepare to that pair changes the median by +7.214 us, but their envelopes
overlap by 1.024 us: **do not declare the pair faster than all**. Original
also overlaps, so do not claim a new winner against the strong/reset route.

Isolated tiled-prepare costs +57.656 us versus scalar in this run. That is
not its +7.214 us marginal change with both later stages tiled. The pair's
additive prediction from state/output single-stage changes is 442.296 us,
versus 439.486 us observed. These are whole-call interactions, not isolated
stage timings and not a measured cache/host-overhead explanation.

FLA spans 486.836–968.548 us. The subject's maximum is below FLA's minimum
by 42.760 us, so the registered within-run winner remains valid. However,
the median ratio 1.6629x is descriptive of this noisy run, not a stable gain
claim or a reason to discard FLA's slower samples. Neither clocks nor host
dispatch have been established as the cause. Weak g=-0.1 has not been
supplied; do not infer its result from the strong case.

Decision: keep state+output as the next profiling subject; retain all and
original, and do not change default/automatic routing or numerical criteria.

## Next capture: existing binary, all three stages

The existing collector already supports this exact role. Run from the box
checkout, on the same physical card as the comparison (replace DEVICE=0 if
the previous run used another card):

```bash
git pull --ff-only &&
env -u OUT -u EXTENSION \
  PPU_SDK=/usr/local/PPU_SDK \
  ACU=/usr/local/PPU_SDK/asight/bin/acu DEVICE=0 \
  bash tools/run_ppu_gdn_fla_acu_box.sh \
  --wy-run /workspace/gdn-wy-fla-90eafeb-20260922T013928Z \
  --wy-delivery tiled-state-output --gate -1.0
```

No rebuild, installation, clock change, new sweep or profiler API. The
collector validates both reused DSOs, input/output/state fingerprints, FLA
identity and the exact `wy-tiled-state-output` comparison role. It performs
separate unprofiled preflights, then direct `acu -f -o ... --set full` captures
of one public call each, WY and FLA sequentially. Existing directories are
never deleted. Upload the single tar printed after `UPLOAD=`; native reports,
plain-text exports, full preceding comparison JSON (including weak if present),
code/binary identities and capture logs are bundled. No CSV copying required.

### Predeclared interpretation

| Actual WY stage required | FLA work to match | Source grid / threads | Source shared bytes |
|---|---|---:|---:|
| `gdn_wy_prepare<false>` | prefix + KKT/solve + W/U | 1024 / 128 | 70,144 |
| `gdn_wy_tiled_state` | chunk recurrence, H snapshots, Vnew | 128 / 128 | 49,408 |
| `gdn_wy_tiled_output` | chunk output QH + PV | 1024 / 256 | 49,408 |

Mask48 must not run `gdn_wy_tiled_prepare` or scalar state/output. Check
actual names and launch geometry before attributing a counter. Local prior
register observations for the selected stages are 84/232/98 with zero stack;
the reused box binary's resource report, not these numbers, is authoritative.

For **each** stage record duration, total/per-opcode instructions, BF16/TF32
MMA work, registers/stack/shared, active warps, achieved CE frequency, fetch/
memory/sync stalls and KVD/TSM/L2/DRAM bytes/requests. Match arithmetic work
and cast/precision boundaries before interpreting instruction differences.
In particular, prepare still retains the TF32 high/residual three-product
solve; matching a cheaper plain-TF32 reference is not an authorized shortcut.

Check whether the scalar prepare's narrow-store footprint returns in this
combination; check whether tiled state/output reduce address/shuffle/control
work and improve useful warp delivery. These are questions, not predictions
of a win. Rank the next edit by the largest measured stage gap, including
prepare/output rather than presuming recurrence is still dominant.

FLA's actual installed stage split is authoritative. List fills, transforms
and repeated autotuning launches separately instead of forcing them into a
matmul stage. A one-call capture can still contain internal tuning launches.
The prior benchmark recorded device properties rather than UUID; it does not
prove cross-run physical identity. Both new capture receipts must match.

ACU replay/cache/process state differs from API timing: never subtract the
profiled sum from 439.486 us and label the remainder host overhead. If the
device stages do not explain FLA's broad API distribution, a same-protocol
timeline is needed; do not infer throttling or alter the timing verdict here.

This handoff changes documentation only. No new device performance result.
The current local Python has no torch: the attempted ACU-contract rerun
stops at import (ENVIRONMENT BLOCKED), not PASS. The prior 23-contract PASS
belongs to the 1b24b37 handoff; collector/profile code is unchanged since then.
