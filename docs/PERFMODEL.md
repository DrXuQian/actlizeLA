# Explicit performance-model metadata

`mode="device"` remains the default. Selecting a PPU backend does **not**
automatically enable perfmodel. Model mode requires both an explicit compiled
backend (for example `ppu17`) and a positive integer SM count. No model size is
hardcoded into the production API.

```python
from actlize_la import get_device_profile, gdn_forward

model = dict(mode="perfmodel", backend="ppu17", sm_count=20)
profile = get_device_profile(**model)
assert profile.sm_count == 20
assert profile.source == "configured"

output, final_state = gdn_forward(
    q, k, v, g, beta, initial_state=h0,
    algorithm="fused_sm90", **model,
)
```

`get_device_profile` returns immediately from the supplied configuration,
without importing Torch or invoking `device_profile` /
`torch.cuda.get_device_properties`. PPU capability is `None`: compatibility
with a Hopper source graph is not a measured CUDA compute capability.
Configured profiles do not enter the measured-device cache. Changing mode back
to `device` uses the real device query again.

The unified forward still executes the requested implementation. This option
controls **metadata discovery**, not whether to run the kernel. It does not
make tensor allocation, transfers, extension loading or PyTorch initialization
device-free. It is not an ACU control, simulator launcher, latency prediction,
GPU partition, grid override, or a claim that physical hardware has 20 SMs.
The present fused GDN grid is determined by batch and heads, not SM count;
the simulator's own model configuration must agree with the supplied metadata.

An explicit `cuda_sm90` auto call with model metadata uses the existing
`value64` unmeasured default, labeled `perfmodel-unmeasured-default`. Even a
configured profile spelling the H800 name/114 SMs cannot claim one of the
measured H800 winners. PPU does not inherit that table. Binary target,
configuration, math-contract and build-receipt checks remain intact.

## Existing single-call simulator input

Add `--mode perfmodel --sm-count 20` to `tools/run_sm90_gdn.py`:

```bash
python tools/run_sm90_gdn.py \
  --extension /workspace/gdn-sm90-source-check/_gdn_fused_sm90.so \
  --backend ppu17 --source-check --configuration control \
  --mode perfmodel --sm-count 20 \
  --batch 1 --length 65 --q-heads 1 --v-heads 2 \
  --out /workspace/gdn-perfmodel-single-call
```

Use the **actual extension filename** emitted by your build, with its adjacent
`build.json`. `--source-check` applies only to a source-check receipt; omit it
for a native PPU17 binary. The metadata mode does not change this distinction.
The application keeps one target invocation, CPU oracle, no warmup/replays,
and no device event timing. Wrap it in the real model command separately.

The log and `result.json` record `mode` and a `device_profile` containing
`name`, `sm_count`, `capability`, and `source`. Model fields are `configured`,
never `measured`. The receipt still binds the original binary and source.
Missing SM count, nonpositive/noninteger count, unknown mode, implicit backend,
or `sm_count` passed in device mode is an error, never a hardware-query fallback.

## Local verification boundary

Host tests exercise the actual resolver, unified dispatcher and single-call
runner with native calls mocked. Forbidden discovery raises immediately;
a mutation that ignores the mode must fail. Configured identity cannot select
a measured winner, and native mode still queries the requested device ordinal.
These checks do not claim simulator/device execution or a speed improvement.
No C++/CUDA kernel, launch grid, numerical contract or default route is changed.

Local result (2026-09-28): **202 tests PASS, 0 SKIP, 0 FAIL**; 11 cover this
option. Relocated-wheel metadata and unified dispatch also pass with Torch
import and hardware queries forbidden, native computation mocked. See the
[verification record](../dev/backends/perfmodel_metadata_20260928.json).
