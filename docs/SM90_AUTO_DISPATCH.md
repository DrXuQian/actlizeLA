# Automatic SM90 forward

Normal install: `CUDA_HOME=/usr/local/cuda bash tools/install_sm90.sh`.
Normal call: `gdn_forward(q, k, v, g, beta, initial_state=h0)`.
There is no caller-owned configuration choice.

## Ownership and selection

`actlize_la/sm90_policy.json` is the single runtime table. Its inputs are
`B,T,Hk,Hv,K,V` plus device name, capability and SM count. K=V=128, chunk=64
and the existing BF16/state contracts remain unchanged. The device is inferred
from the input tensor, not the current default CUDA device. SM90 builds remain
independent of PPU1.0/1.7; PPU generation must still be explicit.

These are **exact shape matches** for the measured H800 PCIe class (CC9.0,
114 SM), not broad thresholds inferred from a few points:

| B | T | Hk | Hv | Selected configuration |
|---:|---:|---:|---:|---|
| 1 | 512, 1024, 2048, 4096, 2051 | 16 | 32 | `value64` |
| 1 | 8192 | 16 | 32 | `value64-local-inverse` |
| 1 | 2048 | 8 | 16 | `value64-local-inverse` |
| 2, 4 | 2048 | 16 | 32 | `value128-paired` |
| 1 | 2048 | 16, 32 | 64 | `value128-paired` |
| 1 | 2048 | 32 | 32 | `value64` |

This is the retained selection in
[the H800 closure](SM90_H800_CLOSURE_20260927.md): 12 shape keys, 14 workloads
(FP32-gate/initial-state variants share the base shape), both decay regimes.
The independent test checks all 56 reference cells: 55 valid selected rows
and the explicitly retained reference numerical failure. No reference failure
is counted as a performance win. This wiring is not a new device measurement.

Any other supported SM90 shape uses `value64` with basis
`unmeasured-shape-default`; a different SM90 device uses the same configuration
with basis `unmeasured-device-default`. This is executable default behavior,
not an optimality claim. Shape-based selection does not certify unmeasured
gate distributions, dtypes or state combinations as performance winners.
Invalid tensor/head layouts are errors, not an excuse to select SM80.

Optional introspection:

```python
from actlize_la import load_sm90
print(load_sm90().select(q, v))  # configuration, basis, policy_id; no launch
```

## Build once, select without GPU work

The installer compiles all three candidates, imports their actual binaries,
checks matching source/compiler/dependency/Torch receipts and registers one
complete hash-bound bundle. At first use the loader verifies the policy,
receipts, binaries and compiled configuration/math labels. A missing arm,
mixed build or changed registration fails closed. It never silently replaces
the selected candidate with a different implementation.

After first use, only cached device metadata and tensor dimensions are read.
No gate data is copied to host, no synchronization/autotuning is introduced,
and no kernel is compiled during a call. Initial-state and final-state flags
are passed through unchanged. Warm up once before timing to exclude native
module loading; Python dispatch latency is not claimed to be zero.

Each candidate remains a separate native specialization. Selection happens
before launch, so it adds no branch, predicate or extra code to the GPU kernel.
The historical native-body/resource identity checks therefore still apply;
this change modifies no C++/CUDA/CuTe implementation.

## Regression requirements

- Every retained workload maps to its measured candidate across both gates.
- Removing a default-valued table row must fail the inventory check; returning
  `value64` by fallback cannot hide a missing measured row.
- Planting the wrong long-sequence winner must fail independently of table code.
- All three native module labels must coexist without pybind reusing the first.
- Missing candidates, changed receipt/policy hashes, mixed sources, escaped
  directories and wrong compiled labels must fail before any forward launch.
- The public default call passes all input/state objects unchanged and never
  invokes a compiler. Subsequent calls do not rehash/reload the bundle.
- A wheel includes the policy but excludes the local absolute installation
  record; installation paths are not portable package data.

Fixed-arm overrides are retained solely for profiling and reproduction; see
[installation](SM90_INSTALL.md). No PPU route or arithmetic policy is changed.

Local validation, 2026-09-28: **191 Python tests, 24 PPU host CTests and four
real CuTe map proofs pass**. Three retained builds import together; 12 native
bodies/resources remain identical and all 12 native negative plants fail as
expected. The public auto dispatch reaches each actual module in editable and
relocated-wheel installs with only its final device call mocked. No GPU was
launched. See the [machine-readable record](../dev/backends/sm90_auto_dispatch_20260928.json).
