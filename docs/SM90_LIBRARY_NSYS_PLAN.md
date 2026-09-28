# SM90 FlashQLA / FlashInfer comparison (registered before measurement)

Scope: physical H800, not native PPU1.7. No shipping kernel or routing changes.
Reference heads resolved from the official repositories on 2026-09-26:

- FlashQLA `a97c9783bbcc42fa8fbfe895dfc674131e376b5c` (TileLang 0.1.9,
  TVM-FFI 0.1.9).
- FlashInfer `5d9f8c8d97fa53e22952ce8672f475d235f07478` (SM90 CuTe DSL,
  dependency versions recorded in receipt; separate environment from FlashQLA).
- Incumbent: unchanged, hash-bound binaries from `SM90_CULA_NSYS_20260926.md`.

## Workload and admission

Same CPU fixture as the cuLA comparison: B1/T2048/Hqk16/Hv32/K128/V128,
BF16 Q/K/V/log-gate/beta, scalar natural-log gate -0.1 and -1.0, seed
0x6A09E667, scale=1/sqrt(128), no Q/K normalization, zero initial state,
FP32 final state. Reference is the independent recurrent CPU implementation.
Output **and** state max-error/max-reference must each be <2%, unchanged.
Eight repeated fingerprints and every captured output must remain stable.
Wrong gate representation, wrong state layout, and zero output must be rejected
by the numerical check; not merely by source-string checks.

Public inference forward, no backward or backward-only CP cache. Both upstream
automatic context-parallel selection and explicit CP-off are measured. The
actual symbols and all helpers establish which path ran. Unsupported/failed
arms are explicitly unavailable, never zero-time or assumed performance.

FlashInfer expects flattened Q/K/V, FP32 alpha=exp(log-gate), FP32 beta and VK
state. Prepared-native-ABI timing uses CPU-prepared representations, with hashes
and exp/log round-trip error recorded. A separate `flashinfer-auto-log-adapter`
arm includes GPU float/exp conversions inside the measured forward. The CPU
oracle interprets state VK->KV outside the measured call. FlashQLA and our
incumbent consume log gate/KV state. No weights, values or algorithm tolerances
are changed to improve timings.

## Timing and interpretation

Use `nsys --trace=cuda,nvtx --capture-range=cudaProfilerApi`, 12 calls per role
and gate. JIT, numerical checks and warmup are outside capture. Rotate/reverse
role order each round. Latest upstream dependency constraints require separate
reference environments; each environment includes the same two incumbent
binaries as an interleaved bridge. Cross-environment comparison must disclose
this distinction and contemporaneous incumbent drift, not pretend every role
ran in one process.

Primary metric: **sum of every GPU kernel duration belonging to one complete
forward**. Memcpy/memset, GPU span/union/gaps and host range are separate fields.
Never substitute API/events, omit preprocessing, or compare sums to spans.
Winner only when observed kernel-sum ranges are disjoint; otherwise UNRESOLVED.
Kernel-count changes are reported, not automatically disqualified for upstream
CP paths. Missing/duplicate calls, extra/unassigned kernels, missing kernels,
wrong role denominator or foreign GPU work invalidate measurement.

Three no-process/0%-util observations precede CUDA work; process/telemetry
monitor throughout. Do not interrupt other users' jobs or change clocks/power.
Polling cannot exclude a foreign task shorter than its interval; disclose it.

Report both decay regimes, auto and CP-off (even if auto loses), full kernel
breakdown, accuracy, pins/binaries/JIT hashes, versions, input identity and raw
sample ranges. Old cuLA results remain historical unless remeasured; no general
shape or physical PPU performance claim from this one workload.
