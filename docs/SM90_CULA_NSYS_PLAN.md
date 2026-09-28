# cuLA original / scalar GDN: physical H800 trace comparison

Registered before measurement, 2026-09-26 UTC. This is a benchmark, not a
kernel/default-route change or a PPU1.7 native performance claim.

- Parent: faa7bf5 (implementation ccd6703); original cuLA:
  79be249e61453808e18e5cef7702b363239e7d8d, its pinned CUTLASS and FLA.
- B1/T2048/Hqk16/Hv32/D128/C64, BF16 Q/K/V/beta, scalar natural-log
  gate from the existing CPU fixture, g=-0.1 and -1.0, zero initial state,
  FP32 final state. Same quantized input values and CPU recurrence oracle.
- For cuLA, broadcast scalar gate over K without changing values. Its original
  preprocessing remains in the complete-forward trace. Also report the cuLA
  fused C++ kernel alone as a component, NOT as the complete-forward total.
- Use the original C++ Hopper path explicitly, not DSL/auto-CP/another backend.
  Record original upstream build flags (including register-usage-level=10 and
  fast-math) and dependency pins. Do not retrofit these to our incumbent.
- Correctness: existing max/max error <2% for output and final state, finite
  results, eight repeated output fingerprints. Gate/layout conversion overhead
  is identified, never silently hidden in a complete-forward claim.
- Physical GPU idle check before capture; monitor competing GPU processes
  throughout. An observed unrelated workload invalidates the timing. Do not
  terminate it or change clocks/power/permissions.
- nsys CUDA/NVTX trace, no statistical CPU sampling and no GPU counter replay.
  Warmup/JIT/reference outside capture. Label each forward, sum ALL its GPU
  kernel durations, report GPU span and gaps separately; preserve raw reports,
  binary/source hashes, per-call records and telemetry.
- Compare observed ranges; overlapping ranges => UNRESOLVED. No event/API
  latency substituted for nsys kernel duration, no hardware-neutral speed claim.
- Negative controls: missing call, unassigned kernel, duplicate call, wrong
  input/binary identity, foreign GPU PID must not become a valid comparison.

WGMMA diagnosis is separate: distinguish recurrence, explicit data-dependency
waits and ptxas register-pressure C7512. Compare original compiler output; a
function-level warning alone does not prove every async instruction serialized.
