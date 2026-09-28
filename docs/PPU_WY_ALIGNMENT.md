# Explicit WY candidate: align the weak path, keep the original fast paths

This is forward-only C++/CuTe/actlize, not a Triton wrapper. It adds
`gdn_chunk_wy`; it **does not replace** `gdn_chunk` or change auto thresholds.
The original seven TUs, target adapter and actlize submodule remain unchanged.
The plan and predeclared target are in [plan.md](plan.md).

## Why this decomposition

Let R be the inverse of the chunk's unit-lower system, B=diag(beta), and Kd
the cumulatively gated keys. The original weak loop evaluates
`Vnew = R @ (B @ V - B @ Kd @ H)`. WY distributes this into chunk-independent
`U = R @ B @ V`, `W = R @ B @ Kd`, then `Vnew = U - W @ H`.
Only `W @ H` and the state update remain in the dependent state loop.
This is an algebra identity, **not** a BF16 bit-equality promise.

| Stage | Mapping at B1/S2048/Hk16/Hv32/D128 | Precision / lifetime |
|---|---|---|
| Prefix + KKT/inverse + W/U | 1024 CTAs x 128 threads; C64 | BF16 dots -> FP32; FP32 16x16 diagonal solve, TF32 high/residual block merges; BF16 W/U |
| State | 128 CTAs x 64 threads, V32 per CTA; 32 sequential chunks | Native FP32 state stays in registers across chunks; BF16 MMA operands and incoming-state snapshots |
| Output | 1024 CTAs x 128 threads; chunk-parallel | BF16 operand dots / FP32 accumulation / BF16 output |

One warp owns all K128 rows for V16; V is independent, not a reduction axis.
This avoids a cross-CTA state reduction. It is deliberately not a claim that
our state thread layout is identical to FLA's observed 128-thread layout.

Native GVA addressing avoids materialized repeated Q/K. There is no gate
mean/readback/host synchronization in the explicit WY call. Workspaces use
`empty`, with complete producer ownership and tail zero-fill; they are not
blindly cleared. All temporary allocation and all three launches remain in
the full-call benchmark. Initial FP32 state is optional; final state is FP32.
Q/K/V/beta are BF16; g may be BF16 or FP32 natural-log decay. D=128 and
contiguous input are explicit constraints. No implicit normalization or gate
activation is introduced. Hk16/Hv64 is supported by the new API, not by a
silent change to upstream's Hv32 serial restriction.

## What is retained beyond FLA's generic route

1. Original admitted strong-decay B-only preparation and fused reset replay
   can avoid the generic full-history work. Its approximation tests and
   thresholds stay intact; it is not assumed valid for arbitrary activations.
2. Original persistent register state, eight-warp replay and double-buffer
   next-chunk prefetch stay intact. Their previous benefits must be compared
   on the same input/protocol, not transferred to the weak route by name.
3. Full affine scan, Blelloch and Hillis-Steele fallback stay available. The
   generic WY path does not replace them or discard non-reset history.
4. The six-product Neumann inverse stays in the original path. Replacing
   the new solve with it is a separate numerical/performance candidate.

The new prepare also fuses prefix/solve/WU rather than writing an inverse to
global memory between those phases. That saves intermediates and launches,
but increases per-CTA shared memory. **This is a hypothesis to measure, not
an already demonstrated advantage over FLA.** If prepare dominates, split
it before adding unrelated tuning axes. Never weaken reset admission to
make an unfavorable weak case take a faster approximate path.

## Local evidence and remaining boundary

`scripts/verify_ppu_wy_local.sh` runs the independent token-recurrent CPU
oracle and explicit rounding model, actual actlize fragment/TF32/shared maps,
old-source preservation, host contracts and real SDK compile/link/resource
checks. Five numerical negatives plant inverse sign, wrong GVA head, history
reset, causal-mask loss, and using the updated rather than incoming snapshot.
Four ownership negatives plant duplicate V slices, wrong TF32 mapping, a
missing chunk, and a register-slot permutation. Binary negatives require
spill, missing native MMA and a missing cross-TU launcher to fail.

These tests cannot establish runtime image loading, PPU instruction semantics,
race freedom, exact BF16 rounding or speed. **No local PPU is available.**
At this local checkpoint no device result or automatic routing promotion was
claimed. Subsequent user-reported 16-case admission and first timing results
are recorded in [PPU_WY_FIRST_DEVICE_RESULTS.md](PPU_WY_FIRST_DEVICE_RESULTS.md):
WY versus FLA remains UNRESOLVED, and routing is still unchanged. Use the
[reuse-only ACU command](PPU_GDN_ACU.md) for the next experiment; the command
below is the original build/admission handoff, not required again to profile.

Local checkpoint (2026-09-21): 3/3 compiled host tests; 45 arithmetic cases
(including both S2048 head counts) plus five numerical negatives; 33 Python
contract tests; 305 original control expressions preserved; original 15 and
new 3 device images compile/link. Native-map and binary negatives all reject
their planted faults. New prepare/state/output use 84/244/80 vector registers
and zero stack. Their explicit dynamic shared allocations are 70,144/37,120/73,984 bytes.
The initial 32-byte-spill candidate was rejected, not promoted. These are
compiler/host facts, **not measured PPU utilization or latency**.

## Device handoff

On an idle PPU, in GDN-QSA-sm80:

```bash
git pull --ff-only
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 \
  bash tools/run_ppu_wy_fla_box.sh
```

It rebuilds for the installed SDK/PyTorch, verifies all original routes,
then tests 16 WY cases (tail, GVA, nonzero initial state, FP32 gate, output-only,
Hv64 and eight-repeat stability). A numerical failure stops all timing.
`PERF=0` runs admission only and needs no FLA installation. Otherwise installed
PPU FLA is mandatory: there is no silent two-arm fallback. `FLA_ROOT` is only
needed to select a checkout instead of the installed package.

The comparison uses identical tensors and the existing 2% output/state gate,
separate weak/strong cases, native GVA, zero initial state, returned final
state and no QK normalization. Six balanced execution permutations alternate
the three arms; no kernels from different arms run concurrently. JIT/autotune
and five warmups are excluded. All stage/allocation/dispatch costs remain in
the public-API event span. Raw samples, errors, state dtypes, fingerprints,
source SHA/diff, submodule pin and binary hashes are saved under `/workspace`.
Overlapping observed timing envelopes are UNRESOLVED; both winner directions
are reported. Do not compare these API spans directly with ACU kernel sums.

Do not apply this benchmark's synthetic g distribution as Qwen model routing
evidence. Actual model gates, normalized Q/K, state lifetime and TP-local heads
need a separate integration admission.
