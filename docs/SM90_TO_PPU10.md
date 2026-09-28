# Remaining SM90 lessons for PPU1.0

This is a transfer plan, not a claim that H800 speedups carry over unchanged.
Execution primitives stay separate: PPU1.0 uses native AIU and matching shared
readers; SM90 uses TMA/WGMMA and warpgroup resources.

## Priorities

1. **Inverse ownership and static indices.** SM90 S55 localized the final
   inverse partials to their row-owning warps. Its actual primitive is SM80
   warp-level MMA, not WGMMA. Transfer that ownership idea to PPU's off-diagonal
   merge and intermediate publication. PPU's diagonal is already warp-local;
   do not pretend the same cross-warp exchange exists there. Its separately
   implemented `solve-static` candidate removes indirect register indexing and
   is a useful first step, not a measured speed win. Keep the three TF32
   high/residual products and the current rounding order unless a separately
   named arithmetic candidate passes independent admission.
   Its [gate-cache composition](PPU10_GATE_CACHE_STATIC_SOLVE.md) is now locally
   built/proved with a same-binary two-gate ACU runner; device timing is pending.
2. **Full-chunk specialization.** SM90 S19 separates interior full chunks
   from the guarded final chunk; paired H800 full-forward times improved
   141.521 -> 135.2965 us in the recorded weak-gate capture. PPU still passes
   dynamic `valid` into staging and masks. For S2048/C64 all32 chunks are full;
   an exact full-chunk path can remove unnecessary bounds work. Tail and
   causal contracts must remain, and real PPU native code must show the checks
   actually disappeared.
3. **Convert before changing operand layout.** Retained SM90 S69 converts
   NewV in logical accumulator order before operand retile. On PPU, prove
   actual pair ownership and use native packed BF16 conversion/publication
   where possible, instead of scalar casts plus moves. Unscaled and scaled
   NewV have distinct rounding boundaries; do not combine them. Keep
   AIU.swzl paired with ld.swzl and account for registers/shared traffic.
4. **Choose tiling by workload.** H800 V64 improved the low-head case but
   V128 remained useful for larger grids. PPU already has value slicing and
   eight-warp delivery; these are not new transfers. Its earlier V16 candidate
   doubled main input traffic and did not deliver a meaningful win. Sweep
   ownership/resources together, not grid size alone or the H800 winner.

Gate-coefficient reuse is only the first, low-risk transfer. The uploaded
PPU gate-cache capture (source3dd407f, g=-1, B1/S2048/Hk16/Hv32) has complete
kernel sums191.18765 -> 186.11471us; FLA225.99588us. The changed state accounts
for92.81294 -> 88.50000us, while solve52.23882us and output42.87765us remain.
This single capture does not admit broad speed claims or a default selector.
Against that FLA capture,1.5x requires150.66392us, another35.45079us reduction.

## Do not copy

- WGMMA asynchronous O2/KV overlap and per-warpgroup register redistribution:
  the same hardware contract is not available on PPU1.0.
- Hopper TMA or barrier instructions through compatibility emulation.
- H800 FP16 inverse over PPU's FP32/three-product-TF32 inverse without admitting
  a changed numerical algorithm.
- Static instruction savings as a speed verdict: the H800 fast-exp2 and some
  synchronization-removal experiments did not produce retained timing wins.

Evaluate one mechanism at a time, then compose admitted winners. Report every
kernel in the complete forward, native instructions/resources, and unchanged
numerical/replay gates. Do not restart H800 to validate a PPU-only transfer.
