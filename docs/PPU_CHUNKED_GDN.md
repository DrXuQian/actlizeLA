# PPU chunked GDN (pure CUDA C++ / CUTLASS)

## Scope and source authorities

This implementation is a native C++ CUTLASS kernel.  It does not compile or
call Triton or CuTeDSL.

- The fused-kernel boundary, one-CTA recurrent-state ownership and tile
  scheduler follow the Apache-2.0 C++ SM90 KDA implementation in
  `inclusionAI/cuLA` at commit `4cc51c5eff79761744ae630536c808a3f8039a0f`.
- GDN semantics follow the MIT-licensed token recurrence and chunk/WY
  decomposition in `fla-org/flash-linear-attention` at commit
  `033b19e81239b13971a410e55dd6c178d430b9d4`.
- PPU transport and MMA use actlize's PPU0010 BF16 AIU load and
  `m16n16k16` FP32-accumulating atom.  NVIDIA TMA, WGMMA, warp groups and
  register reconfiguration are not carried across.

cuLA's current tree does not contain a completed C++ GDN kernel; its SM90 KDA
is the structural reference, not a claim that the GDN code was copied intact.

## Version-one contract

The first specialization is fixed-length forward inference with:

- chunk size 64;
- key and value head dimensions 128;
- BF16 Q/K/V/output and FP32 recurrent state;
- grouped value attention where `num_v_heads` is divisible by
  `num_qk_heads`;
- two independent BV64 CTAs owning disjoint recurrence columns for one
  `(sequence, value-head)`;
- chunk-local cumulative log2 decay and post-sigmoid beta supplied by the
  caller.

The public tensor/state layout is recorded in
`quactlize_ppu_linear_attention.h`.  Variable sequence lengths are represented
in the internal ABI but fail closed in v1; a non-null `cu_seqlens` is never
silently ignored.  The same admission path rejects a Q/K base that is not
16-byte aligned and rejects a sequence/grid product that cannot be represented
without signed overflow.  The public header also states the in-place and
no-overlap contract: exact V-to-output and initial-to-final aliasing are allowed;
other overlap is not.

## Chunk algebra

For a chunk with cumulative log2 decay `gamma`, the implementation constructs

```
B[i,j] = 1[i>j] beta[i] exp2(gamma[i]-gamma[j]) dot(K[i],K[j])
T      = inverse(I+B)
U      = T diag(beta) V
W      = T diag(beta exp2(gamma)) K
R      = U - W H_old
```

then evaluates inter-chunk and causal intra-chunk output terms and updates the
state.  Invalid tail rows form an identity extension: they do not advance the
decay and cannot contribute Q/K/V data.

`l203_chunked_gdn_oracle.cpp` compares this decomposition against an
independent token recurrence.  It covers chunk sizes 16/32/64, both zero and
nonzero initial states, every declared tail length, and one full
`C64/K128/V128` production instance.  The gate also proves the recursive
16-to-32-to-64 unit-lower inverse and requires four semantic faults to turn
red.

## PPU dataflow

Global Q/K operands use actlize's `CollectiveBuilder` path so AIU descriptor,
`partition_S`, `retile_D`, and the four-register PPU MMA operand ABI have one
authority. The original repository's raw-atom probe was deliberately left out
of this standalone runtime; its hidden trait-registration side effect is
replaced by explicit PPU0010 operation and trait dependencies.
The L206 oracle instantiates that exact `__HGGCCC__` builder type and exhausts
all 128 thread slices of its real `TiledMma`: the 4096 accumulator slots own
the 64x64 destination exactly once.  A 64-thread plant leaves 2048 holes and a
wrong destination-stride plant creates 2048 holes plus 2048 duplicate owners.

Generated matrices bypass the unavailable register-to-swizzled-TSM store.
They are gathered directly into the production `TiledMma` register fragments:
the coordinate tensor selects the logical source and CuTe's compact fragment
layout selects the physical register.  Six product kinds contribute 640
`m16n16k16` BF16 MMA instructions per BV64 work tile, in addition to 256 QK/KK
instructions.  Legacy v1 repeats both sets for two V tiles: 1,792 BF16 and 80
TF32 MMA per logical head/chunk.  Two-stage v2 prepares QK/KK/inverse/W once
and runs only the 512 V-dependent BF16 MMA per BV64 recurrence tile: 1,408 BF16
and 40 TF32 MMA per logical head/chunk.  L207 anchors the
logical A/B/C coordinates against the public PPU0010 atom formula.  L208 then
anchors the actual compact physical-register maps for BF16 C64 and both TF32
inverse tiles; rotating one physical slot or transposing B must turn red.

The unit-lower inverse retains four direct 16x16 FP32 triangular solves, then
uses the block identity `-D^-1 C A^-1` at 16-to-32 and 32-to-64.  Its six block
products are 40 `m16n16k8` TF32-input/FP32-accumulate MMA instructions.  The
base solve owns a complete RHS column per thread, so row-to-row dependencies
are thread-local rather than CTA barriers; the complete inverse now has eight
required CTA barriers per chunk.  TF32 is an explicit numerical seam, not a
claim of all-FP32 bit identity: L209 requires an exact dyadic fixture to remain
raw-bit equal and separately bounds a non-TF32-exact fixture's inverse
residual.

One 128x64 FP32 state slice stays CTA-local across chunks.  Phase storage is
unioned; the checked peak is 107,008 bytes (32 KiB state, 512 bytes of gates,
and a 72 KiB phase arena), below the repository's 256-KiB PPU block budget.

The v2 prepare kernel emits one 32-KiB BF16 A/W/P record per
`(sequence,V-head,chunk)` into caller-owned workspace.  Its recurrence kernel
loads that seam and does not execute QK, KK, inverse, or W.  On the published
Qwen3.5-35B-A3B `B1,T2048,Hqk16,Hv32,K128,V128,C64` shape this is a 1,024-CTA
prepare launch, a 64-CTA recurrence launch, and exactly 32 MiB of workspace.

## Deliberate limits and next measurements

The local gates establish algebra, scheduler ownership, tail handling and
generated-code reachability.  On a local RTX 5090, the same scalar collective
body is also executed with test-only global scratch (the 107,008-byte PPU
shared ledger exceeds sm_120's per-block limit): both zero and nonzero initial
state are raw-bit equal to an exact token-recurrence fixture across a 64+1
tail.  A second fixture makes the WY path nontrivial while remaining
constructively exact: strict-lower, inverse off-diagonal and causal
off-diagonal each contain exactly 64 nonzero entries, and both zero/nonzero
initial-state executions remain raw-bit equal.  This is a correctness check,
not a CUDA performance proxy.

A PPU box remains necessary for the hardware-only facts: the actual AIU opcode
path, 107,008-byte shared-memory admission, registers, numerical agreement and
timing.  v2 preregisters 1,408 BF16 `m16n16k16` plus 40 TF32 `m16n16k8`
instructions per logical head/chunk, with zero spills.  No PPU performance
claim is made from the host oracle or CUDA arm.

After the all-AIU device verdict, the next additive tactics are variable-length
scheduling, raw-gate preprocessing, and alternate value-tile/CTA sizes.  They
must retain the v1 admission and oracle rather than weakening it.
