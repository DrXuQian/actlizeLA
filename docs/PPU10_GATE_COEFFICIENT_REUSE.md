# First controlled transfer from the H800 campaign

Scope: forward scalar GDN on **PPU1.0**, using actlize AIU.swzl with its
matching native ld.swzl readers. This is not a Hopper instruction port.

## What transfers

H800's scalar gate producer demonstrated a useful structural separation:
compute a coefficient once per token, then let matrix consumers reuse it.
The PPU candidate applies only that idea to the existing eight-warp paired
H/V residual state kernel. It does not adopt WGMMA async, named warpgroup
barriers, TMA, fast exp2, a new inverse algorithm, or a new output layout.

For each chunk row, the unique producer computes exactly the existing
`expf(prefix[row])` and `expf(last - prefix[row])`. The FP32 subtraction,
expf implementation, BF16 rounding boundaries and FP32 state accumulation
stay unchanged. In particular, the relative coefficient is **not**
`exp(last)/exp(prefix)`, which can become 0/0 on strong decay.
The existing INPUTS_READY barrier publishes the coefficients; RETIRE already
protects reuse. There is no added barrier.

The new kernel is explicit `delivery="gate-cache"`; the performance control
is `warps8-hvlayout`. Public default routing is unchanged.
Prepare, prefix, inverse and output call the existing functions/symbols.

## Local evidence (not device admission)

| Item | Control | Candidate |
|---|---:|---:|
| State CTA threads / grid rule | 256 / B*Hv*4 | unchanged |
| Shared bytes | 45,568 | 46,080 |
| Vector registers | 122 | 124 |
| Stack / spills | 0 | 0 |
| Static state instruction positions | 1,194 | 1,166 |
| Static exp2 sites in the native expf lowering | 5 | 2 |
| Logical lane exp evaluations per CTA/chunk | 1,280 | 128 |
| BF16 MMA / AIU / CTA barrier sites | 20 / 4 / 5 | unchanged |

This prices the +512 B shared memory, two additional registers and the
producer's last-prefix global read. Fewer exponential evaluations do not
establish a speedup. There is no new PPU timing result.

- Real hgcc/PPU1.0 compilation, shared-library and Python-extension link.
- Local Python-extension import is SKIP: this host's glibc is older than the
  SDK runtime's required GLIBC_2.38. No import success or device execution is
  claimed; the existing SDK-matched box container must perform admission.
- All 36 old device bodies retain the exact instruction/operand streams.
- Production coefficient helper + actual native MMA C-layout: all 64 tail
  lengths, 12,288 unique writers, 393,216 consumer reads checked.
- Host negatives: wrong relative formula, missing writer, missing tail.
- Source negatives: wrong producer coverage, removed publication barrier,
  wrong last row, reciprocal formula, wrong output dispatch, wrong binding.
- Native negatives: missing MMA, exponential, matching swizzle load, barrier.

## Box handoff

Run from the updated GDN-QSA-sm80 checkout, with no competing device job:

```bash
git pull --ff-only
CANDIDATE=residual-gate-cache GATE=-1.0 JOBS=16 \
  ACU=/sim/eec/shared/junfu.qx/asight/bin/acu \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

The runner builds with the local box SDK, runs independent 2% oracle
admission and eight raw-bit control replays (including weak/strong decay,
tails, GVA, initial/final-state cases), then captures control/candidate/FLA
with the site ACU. Results and upload tar stay under a fresh /workspace path.
For a weak-decay profile repeat with `GATE=-0.1`; correctness is not inferred
from the strong-decay timing run.

Performance verdict remains **the sum of all kernels in a complete forward
in paired ACU captures**, not API time or this state's isolated time.
Report both gates and resource/traffic changes, including regressions.
No device pass means no default-route promotion.
