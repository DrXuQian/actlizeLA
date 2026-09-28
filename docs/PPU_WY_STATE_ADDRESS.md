# WY state: independent address and row-gate experiments

## Scope and controls

This follows the [verified mask48 ACU diagnosis](PPU_WY_STATE_OUTPUT_ACU_20260922.md):
state199.598 us versus FLA90.024 us, with equal BF16 MMA count, store footprint
and approximately7.1 active warps/CU. Extra signed coordinate decomposition
and repeated gate evaluation are measured instruction work, not an inferred
microsecond breakdown.

Only the tiled **state** changes. Prepare, output, the original strong/reset
route, default scalar WY, public C ABI, workspace layout, FP32 state,
BF16 boundaries and per-output reduction order stay unchanged. No promotion.

| Delivery name | Mask | State address change | State row-factor reuse |
|---|---:|---|---|
| `tiled-state-output` | 48 | no; retained control | no |
| `tiled-all` | 56 | no; retained incumbent | no |
| `tiled-state-output-address` | 112 | yes | no |
| `tiled-state-output-gates` | 176 | no | yes |
| `tiled-state-output-both` | 240 | yes | yes |

Addressing keeps exactly the same16-byte vector ownership, shared swizzle,
source strides, tail zfill and publication bytes. Nonnegative coordinates
and compile-time per-thread offsets replace the rolled signed divide/remainder
loops in W/K/U staging and H/Vnew/final publication. No offline repacking.

Row reuse evaluates the **same `expf(last - g[row])`** once per unique owned
row, reusing it across four fragment columns. It does not replace expf with
another approximation, change the log domain, narrow state precision or
reassociate an accumulation. Four row evaluations plus decay replace sixteen
per-slot evaluations plus decay.

The three candidate kernels share one template in
`csrc/gdn_chunk/gdn_wy_state_ab_ppu.cu`. The admitted state body intentionally
remains verbatim in `gdn_wy_tiles_ppu.cu`. An initial attempt to factor that
control through an inline helper changed its generated instructions; that
attempt was rejected before device testing. Preserving this frozen control
is the reason for the separate candidate TU, not two competing layout truths.

`wy_contract.hpp::visit_state_options` is the shared typed selector used by
both resource configuration and actual launch. Address/reuse bits require
tiled-state16; they cannot silently act on scalar/packed state. The original
27 valid base masks remain valid with identical meanings;27 option-bearing
combinations are added. Invalid or conflicting bits fail closed.

## Local admission

PPU SDK2.1.1-a5c56e, real hgcc compilation and linked `libgdn_wy_ppu.so`:

| State body | Registers | Stack bytes | Static native instructions | Static cp.async sites | Static exponent sites | Static BF16 MMA |
|---|---:|---:|---:|---:|---:|---:|
| Retained tiled control | 232 | 0 | 2153 | 3 | 17 | 32 |
| Address only | 238 | 0 | 2473 | 18 | 17 | 32 |
| Gates only | 234 | 0 | 2012 | 3 | 5 | 32 |
| Both | 242 | 0 | 2300 | 18 | 5 | 32 |

These are **static-body counts**, not dynamic instructions or latency.
The3 old copy sites were rolled8/8/2 times;18 explicit sites perform the
same transfers, not6x traffic. Straight-line footprint and register count
increase on the address arm, so a speedup must still be measured. All9 old
native instruction **and operand sequences** match a same-SDK parent build
exactly. Kernel symbols/resources,12-image denominator, cross-TU launch links,
MMA counts, vector operations and actual option-dependent bodies are audited.

Local checks completed:

- 6/6 CTests, including unchanged native ownership/reduction-order gates.
- New exhaustive gate:18,432 coordinates,1,406,720 vector transactions across
  all valid-row counts and five strides (including a64-bit stride),131,072
  gate reuse values. Both original CuTe layout and an independent hardware
  cube formula anchor the address check; native MMA traits anchor the row map.
- Typed launch selector:512 option values, exactly3 experimental selections;
  complete delivery census:512 values,54 valid. Six negatives cover wrong
  swizzle bit, stride units, missing vectors, tail overread, wrong row reuse
  and an ignored option bit. Each is red as required.
- 57 Python contracts:18 WY,24 ACU,8 FLA,7 hgcc. Real API mask forwarding,
  eight-role denominator/balanced order, combined-versus-single verdicts,
  sub-2%-but-not-raw-equal output rejection, and cross-candidate ACU binding
  are checked. Losing/overlapping envelopes remain losing/UNRESOLVED.
- CPU WY algebra45 cases and five planted defects pass the unchanged gate.
- All12 SDK device images link;12 binary negative checks fail as required,
  including a changed control, omitted candidate and missing cross-TU link.

**Local environment limit:** the installed private test torch is2.9 CPU on
Python3.10. It lacks the generated `c10/cuda/impl/cuda_cmake_macros.h` and CUDA
torch libraries. The unchanged PyTorch binding cannot be compiled/linked in
that environment: this check is **SKIP/environment**, not PASS. No header
stubs or fake runtime were inserted. The box script builds the full extension
against its installed PPU PyTorch before any device admission. Native device
library compilation/linking above is a distinct, completed check.

The collector's SHA helper now streams SHA256 without Python3.11-only
`file_digest`; a multi-chunk/non-ASCII-path test confirms identical hashes.
This does not change provenance fields or their meaning.

No local device execution, device raw-bit result or performance result is
claimed. Evidence and pre-edit plan:
`/workspace/gdn-wy-state-address-evidence-20260922/`.

## Box command

From the `GDN-QSA-sm80` checkout, on the same physical card as the baseline:

```bash
git pull --ff-only &&
env -u OUT PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 \
  STATE_AB=1 TILE_AB=0 DELIVERY_AB=0 SAMPLES=16 \
  bash tools/run_ppu_wy_fla_box.sh
```

This builds all required objects locally on the box, checks generated code,
then runs the original admission and16 WY cases. Every candidate must match
scalar output **and state bits**, preserve GVA/output-only behavior and inputs,
and repeat stably eight times before timing. A failure stops timing; it is
not reclassified as an unresolved speed result.

Performance workload: B1/S2048/Hk16/Hv32/K128/V128/C64, g=-0.1 and-1.0,
zero initial state, final state returned. Eight arms retain original, scalar,
the previous mask48 pair, all-tiled, the three new candidates and FLA.
Five warmups,16 balanced sequential samples,10 complete calls per sample;
no concurrent arms, trimming or numerical-criterion changes.

The combined arm reports direct comparisons to **both single changes** as
well as mask48/all/original/scalar/FLA. Disjoint observed envelopes remain the
decision rule. A win against scalar alone does not establish a win against
mask48; fewer instructions alone is not a speed verdict. Original still has
BF16 final state versus WY/FLA FP32, and that scope difference remains visible.

The runner prints its `/workspace/gdn-wy-fla-<sha>-<UTC>/` artifact directory.
Read `comparison.log` for per-role medians and paired verdicts;
`comparison.json` retains every raw sample and identity. Codegen, correctness,
source SHA/diff and binary hashes are saved alongside. The existing direct
ACU collector accepts all three new delivery names for a later exact-binary
capture; no recapture or rebuilt candidate is silently substituted.

## Next independent stage experiment

[Prepare/output address ablations](PPU_WY_STAGE_ADDRESS.md) keep the state-both
body frozen and test each other stage separately with `STAGE_AB=1`. They do
not change this STATE_AB inventory or any default route.
