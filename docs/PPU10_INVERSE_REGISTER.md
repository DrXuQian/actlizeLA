# PPU1.0 inverse register retention

Local admission:2026-09-28. Device correctness and performance: **NOT_RUN**.
Parent: `2bde8331b05471293dd834cf51d57ac7e21fd496`.
Opt-in: `gdn_chunk_residual(..., delivery="inverse-register")`.
Same-binary performance control: `full-chunk`; scalar residual remains the
RAW-BIT numeric anchor. Public/default routing is unchanged.

## Exactly one delivery seam

The off-diagonal inverse merge previously materialized each warp's FP32
`acc[8]` in `sm.temp[warp][16*16]`, then read it as the next TF32 B operand.
All six lower off-diagonal blocks have one warp owner. No other warp reads
this temporary. The candidate gathers the same FP32 values from that owner's
registers; inter-warp inverse publication and its CTA barriers remain intact.

The native TF32 C/B traits determine the owner, not an assumed NVIDIA layout.
The source slot depends on the **destination** lane: shuffle both candidate
slots first, then select. Selecting before the shuffle silently takes the
source lane's slot. L038 detects that exact mistake.

This is not Hopper's FP16 inverse. Keep both K8 halves and the ordered
`Alo*Bhi`, `Ahi*Blo`, `Ahi*Bhi` TF32 products, FP32 diagonal arithmetic,
BF16 inverse publication, workspace allocation and all four call stages.
Full C64 uses the original full-chunk state; tails use the original generic
state, but **both execute the new solve**. Tails cannot escape its admission
through a whole-call fallback. State/output kernel symbols are reused.

Per chunk, six1024B private temporaries formerly caused6144B stores plus
6144B reads. The candidate removes that logical shared-memory round trip
and adds96warp shuffle instructions per chunk. For the S2048/Hv32 workload,
that is6MiB written +6MiB read removed, with98304shuffle issues added.
These are structural counts, not measured TSM transactions or wall-time
savings. Shared capacity stays49664B: the scratch already aliases the larger
key tile, so this does **not** claim a shared-capacity/occupancy improvement.

## Local evidence

SDK2.1.1, native PPU1.0, actlize423253c. Actual generated body and linked
Python binding; no PPU execution:

| Native compile property | Static-solve control | Register candidate |
|---|---:|---:|
| Static instruction sites |1472|1468|
| Registers/thread |84|76|
| Stack bytes |0|0|
| Scalar FP32 TSM store sites |42|34|
| Scalar FP32 TSM load sites |65|57|
| Indexed warp shuffle sites |0|16|
| BF16 / TF32 MMA sites |8 /12|8 /12|
| TF32 conversion sites |64|64|
| CTA barrier sites |5|5|

All43 retained native instruction/operand sequences and resource records
are identical to the same-SDK parent. The library now has44device bodies.
The native check also binds high/residual operand reuse in every three-MMA
sequence; successful compilation alone is not the criterion.

L038 invokes the shipping mapper against independently indexed actlize C/B
traits:64extents ×8bit patterns ×6off-diagonal owners,3072contexts,
786432FP32 words and1572864TF32 high/residual inputs. Coverage is exact-once.
Wrong slot, destination/source preselection, wrong K half and inserted BF16
rounding all fail; omitting one block fails the fixed coverage denominator.
Source/native/linked-entry plants cover arithmetic, lifetime, aliasing,
tail resource admission and stale dispatch. All259CPU tests and29CTest
cases pass. Four NVIDIA-only device suites are explicitly SKIP, not PASS;
the PPU device gate remains NOT_RUN. Whole-library native gate:76negative
controls including old-body/resource changes, all expected red.

Files: `wy_inverse_register.cuh` holds the owner/gather/three-product helper;
`gdn_wy_inverse_register_ppu.cu` holds the isolated solve and composition.
The only old kernel-TU edit exposes host wrappers for the existing generic
state, with no device-body modification. CMake, explicit Python binding,
local checks and the one-run handoff complete the change.

## Fixed device decision and command

Run without other work on the selected PPU:

```bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_inverse_register_box.sh
```

Uses the configured PPU SDK and `/sim/eec/shared/junfu.qx/asight/bin/acu`.
Optional `PPU_SDK=/path/to/PPU_SDK`; explicit SDK overrides inherited root.
Build once, run retained36cases ×8, then256extent/gate/initial-state cases
×8; both variants must be RAW-BIT equal to residual and pass the unchanged
2% independent recurrence criterion. Then same-fixture FLA admission precedes
ACU for S2048/B1/Hk16/Hv32/D128/C64, initial=None, g=-1/-0.1.

Metric: sum **all** actual kernels in each complete call, including any FLA
fill helpers (expected4/4/7 per cell). No Python/API speed comparison and no
kernel-name/launch-count filtering. SHA, compiler/dependencies, binary hashes,
device identity, raw reports, native code and admission receipts are retained.
Only upload the final printed `inverse-register.tar.gz`; two captures are
inside that one archive. Incomplete/failed admission cannot produce a complete
outer tar. Existing reports are not overwritten by repacking.

Lower complete-call time at both gates -> retain for repeated confirmation.
Mixed/slower -> retain the control and record the register-delivery tradeoff.
A lower register count, fewer shared bytes or lower solve time alone does not
promote the candidate. Unchanged-stage timing variation is reported separately.
Neither result changes default routing or the numerical contract. The old
1.5x FLA target is suspended; this closes a migration applicability experiment.
