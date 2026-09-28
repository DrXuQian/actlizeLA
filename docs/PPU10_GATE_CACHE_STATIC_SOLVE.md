# PPU1.0: static diagonal solve on the gate-cache control

Parent: actlizeLA `7616123`. Opt-in candidate
`residual-gate-cache-solve-static`, control `residual-gate-cache`.
This completes the previously unmeasured solve-index experiment on the newer
state control; it is not a new residual algorithm or an SM90 instruction port.
Default dispatch remains unchanged.

Device follow-up: [2026-09-28 ACU result](PPU10_STATIC_SOLVE_ACU_20260928.md).
Both gates improve: complete control→candidate 185.710→180.070us (strong),
187.102→181.194us (weak), approximately1.247xFLA, not the1.5x goal.
Single captures; retain opt-in and confirm before any routing promotion.

## One changed stage

| Stage | Control | Candidate |
|---|---|---|
| Prefix | existing split prefix | same symbol |
| Solve | `gdn_wy_split_solve` | `gdn_wy_split_solve_static` |
| State | `gdn_wy_residual_gate_cache_state` | same symbol |
| Output | `gdn_wy_residual_warps8_hvlayout_output` | same symbol |

Both solve kernels already exist. The host-only composition selects between
them at compile time; all37 device kernels and their instruction/operand
streams stay unchanged. No new matrix layout, precision boundary, workspace,
launch, barrier, or device-side selection branch is added. The inverse lives
in its separate plane, not a future H snapshot. Initial/final state and error
propagation remain the frozen control's contract.

The [static-index solve](PPU_GDN_SOLVE_STATIC_INDEX.md) preserves ascending
FP32 diagonal FMA order and all three off-diagonal TF32 products. On SDK2.1.1
it removes12 indirect register sites and42 diagonal pipe-flush sites; whole
solve1678->1472 static sites,84registers and zero stack in both. These are
native code facts, not microseconds. The old-index body under the static symbol
must still fail its negative control.

## Baseline and preregistered judgment

Priority workload: B1/S2048/Hk16/Hv32/K=V128/C64, g=-1 and -0.1 separately.
Inputs BF16; FP32 gate also remains supported; FP32 recurrence state.
The prior gate-cache capture (GDN source3dd407f, g=-1) had complete ACU sums
186.11471us versus FLA225.99588us, with solve52.23882us, state88.50000us and
output42.87765us. This is a historical single capture, not today's baseline.
The1.5x goal would require150.66392us there: another35.45079us reduction.
Removing diagonal index work alone is not promised to close that entire gap.

Measure **all kernels in each complete forward**, control4/candidate4/FLA7,
using the site ACU in one binary/device/fixture epoch. Report solve and total
times separately; variation in unchanged stages is not credited to solve.
Fewer index/wait instructions without faster complete-call time means
NO OBSERVED SPEED GAIN; retain the control. A faster full call is a candidate
for repeat confirmation, not automatic routing. Keep both gate results,
including any regression. No API-event latency is substituted for kernel sums.

## Local checks and device admission

- Frozen pre-refactor host-body hash protects all non-selection logic.
- Read the real linked x86 host calls: control must reach the old solve,
  candidate the static solve, and both the same state/output. The parser follows
  HGGC's weak-PLT composition helper; a wrapper name alone is not evidence.
- Wrong solve/configure/state/output, inverse alias and wrong Python binding
  are constructive source negatives; retargeting the actual linked static
  call to the old solve must fail too.
- Reuse actual production host proofs:65536solve contexts /1048576values /
 7864320ordered products, all64tails; gate coefficient ownership checks64tails,
 12288writers and393216consumer reads. Retain all their negative controls.
- Compile/link with real PPU1.0 hgcc and compare37/37 native bodies to the
  immutable local parent. No device kernel is added by composition.
- On box: unchanged30-case independent2%oracle, eight RAW-BIT repetitions,
  tail/GVA/nonzero state/variable metadata/output-only. Failure stops profiling.

Local runtime import and device execution are separate gates. This local
host's glibc cannot load the SDK's GLIBC_2.38 runtime; runtime/device status is
SKIP/NOT_RUN, never inferred from successful compilation/linking.

Completed local replay, 2026-09-28:196Python tests and24host CTests PASS;
all61linked-native negative controls PASS. All37native bodies, operands and
the complete resource report are identical to the immutable parent build.
The real linked host graph passes both solve-selection negatives. PPU hgcc
2.1.1-a5c56e builds and links the library and Python extension. The two-gate
script's executable dry-run verifies one build and stops after a failed first
child instead of continuing to weak-gate capture. These are local facts only.
Evidence: [gate_cache_solve_local_20260928.json](../dev/ppu/results/gate_cache_solve_local_20260928.json).

## Box command

From this branch in **actlizeLA**, in the SDK-matched container with the chosen
PPU idle:

```bash
DEVICE=0 JOBS=16 bash tools/run_ppu10_solve_gate_cache_box.sh
```

This builds once, runs the numerical admissions, and captures
gate-cache/static-solve/FLA at strong then weak decay on that same binary.
`PPU_SDK` is optional for a non-default SDK installation. ACU defaults to
`/sim/eec/shared/junfu.qx/asight/bin/acu`. Reports/source/binary identity and
two upload tar files stay under the printed fresh `/workspace/actlizeLA-*`
directory. No PPUProfiler, CSV-paste requirement, or remote compilation by the
agent. A completed capture is not a speed verdict.
