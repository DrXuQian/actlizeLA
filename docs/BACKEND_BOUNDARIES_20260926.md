# Backend boundary refactor: local evidence

Parent: `e2c4e7a` (design checkpoint). Legacy device implementation remains the
`ec6e5d4` static-solve generation. Worktree:
`/workspace/gdn-backend-boundaries-20260926`. Evidence:
`/workspace/gdn-backend-boundaries-evidence-20260926`.

## Changes and exclusions

The mixed target primitive header is split into CUDA SM80 and legacy PPU AIU
files. Algorithm TUs, arithmetic, output ownership, reset decision and launch
geometry are not edited. PPU and CUDA build graphs have independent leaf files,
selected by a shared target catalog. The new optional `gdn_forward` entry selects
a complete existing algorithm; it does not impose a shared intermediate format.

Existing public APIs, environment-variable selection, old default algorithm and
experimental delivery selectors remain. Loading is cached by module/path rather
than a zero-argument function that can ignore a changed environment. An unchanged
selection does not repeat filesystem validation in the timing hot path.

No automatic selection of a different algorithm, new numerical tolerance,
performance optimization, new SM90 kernel or hardware run is included. Original
BF16 state and WY/residual FP32 state remain different explicit numerical
contracts. KDA is not registered as GDN. A new SM90/PPU1.7 implementation must
provide its own graph; no legacy fallback or empty successful build is available.
PPU1.5 retains the actlize policy, but a distinct native target is not newly admitted.

## Validation

| Check | Local result |
|---|---|
| Legacy PPU C++ ownership/layout suite | 23/23 PASS |
| New backend/build/dispatch/loader contracts | 26/26 PASS |
| Existing benchmark/ACU/WY contracts, including new source-bundle check | 96/96 PASS |
| HGGC spelling/target negatives | 7/7 PASS |
| Independent host algebra | 45 WY + 61 residual cases PASS; tolerances unchanged |
| Original control structure | 7 TUs / 305 control expressions and original host dispatch preserved |
| PPU original + WY/residual compile/link | Actual SDK2.1.1 native libraries and both Python binding DSOs linked |
| PPU native preservation | 15 original + 36 WY/residual kernels, all instructions and operands identical; full resource dumps identical |
| CUDA SM80 compile/link | All 6 original device TUs, both parent and candidate, NVCC12.8 |
| CUDA SM80 native preservation | All 15 kernels / 31,200 instruction sites; SASS including encoding/control bits identical, excluding source-file identifier lines only |
| Device execution / performance | NOT_RUN; no new speed claim |

The PPU comparison covers 42,477 original and 67,815 WY/residual static sites.
These are code-preservation counts, not executed-instruction performance data.
Native comparison negatives remove a kernel, change an instruction operand,
reverse instruction order and empty a kernel. All are rejected for each library.

Boundary negatives exercise the real CMake/setup dispatcher, not a host model:
unknown target; incompatible legacy alias; ACOMPUTE10700 routed into PPU1.0;
different target in an existing build directory; a falsely enabled SM90/PPU1.7
with missing or borrowed legacy build entry; unsupported algorithm/target;
dropped initial state; stale loader selection; missing moved header/catalog in
an ACU bundle. These fail explicitly, not as SKIP or silent fallback.

No complete NVIDIA Python-extension device test or PPU1.7 native compile is
claimed. The NVIDIA check is a device-library compile/link and SASS comparison;
the PPU check also links the actual Torch host dispatch/binding. Unimplemented
targets are implementation gaps, not successful environment skips.

## Reproduce the boundary checks

```sh
python tests/test_backend_boundaries.py -v
python -m unittest discover -s tests -p 'test_ppu_*contract.py' -v
python dev/ppu/check_original_structure.py --self-test
python dev/backends/check_native_preserved.py BEFORE.isa AFTER.isa --self-test
```

The evidence directory contains `verify.sh`, `cuda_compile.sh`,
`complete-replay.log`, `cuda-native.diff` (empty), before/after native dumps and
all linked artifacts. The local scripts record exact SDK/Python/dependency paths;
they do not execute a device kernel. Existing box runners are unchanged in
purpose; the ACU collector now archives the moved dependencies and target JSON.

Compiled artifacts before publication (debug/source paths can change a binary
hash even when native code is identical):

```text
libgdn_qsa_ppu.so  96a272d8217f4f9e525e343d9da403c3e45c80aebd62e60c0c9727342c9597ef
libgdn_wy_ppu.so   f3256ecf391111f521c97fcf03f6788f6bedfa6defff45bcf6eec507754b6a19
_gdn_chunk_ppu.so  fd30597ca2b1e000274319c3106e1baaf6df6ef3135617eb8000b8f063316a5f
_gdn_wy_ppu.so     afb21caf48f46e898cffb6d4a6387cb716e1e843e6bf73a9a65b813f648faa3b
libgdn_sm80.so     0f04ac94ee13efed907e3826084b4c145b662a836c900e5f9ecd2f296f318088
```

Next algorithm work is independent: select a mathematically compatible cuLA
CUDA C++ SM90 entry, preserve its optimized DAG, then map PPU1.7 descriptors,
copies, MMA and synchronization against the supplied CUTLASS fork. The new
boundary does not require porting that algorithm to SM80/legacy AIU.
