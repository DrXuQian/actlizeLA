# actlizeLA source migration and installation validation

The new actlizeLA history starts from source snapshot
`DrXuQian/GDN-QSA-sm80@3dd407fde443648ec67055e201d9721ed3025509`.
Python imports are now `actlize_la`; no old-package alias is installed.
The original destination history is replaced, not merged. Existing clones of
the old actlizeLA should be kept separately and replaced by a fresh clone.
Copyright/license files remain; see NOTICE.

## Scope

- Package/frontend and all in-repository callers renamed together.
- Frontend installation is compiler-free by default. SM80/SM90/PPU native
  builds are explicit; an SM90 installation cannot silently compile SM80.
- `tools/install_sm90.sh` builds an explicit configuration; `load_sm90`
  verifies receipt/target/configuration/hash once, outside the invocation path.
- Three binaries can coexist in one process. Qualified native import names
  include build identity: pybind's own module-name cache must not substitute
  the first configuration for a subsequently loaded one. The old behavior was
  reproduced locally and a name-collision negative was added.
- No GDN device arithmetic, kernel source/header, default algorithm selector,
  numerical tolerance, or measured configuration was changed in migration.
- PPU1.0 remains actlize-based; SM90 remains independent of the PPU SDK.

## Local validation

| Check | Result |
|---|---|
| Python host contracts, independent algebra and negatives |178/178 PASS|
| PPU host ownership/layout CTests |24/24 PASS|
| Actual SM90 CuTe host maps |4/4 PASS|
| `value64`, `value64-local-inverse`, `value128-paired` |3/3 compile/link/import PASS|
| Native instruction/operand/resource identity against retained H800 anchors |12/12 bodies identical;12 planted failures rejected|
| Fresh wheel imported outside the source tree |PASS, all3 configurations coexist|

Native identity hashes (normalized instructions/operands, not ELF hashes):

- `value64`: `a019ff43656dead1daf26953dec73740ef375ff833eae9d4558b9dda75fa9e04`
- `value64-local-inverse`: `15f23d0ca2c85ea576864f86fa6b0a1818e0ba21fccce26d28ce836d93a1d342`
- `value128-paired`: `78a03947a5642672d62e11c695143dadb7a13ba1e5f58d8d76e31cd18abc829b`

Validation used CUDA12.8 and Python3.12/PyTorch2.9+cu128. The pinned vendored
CUTLASS header tree exactly matches the measured H800 dependency. The actlize
submodule remains `423253c00df333ead6fb72ea623d526f24f56b5a`.

No device kernels were launched for this migration; H800 was not restarted.
This is installation/refactor validation, not a new device-performance result.
The retained device campaign remains in `SM90_H800_CLOSURE_20260927.md`.
The further SM90-to-PPU1.0 work is in `SM90_TO_PPU10.md`.
