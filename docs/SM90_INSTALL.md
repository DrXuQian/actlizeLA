# SM90 installation and invocation

The package is `actlize-la`; the Python module is `actlize_la`. There is no
`gdn_qsa_sm80` alias. Install using the same Python environment that owns
CUDA-enabled PyTorch. CUDA 12.8 was used for the retained H800 measurements.
Use an activated virtual environment if the system Python is externally
managed; the installer does not override the package manager's protection.

```bash
git clone git@github.com:DrXuQian/actlizeLA.git
cd actlizeLA
CUDA_HOME=/usr/local/cuda bash tools/install_sm90.sh
```

The installer runs these two distinct steps:

```bash
GDN_QSA_TARGET=python python -m pip install --no-build-isolation --no-deps -e .
python tools/build_sm90_bundle.py --out build/sm90 --register
```

All three retained configurations are compiled before registration succeeds.
`OUT=/workspace/actlizeLA-sm90` changes the artifact directory.
`PYTHON=/path/to/python` changes the installer interpreter. Compiler discovery
uses `$CUDA_HOME/bin/nvcc`; direct builds also accept `--compiler`.
The SM90 target never initializes the PPU SDK or compiles actlize.

## Loading

```python
from actlize_la import gdn_forward
output, final = gdn_forward(q, k, v, g, beta, initial_state=h0)
```

`gdn_forward` detects CUDA SM90 from the tensor's device and selects using
shape metadata. No `CONFIGURATION`, backend, or extension environment variable
is needed for the normal SM90 call. Selection does not inspect gate values.
The first call loads/verifies the installed bundle; warm it up before timing.
Later calls do not read files, hash binaries, compile, autotune, or synchronize
the GPU for selection.

Loading checks the policy/receipt hashes, all three complete binary SHA256s,
native targets and configuration labels. Mixed source/compiler/dependency
receipts or a missing candidate are errors, not fallback conditions.
Distinct binaries receive distinct import identities, so separately built
configurations can coexist in one process without pybind reusing the first.
Do not mutate a loaded binary in place; use a new build directory.

The installer registers an absolute bundle directory in a local, untracked
installation record. It is not included in wheels. To relocate, copy the
entire bundle (manifest, three `.so` files and their `build.json` receipts),
then set `ACTLIZE_LA_SM90_BUNDLE=/absolute/path/to/bundle`. This is a deployment
override, not a configuration selection. Rebuild for a different
Python/PyTorch/CUDA ABI. A frontend-only wheel requires this separately built
bundle or the installer; it cannot pretend that a kernel is installed.

See the README for tensor dimensions, dtypes and preprocessing boundaries.
`output_final_state=False` returns `(output, None)`. Calls use PyTorch's current
CUDA stream and the binding's existing device/tensor validation.

## Numerical device check (not a speed benchmark)

```bash
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

directory = Path("build/sm90/value64").resolve()
receipt = json.loads((directory / "build.json").read_text())
binary = directory / Path(receipt["extension"]).name
subprocess.run([
    sys.executable, "tools/run_sm90_gdn.py", "--backend", "cuda_sm90",
    "--configuration", "value64", "--extension", str(binary),
    "--out", "/workspace/actlizeLA-sm90-check", "--length", "65",
    "--q-heads", "1", "--v-heads", "2", "--initial",
], check=True)
PY
```

This checks a 64+1 tail, grouped-value heads and nonzero initial state against
the independent CPU recurrence. It is not a performance admission. Full
campaigns must include both gate regimes and multiple shapes; the retained
14-workload H800 results are documented in the closure report.

## Diagnostic overrides only

Normal users should omit `CONFIGURATION`. For a fixed-arm experiment only,
`CONFIGURATION=value64-local-inverse bash tools/install_sm90.sh` builds that
arm in its own directory, without replacing the registered automatic bundle.
Load it explicitly using `load_sm90("build/sm90-value64-local-inverse")`.
`control` remains the original comparison arm, not a normal candidate.

`load_sm90()` (no argument) returns the installed automatic callable;
`load_sm90("build/sm90")` loads a particular full bundle. Its `.select(q, v)`
method returns the chosen configuration, policy ID and measured/default basis
without launching a kernel. The exact policy and evidence scope are in
[SM90_AUTO_DISPATCH.md](SM90_AUTO_DISPATCH.md).
