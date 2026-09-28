# SM90 installation and invocation

The package is `actlize-la`; the Python module is `actlize_la`. There is no
`gdn_qsa_sm80` alias. Install using the same Python environment that owns
CUDA-enabled PyTorch. CUDA 12.8 was used for the retained H800 measurements.

```bash
git clone git@github.com:DrXuQian/actlizeLA.git
cd actlizeLA
CUDA_HOME=/usr/local/cuda CONFIGURATION=value64 bash tools/install_sm90.sh
```

The installer runs these two distinct steps:

```bash
GDN_QSA_TARGET=python python -m pip install --no-build-isolation --no-deps -e .
python tools/build_gdn_sm90.py --target cuda_sm90 \
  --configuration value64 --out build/sm90-value64
```

`OUT=/workspace/actlizeLA-sm90-value64` changes the artifact directory.
`PYTHON=/path/to/python` changes the installer interpreter. Compiler discovery
uses `$CUDA_HOME/bin/nvcc`; direct builds also accept `--compiler`.
The SM90 target never initializes the PPU SDK or compiles actlize.

## Loading

```python
from actlize_la import load_sm90
forward = load_sm90("build/sm90-value64")
output, final = forward(q, k, v, g, beta, initial_state=h0)
```

`load_sm90` checks `build.json`, the complete binary SHA256, native CUDA target
and configuration once. Move the `.so` and `build.json` together if relocating
a build. Rebuild for a different Python/PyTorch/CUDA ABI. It does not choose a
configuration by inspecting gates or silently fall back to another backend.
Do not mutate a loaded binary in place; use a new build directory.

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

directory = Path("build/sm90-value64").resolve()
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

## Other configurations

Use `CONFIGURATION=value64-local-inverse` or `value128-paired` with the same
installer. They use distinct default output directories. `control` is the
original comparison arm. Configuration-specific advantages are measured on
H800, not guaranteed on every SM90 system or every sequence/head count.
