import os
import runpy
from pathlib import Path
import json
from setuptools import setup


HERE = os.path.dirname(os.path.abspath(__file__))
# Reject another target BEFORE importing Torch/compiler discovery. Availability
# is shared with CMake/runtime; unsupported SM90 must never build an SM80 wheel.
_target = os.getenv("GDN_QSA_TARGET", "python")
if _target == "python":
    # Install the frontend without discovering CUDA or compiling an unrelated
    # architecture. Native backends are built explicitly and loaded by identity.
    setup()
else:
    _catalog = json.loads((Path(HERE) / "actlize_la/backends/targets.json").read_text())
    if _catalog.get("schema") != 1:
        raise RuntimeError("unsupported GDN backend catalog schema")
    if _target not in _catalog["targets"]:
        raise RuntimeError(f"unknown GDN target: {_target!r}")
    _backend = _catalog["targets"][_target]
    if not _backend["implemented"]:
        raise RuntimeError(f"GDN target {_target}: {_backend['reason']}")
    if _backend["build"] != "setuptools":
        raise RuntimeError(f"GDN target {_target} uses {_backend['build']}, not CUDA setuptools")

    _entry_name = _backend.get("build_entry")
    if not _entry_name:
        raise RuntimeError(f"GDN target {_target}: no registered build entry")
    _entry = (Path(HERE) / _entry_name).resolve()
    if not _entry.is_relative_to(Path(HERE)) or not _entry.is_file():
        raise RuntimeError(f"GDN target {_target}: missing/outside build entry {_entry}")
    runpy.run_path(str(_entry), init_globals={
        "HERE": HERE, "_backend": _backend, "GDN_SELECTED_TARGET": _target,
    })
