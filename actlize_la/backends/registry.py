"""Source availability, not a claim of installed binaries or device speed.

The JSON is also consumed by CMake/setuptools. Algorithms own their complete
launch sequence and workspace. This module must not import torch or a backend
extension merely to list/reject targets.
"""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _catalog():
    data = json.loads(Path(__file__).with_name("targets.json").read_text())
    if data.get("schema") != 1:
        raise RuntimeError("unsupported GDN backend catalog schema")
    return data


def backend_inventory():
    """Return independent data; implemented != built, admitted or fastest."""
    return deepcopy(_catalog())


def require_backend(name):
    targets = _catalog()["targets"]
    if name not in targets:
        raise ValueError(f"unknown GDN backend {name!r}; choose {tuple(targets)}")
    backend = targets[name]
    if not backend["implemented"]:
        raise RuntimeError(f"GDN backend {name}: {backend['reason']}")
    return backend


def require_implementation(algorithm, backend):
    require_backend(backend)
    algorithms = _catalog()["algorithms"]
    if algorithm not in algorithms:
        raise ValueError(f"unknown GDN algorithm {algorithm!r}; choose {tuple(algorithms)}")
    implementation = algorithms[algorithm]
    if backend not in implementation["targets"]:
        raise RuntimeError(f"GDN algorithm {algorithm} has no {backend} implementation; no fallback")
    return implementation
