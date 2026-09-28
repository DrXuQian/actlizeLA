"""Lazy native loading, keyed by the selected target module and resolved path.

No Torch/CUDA initialization, guessed hardware target or architecture fallback.
This is loading only: each algorithm's existing wrapper still owns its inputs,
workspace and launches. A changed environment must not borrow an old cached DSO.
"""
from functools import lru_cache
from importlib import import_module, util
import os
from pathlib import Path


@lru_cache(maxsize=None)
def _load(module_name, path):
    if path is None:
        return import_module("._gdn_chunk", "actlize_la")
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GDN extension: {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _resolve_path(variable, path, cwd):
    if not path or not path.strip():
        raise RuntimeError(f"set {variable} to a built extension file, got {path!r}")
    candidate = Path(cwd) / path
    if not candidate.is_file():
        raise RuntimeError(f"set {variable} to a built extension file, got {path!r}")
    return str(candidate.resolve())


def _explicit_path(variable):
    path = os.environ.get(variable)
    # Resolve/check a selection once, not a filesystem stat for every timed call.
    # Relative paths include cwd in the key; chdir must not reuse another DSO.
    cwd = "" if path and os.path.isabs(path) else os.getcwd()
    return _resolve_path(variable, path, cwd)


def load_original():
    if "GDN_QSA_PPU_EXTENSION" not in os.environ:
        return _load("_gdn_chunk", None)
    return _load("_gdn_chunk_ppu", _explicit_path("GDN_QSA_PPU_EXTENSION"))


def load_wy():
    return _load("_gdn_wy_ppu", _explicit_path("GDN_QSA_WY_EXTENSION"))


def clear_backend_cache():
    """Testing hook; this does not unload native libraries from the process."""
    _load.cache_clear()
    _resolve_path.cache_clear()
