"""Load the automatic Hopper bundle or an explicitly built diagnostic binary."""
import hashlib
import json
from pathlib import Path

from .backends.loading import _load
from .gdn_sm90_interface import MATH_CONTRACT


class Sm90Forward:
    """A fixed SM90 binary; construction verifies identity once, never per call."""

    def __init__(self, module, configuration, extension):
        self._module = module
        self.configuration = configuration
        self.extension = extension

    def __call__(self, q, k, v, g, beta, initial_state=None, output_final_state=True):
        output, final = self._module.forward(q, k, v, g, beta, initial_state, output_final_state)
        return output, final if output_final_state else None


def load_sm90(build_directory=None):
    """Load the installed auto bundle, a bundle directory, or one fixed build.

    Omit the directory to use the registered installation. A single build
    directory is an explicit diagnostic override, not automatic selection.
    No kernel is launched during loading.
    Q/K normalization, gate preprocessing and the device stream stay with the
    caller. The existing C++ binding validates all tensors at invocation.
    """
    if build_directory is None:
        from .sm90_auto import default_forward
        return default_forward()
    directory = Path(build_directory).resolve()
    if (directory / "bundle.json").is_file():
        from .sm90_auto import load_bundle
        return load_bundle(directory)
    receipt = json.loads((directory / "build.json").read_text())
    if receipt.get("complete") is not True or receipt.get("target") != "cuda_sm90" or receipt.get("mode") != "native":
        raise ValueError("load_sm90 requires a complete native CUDA SM90 build")
    configuration = receipt.get("configuration")
    if configuration not in ("control", "value64", "value64-local-inverse", "value128-paired"):
        raise ValueError("unknown SM90 configuration in build receipt")
    extension_name = receipt.get("extension")
    if not isinstance(extension_name, str) or not extension_name:
        raise ValueError("build receipt has no extension")
    # The receipt may have been moved with its binary to another machine.
    # Do not follow the old absolute path into an unrelated build directory.
    extension = directory / Path(extension_name).name
    if not extension.name.startswith("_gdn_fused_sm90.") or not extension.name.endswith(".so"):
        raise ValueError("unexpected SM90 extension filename")
    if not extension.is_file() or extension.is_symlink():
        raise ValueError("SM90 extension missing or symlinked outside its bundle")
    hasher = hashlib.sha256()
    with extension.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if receipt.get("extension_sha256") != digest:
        raise ValueError("SM90 binary hash differs from build receipt")
    module = _load("_gdn_fused_sm90", str(extension), digest)
    if (getattr(module, "target", None) != "cuda_sm90" or
            getattr(module, "configuration", None) != configuration or
            getattr(module, "math_contract", None) != MATH_CONTRACT):
        raise ValueError("loaded SM90 binary identity differs from build receipt")
    return Sm90Forward(module, configuration, str(extension))
