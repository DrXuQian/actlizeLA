"""Lazy native loading, keyed by the selected target module and resolved path.

No Torch/CUDA initialization, guessed hardware target or architecture fallback.
This is loading only: each algorithm's existing wrapper still owns its inputs,
workspace and launches. A changed environment must not borrow an old cached DSO.
"""
from functools import lru_cache
import hashlib
from importlib import import_module, util
import os
from pathlib import Path
import re


_loaded_images = {}
_PPU_DEVICE_LIBRARIES = ("libgdn_qsa_ppu.so", "libgdn_wy_ppu.so")


def _file_image(path):
    file = Path(path).resolve(strict=True)
    before = file.stat()
    with file.open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = file.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"native image changed while hashing: {file}")
    return dict(path=str(file), sha256=digest.hexdigest(),
                file_identity=list(identity(after)))


def _mapped_ppu_libraries():
    """Read actual mapped DSOs, not an adjacent file or LD_LIBRARY_PATH guess."""
    paths = set()
    for row in Path("/proc/self/maps").read_text().splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[5])
        name = Path(path.removesuffix(" (deleted)")).name
        if name in _PPU_DEVICE_LIBRARIES:
            if path.endswith(" (deleted)"):
                raise RuntimeError(f"mapped native image was deleted: {path}")
            paths.add(path)
    return sorted(paths)


def _remember_image(module, binding):
    if _file_image(binding["path"]) != binding:
        raise RuntimeError("native binding changed during load")
    devices = [_file_image(path) for path in _mapped_ppu_libraries()]
    _loaded_images[id(module)] = (module, binding, devices)


def loaded_image_info(module, dependency):
    """Explicit preparation-time audit. No hashes or /proc reads on replay."""
    record = _loaded_images.get(id(module))
    if record is None or record[0] is not module:
        raise RuntimeError("native module has no load-time image identity")
    _, binding, devices = record
    selected = [image for image in devices if Path(image["path"]).name == dependency]
    current_paths = [str(Path(p).resolve()) for p in _mapped_ppu_libraries()
                     if Path(p).name == dependency]
    if len(selected) != 1 or current_paths != [selected[0]["path"]]:
        raise RuntimeError(f"require exactly one loaded {dependency}; candidates={current_paths}")
    for image in (binding, *selected):
        if _file_image(image["path"]) != image:
            raise RuntimeError(f"native image changed after load: {image['path']}; restart process")
    return [dict(image) for image in (binding, *selected)]


def _qualified_name(module_name, path, image_sha):
    # Pybind's process-wide module cache uses spec.name, independently of this
    # Python lru_cache. Keep the leaf equal to the compiled PyInit symbol while
    # isolating distinct builds in separate qualified module namespaces.
    identity = hashlib.sha256(f"{module_name}\0{path}\0{image_sha or ''}".encode()).hexdigest()
    return f"actlize_la._native_{identity}.{module_name}"


@lru_cache(maxsize=None)
def _load(module_name, path, image_sha=None):
    if path is None:
        return import_module("._gdn_chunk", "actlize_la")
    binding = _file_image(path)
    spec = util.spec_from_file_location(_qualified_name(module_name, path, image_sha or binding["sha256"]), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GDN extension: {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _remember_image(module, binding)
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


def load_ppu15():
    module = _load("_gdn_residual_ppu15", _explicit_path("GDN_QSA_PPU15_EXTENSION"))
    if not hasattr(module, "target_arch") or module.target_arch() != 150:
        raise RuntimeError("GDN_QSA_PPU15_EXTENSION is not a native PPU1.5 image; no fallback")
    return module


def clear_backend_cache():
    """Testing hook; this does not unload native libraries from the process."""
    _load.cache_clear()
    _resolve_path.cache_clear()
    _loaded_images.clear()
