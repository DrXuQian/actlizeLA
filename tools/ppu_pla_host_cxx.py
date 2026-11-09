#!/usr/bin/env python3
"""Host-only PLA compiler adapter for the selected PPU SDK's CUDA API headers.

HGCC device commands stay upstream-identical. Do not use a system NVIDIA
toolkit or globally alter CPATH to make a PPU PyTorch binding compile.
"""
import os
from pathlib import Path
import shlex
import shutil
import sys


def include_dirs(sdk):
    sdk = Path(sdk).resolve(strict=True)
    candidates = [sdk / "CUDA_SDK/include", sdk / "CUDA_SDK/targets/x86_64-linux/include"]
    found = {p.resolve() for p in candidates if (p / "cuda_runtime_api.h").is_file()}
    if len(found) != 1:
        raise ValueError(f"selected PPU SDK needs one CUDA API include tree; found {sorted(map(str, found))}; "
                         f"checked {candidates}")
    include = found.pop()
    for name in ("cuda_runtime_api.h", "cuda.h", "cublas_v2.h", "cusparse.h", "cusolverDn.h"):
        if not (include / name).is_file():
            raise ValueError(f"selected SDK lacks host PyTorch CUDA API header: {include / name}")
    return [include] + ([include / "cccl"] if (include / "cccl").is_dir() else [])


def compiler(env):
    selected = env.get("CXX", "c++")
    path = shutil.which(selected, path=env.get("PATH"))
    if path is None:
        raise ValueError(f"CXX must name an available host C++ executable: {selected!r}")
    if Path(path).resolve() == Path(__file__).resolve():
        raise ValueError("host CXX adapter must not recursively select itself")
    return str(Path(path).resolve())


def command(env, arguments):
    selected = Path(env["ACTLIZE_LA_PLA_HOST_CXX"]).resolve(strict=True)
    if selected == Path(__file__).resolve() or not os.access(selected, os.X_OK):
        raise ValueError("invalid underlying host C++ compiler")
    return [str(selected), *[f"-I{p}" for p in include_dirs(env["PPU_SDK"])], *arguments]


def main():
    argv = command(os.environ, sys.argv[1:])
    print("[PLA host CXX] " + shlex.join(argv), flush=True)
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
