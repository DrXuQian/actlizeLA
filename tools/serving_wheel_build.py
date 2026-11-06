"""Setuptools hook for an explicitly prepared native serving bundle.

No compiler or network is invoked here. Ordinary source installs remain the
lazy host frontend; tools/build_serving_wheel.py is the release build entry.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ServingBuildPy(build_py):
    def run(self):
        super().run()
        bundle = Path(os.environ["ACTLIZE_LA_SERVING_BUNDLE"]).resolve(strict=True)
        manifest = json.loads((bundle / "build.json").read_text())
        package = Path(self.build_lib) / "actlize_la"
        target = manifest["default_target"]
        destination = package / "_native" / target
        destination.mkdir(parents=True, exist_ok=False)
        for name, expected in manifest.pop("payloads").items():
            source = bundle / name
            if source.is_symlink() or sha(source) != expected:
                raise RuntimeError(f"native build payload changed: {name}")
            shutil.copy2(source, destination / name)
        manifest["resources"] = {
            p.relative_to(package).as_posix(): sha(p)
            for p in sorted(package.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
        metadata = package / "_native" / "manifest.json"
        metadata.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        (package / "_native" / "wheel.json").write_text(
            json.dumps(dict(schema=1, manifest_sha256=sha(metadata)), sort_keys=True)
            + "\n"
        )


class ServingWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        # Serving uses a C ABI, not CPython or libtorch. The platform tag does
        # NOT select a PPU generation: the artifact channel and manifest do.
        return "py3", "none", super().get_tag()[2]
