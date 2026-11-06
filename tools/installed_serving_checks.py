#!/usr/bin/env python3
"""Test-only dispatcher. Production imports MUST resolve to the installed wheel."""

import argparse
import os
import runpy
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase", choices=("host", "serving", "aux", "public_default", "flare"), required=True
    )
    p.add_argument("--target", required=True)
    p.add_argument("--tests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flare", type=Path)
    p.add_argument("--blas", type=Path)
    args = p.parse_args()
    assert sys.flags.isolated, "installed checks require python -I"
    import actlize_la

    root = Path(actlize_la.__file__).resolve().parent
    if not {"site-packages", "dist-packages"}.intersection(root.parts):
        raise RuntimeError(
            f"source checkout selected instead of installed wheel: {root}"
        )
    info = actlize_la.get_library_info(args.target)
    library = info["library_path"]
    # Only copied tests/oracles live here; never append the actlizeLA checkout.
    sys.path.insert(0, str(args.tests.resolve()))
    if (args.tests / "actlize_la").exists():
        raise RuntimeError("test directory must not contain a production package")
    from serving_qualification import FLARE_TESTS, HOST_TESTS

    if args.phase == "public_default":
        if args.target != "ppu10":
            raise ValueError("installed public residual bridge is PPU1.0-only")
        entry = args.tests / "test_ppu_default_device.py"
        sys.argv = [str(entry), "--installed", "--results", str(args.output)]
        runpy.run_path(str(entry), run_name="__main__")
        return 0
    if args.phase in ("serving", "aux"):
        entry = args.tests / (
            "serving_device.py" if args.phase == "serving" else "aux_device.py"
        )
        sys.argv = [
            str(entry),
            "--library",
            library,
            "--target",
            args.target,
            "--results",
            str(args.output),
        ]
        runpy.run_path(str(entry), run_name="__main__")
        return 0
    import pytest

    # Do not let ambient -k/--lf options silently shrink the admission suite.
    os.environ.pop("PYTEST_ADDOPTS", None)

    files = [args.tests / (name + ".py") for name in HOST_TESTS]
    os.environ["ALA_SERVING_TEST_LIBRARY"] = library
    if args.phase == "flare":
        if not args.flare or not args.blas or not args.blas.is_file():
            raise RuntimeError(
                "Flare component gates require the existing Flare tree and platform BLAS"
            )
        # Flare supplies the actual framework under test, not an alternate copy
        # of our production Python. Its gates receive discovered paths only.
        sys.path.insert(0, str(args.flare.resolve()))
        from quactlize import get_library_path as quactlize_library

        os.environ.update(
            FLARE_RUN_CUDA_TESTS="1",
            FLARE_ACTLIZE_TARGET=args.target,
            FLARE_ACTLIZE_LIBRARY=library,
            FLARE_QUACTLIZE_HELPERS_LIBRARY=str(quactlize_library(target=args.target)),
            FLARE_TEST_BLAS_LIBRARY=str(args.blas.resolve()),
            FLARE_TEST_BLAS_API="cublas",
        )
        files = [
            args.flare / "tests/components" / (name + ".py") for name in FLARE_TESTS
        ]

    class InstalledOnly:
        def pytest_sessionfinish(self, session, exitstatus):
            for name, mod in tuple(sys.modules.items()):
                if name == "actlize_la" or name.startswith("actlize_la."):
                    if not Path(mod.__file__).resolve().is_relative_to(root):
                        raise RuntimeError(
                            f"test loaded production module outside wheel: {name}"
                        )

    return pytest.main(
        [
            "-q",
            "-p",
            "no:capture",  # outer runner already records complete logs
            "--import-mode=importlib",
            "--override-ini",
            "addopts=",
            f"--junitxml={args.output}",
            *map(str, files),
        ],
        plugins=[InstalledOnly()],
    )


if __name__ == "__main__":
    sys.exit(main())
