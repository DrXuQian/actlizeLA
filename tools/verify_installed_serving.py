#!/usr/bin/env python3
"""Qualify a NON-editable serving wheel outside its source/build directories.

Run with python -I. Discovery is guarded against Torch/import, subprocess and
dlopen side effects. Native checks below are preparation-time, not device tests.
"""

import argparse
import importlib.abc
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "triton"):
            raise AssertionError(f"file discovery imported {fullname}")


def forbidden(*args, **kwargs):
    raise AssertionError("discovery performed native/process/device work")


def module_locations(package_root):
    for name, module in tuple(sys.modules.items()):
        if name == "actlize_la" or name.startswith("actlize_la."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(package_root):
                raise AssertionError(
                    f"production module escaped installed package: {name}: {path}"
                )


def check_discovery(target):
    hook = NoTorch()
    sys.meta_path.insert(0, hook)
    try:
        with (
            patch("ctypes.CDLL", forbidden),
            patch("subprocess.Popen", forbidden),
            patch("os.system", forbidden),
        ):
            import actlize_la

            info = actlize_la.get_library_info(target)
            path = actlize_la.get_library_path(target)
            assert isinstance(path, Path) and path.is_absolute() and path.is_file()
            assert str(path) == info["library_path"]
            assert actlize_la.get_library_path() == path
            json.dumps(info)
            assert "torch" not in sys.modules
            assert "triton" not in sys.modules
    finally:
        sys.meta_path.remove(hook)
    root = Path(actlize_la.__file__).resolve().parent
    if not any(part in ("site-packages", "dist-packages") for part in root.parts):
        raise AssertionError(f"not an installed wheel: {root}")
    module_locations(root)
    return root, info


def negatives(root, info, out):
    from actlize_la._library import LibraryError, _inspect

    results = []
    native = json.loads(Path(info["manifest_path"]).read_text())
    target = info["target"]
    cases = {
        "missing-dso": native["targets"][target]["library"],
        "missing-private-resource": native["targets"][target]["native_graph_example"],
        "changed-binding": "serving/gdn_transaction.py",
        "changed-manifest": "_native/manifest.json",
        "extra-shadow-module": "serving/unregistered.py",
        "missing-header": f"_native/{target}/aux_v1.h",
    }
    for plant, resource in cases.items():
        copy = out / plant / "actlize_la"
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__"))
        path = copy / resource
        if plant.startswith("missing"):
            path.rename(path.with_name(path.name + ".removed"))
        else:
            with path.open("a") as stream:
                stream.write("\n# planted corruption\n")
        try:
            _inspect(copy, target)
        except LibraryError as exc:
            results.append(dict(plant=plant, status="EXPECTED_RED", reason=str(exc)))
        else:
            raise AssertionError(f"false green: {plant}")
    for requested in ("sm90", "ppu15" if target == "ppu10" else "ppu10"):
        try:
            _inspect(root, requested)
        except LibraryError as exc:
            results.append(
                dict(
                    plant="unsupported-target",
                    requested=requested,
                    status="EXPECTED_RED",
                    reason=str(exc),
                )
            )
        else:
            raise AssertionError("wrong target fell back to packaged binary")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--out", type=Path, required=True, help="new verification directory")
    p.add_argument("--discovery-only", action="store_true")
    p.add_argument(
        "--old-abi", type=Path, help="optional independently built ABI-negative DSO"
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="FAIL", device="NOT_RUN", python=sys.version, executable=sys.executable
    )
    try:
        assert sys.flags.isolated, (
            "use python -I; PYTHONPATH must not select the implementation"
        )
        root, info = check_discovery(args.target)
        report.update(
            identity=info,
            package_root=str(root),
            discovery="NO_TORCH_NO_DLOPEN_NO_SUBPROCESS/PASS",
        )
        report["negatives"] = negatives(root, info, args.out)
        if not args.discovery_only:
            import torch

            # Serving import may register custom ops but must not load a device
            # library, initialize CUDA or run compiler discovery.
            with (
                patch("ctypes.CDLL", forbidden),
                patch("subprocess.Popen", forbidden),
                patch("torch.cuda._lazy_init", forbidden),
                patch("torch.cuda.init", forbidden),
            ):
                import actlize_la.serving as serving
                from actlize_la.serving import flare

                for name in serving.__all__:
                    getattr(serving, name)
                for name in (
                    "bind_gdn",
                    "bind_gdn_commit",
                    "bind_qsa",
                    "bind_shortconv",
                    "bind_shortconv_history",
                    "bind_gated_norm",
                    "bind_ple_hash",
                    "auxiliary_rows",
                ):
                    getattr(flare, name)
            library = serving.NativeLibrary(target=args.target)
            explicit = serving.NativeLibrary(info["library_path"], target=args.target)
            assert library.abi == info["abi"] == explicit.abi
            report.update(
                torch=torch.__version__,
                native_abi=library.abi,
                serving_import="NO_DEVICE_OR_NATIVE_LOAD/PASS",
            )
            wrong = "ppu15" if args.target == "ppu10" else "ppu10"
            try:
                serving.NativeLibrary(info["library_path"], target=wrong)
            except ValueError as exc:
                report["negatives"].append(
                    dict(
                        plant="real-dso-wrong-target",
                        status="EXPECTED_RED",
                        reason=str(exc),
                    )
                )
            else:
                raise AssertionError("actual DSO accepted wrong target")
            if args.old_abi:
                try:
                    serving.NativeLibrary(args.old_abi, target=args.target)
                except ValueError as exc:
                    report["negatives"].append(
                        dict(
                            plant="old-native-abi",
                            status="EXPECTED_RED",
                            reason=str(exc),
                        )
                    )
                else:
                    raise AssertionError("old native ABI was accepted")
            assert not torch.cuda.is_initialized(), (
                "host wheel qualification initialized a device"
            )
            module_locations(root)
        report["status"] = (
            "DISCOVERY_PASS_ABI_AND_DEVICE_NOT_RUN"
            if args.discovery_only
            else "HOST_PASS_NOT_DEVICE_ADMISSION"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
