#!/usr/bin/env python3
"""Build the complete existing serving target and seal a precompiled wheel.

Requires the matching platform SDK, Torch, setuptools, wheel and CMake already
installed. Never installs dependencies or guesses another target. No device run.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(*args, **kwargs):
    print("[serving wheel]", *map(str, args), flush=True)
    return subprocess.check_output(list(map(str, args)), text=True, **kwargs).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    sources = command(
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", cwd=ROOT
    ).split("\0")
    hashes = {
        name: sha(ROOT / name)
        for name in sources
        if name and (ROOT / name).is_file() and not (ROOT / name).is_symlink()
    }
    result = dict(
        source_sha256=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        actlize_revision=command(
            "git", "rev-parse", "HEAD", cwd=ROOT / "third_party/actlize"
        ),
    )
    pinned = command("git", "rev-parse", "HEAD:third_party/actlize", cwd=ROOT)
    if pinned != result["actlize_revision"] or command(
        "git", "status", "--porcelain", cwd=ROOT / "third_party/actlize"
    ):
        raise RuntimeError("actlize checkout differs from the pinned source dependency")
    return result


def elf_identity(path):
    dynamic = command("readelf", "-d", path)
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
    search = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[(.*?)\]", dynamic)
    if search != ["$ORIGIN"] or any("/" in dep for dep in needed):
        raise RuntimeError(
            f"non-relocatable native artifact: {path.name}: {needed}, {search}"
        )
    versions = command("readelf", "--version-info", path)
    return dict(
        needed=needed,
        runpath=search,
        required_symbol_versions=sorted(
            set(re.findall(r"Name: ((?:GLIBC|GLIBCXX|CXXABI)_[\d.]+)", versions))
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("ppu10", "ppu15", "cuda_sm120"), required=True)
    p.add_argument("--sdk", type=Path, help="matching PPU SDK root (required for PPU)")
    p.add_argument(
        "--out", type=Path, required=True, help="NEW build/artifact directory"
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--cmake-arg", action="append", default=[])
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development artifact only, never release-admitted",
    )
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    revision = command("git", "rev-parse", "HEAD", cwd=ROOT)
    status = command(
        "git", "status", "--porcelain", "--untracked-files=normal", cwd=ROOT
    )
    if status and not args.allow_dirty:
        raise RuntimeError(
            "release wheel requires clean source; --allow-dirty is development-only"
        )
    identity = source_identity()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    config = [
        "cmake",
        "-S",
        ROOT / "csrc/serving",
        "-B",
        out / "cmake",
        f"-DALA_TARGET={args.target}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DALA_BUILD_SERVING_EXAMPLE=ON",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF",
    ]
    compatibility = dict(
        platform="linux_x86_64",
        native_interface="C ABI; no libtorch/CPython dependency",
        python=">=3.10",
        torch="platform-provided; requires custom_op and FakeTensor support",
        intended_platform="Python 3.12 / platform Torch 2.9; separate device qualification required",
        bundled_platform_runtime=False,
    )
    if args.target.startswith("ppu"):
        if not args.sdk:
            raise ValueError("PPU build requires --sdk; no implicit runtime fallback")
        sdk = args.sdk.resolve(strict=True)
        compiler = sdk / "bin/hgcc"
        compatibility.update(
            compiler_version=command(compiler, "--version"),
            sdk_release=(sdk / "release.yaml").read_text().strip(),
            sdk_runtime=(sdk / "VERSION.txt").read_text().strip(),
            runtime_policy="use the matching platform SDK/driver, never a wheel-bundled runtime",
        )
        compatibility["sdk_runtime_libraries"] = {
            name: dict(
                sha256=sha(sdk / "lib" / name),
                required_symbol_versions=sorted(
                    set(
                        re.findall(
                            r"Name: ((?:GLIBC|GLIBCXX|CXXABI)_[\d.]+)",
                            command("readelf", "--version-info", sdk / "lib" / name),
                        )
                    )
                ),
            )
            for name in (
                "libhg_wrapper.so",
                "libhggc_wrapper.so",
                "libhggcrt1.so",
                "libhggc.so",
            )
        }
        config.append(f"-DPPU_SDK_ROOT={sdk}")
    else:
        compiler = Path(
            shutil.which(os.environ.get("CUDACXX", "nvcc")) or "nvcc"
        ).resolve(strict=True)
        compatibility.update(
            compiler_version=command(compiler, "--version"),
            runtime_policy="platform CUDA runtime",
        )
        config.append(f"-DCMAKE_CUDA_COMPILER={compiler}")
    compatibility["compiler_sha256"] = sha(compiler)
    config += args.cmake_arg
    with (out / "build.log").open("w") as log:
        for cmd in (
            config,
            [
                "cmake",
                "--build",
                out / "cmake",
                "--target",
                "actlize_la_serving",
                "actlize_la_native_graph",
                "-j",
                str(args.jobs),
            ],
        ):
            subprocess.run(
                list(map(str, cmd)), check=True, stdout=log, stderr=subprocess.STDOUT
            )
    library = out / "cmake/libactlize_la_serving.so"
    graph = out / "cmake/actlize_la_native_graph"
    compatibility["elf"] = {p.name: elf_identity(p) for p in (library, graph)}
    # Explicit preparation-time loading of the ACTUAL built DSO, not an nm-only
    # claim or a hand-authored manifest. No launch/device query is made here.
    sys.path.insert(0, str(ROOT))
    from actlize_la._version import __version__
    from actlize_la.serving import NativeLibrary

    native = NativeLibrary(library, target=args.target)
    bundle = out / "bundle"
    bundle.mkdir()
    payloads = [
        library,
        graph,
        *(
            ROOT / "csrc/serving" / n
            for n in ("abi.h", "qsa.h", "aux_v1.h", "gdn_transaction_v2.h")
        ),
    ]
    for source in payloads:
        shutil.copy2(source, bundle / source.name)
    notices = {
        "LICENSE.actlize": ROOT / "third_party/actlize/LICENSE.txt",
        "NOTICE.actlizeLA": ROOT / "NOTICE",
    }
    for name, source in notices.items():
        shutil.copy2(source, bundle / name)
    if (
        source_identity() != identity
        or command("git", "rev-parse", "HEAD", cwd=ROOT) != revision
    ):
        raise RuntimeError("source changed during native build; discard this artifact")
    manifest = dict(
        schema=1,
        version=__version__,
        source_revision=revision,
        source_dirty=bool(status),
        source_identity=identity,
        default_target=args.target,
        targets={
            args.target: dict(
                library=f"_native/{args.target}/{library.name}",
                native_graph_example=f"_native/{args.target}/{graph.name}",
                abi=native.abi,
                compatibility=compatibility,
            )
        },
        payloads={
            **{path.name: sha(path) for path in payloads},
            **{name: sha(source) for name, source in notices.items()},
        },
    )
    (bundle / "build.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    # SDK-specific artifact channel. Linux tags alone do not encode a PPU ISA.
    label = re.sub("[^a-z0-9]", "", compatibility.get("sdk_release", "cuda").lower())
    tag = "1" + args.target.replace("_", "") + label
    env = dict(
        os.environ, GDN_QSA_TARGET="python", ACTLIZE_LA_SERVING_BUNDLE=str(bundle)
    )
    with (out / "wheel.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "setup.py",
                "egg_info",
                "--egg-base",
                str(out),
                "build",
                "--build-base",
                str(out / "python-build"),
                "bdist_wheel",
                "--bdist-dir",
                str(out / "bdist"),
                "--dist-dir",
                str(out / "dist"),
                "--build-number",
                tag,
            ],
            cwd=ROOT,
            env=env,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    wheels = list((out / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected exactly one new serving wheel")
    if (
        source_identity() != identity
        or command("git", "rev-parse", "HEAD", cwd=ROOT) != revision
    ):
        raise RuntimeError("source changed during wheel build; discard this artifact")
    result = dict(
        wheel=str(wheels[0]),
        sha256=sha(wheels[0]),
        target=args.target,
        source_revision=revision,
        source_dirty=bool(status),
        abi="VERIFIED_ON_ACTUAL_DSO",
        device_numerics="NOT_RUN",
        admission="DEVELOPMENT_ONLY" if status else "INSTALL_TEST_REQUIRED",
        artifact_channel=f"{args.target}/{label}",
    )
    kit = out / "qualification-kit.tar.gz"
    with tarfile.open(kit, "w:gz") as archive:
        for name in (
            "tools/run_serving_wheel_box.sh",
            "tools/qualify_serving_wheel.py",
            "tools/installed_serving_checks.py",
            "tools/verify_installed_serving.py",
            "tests/test_serving_contract.py",
            "tests/test_aux_contract.py",
            "tests/serving_reference.py",
            "tests/aux_reference.py",
            "tests/serving_device.py",
            "tests/aux_device.py",
        ):
            archive.add(ROOT / name, arcname=name)
    result["qualification_kit"] = dict(path=str(kit), sha256=sha(kit))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
