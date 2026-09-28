#!/usr/bin/env python3
"""Collect original/FLA or explicit WY/control/FLA ACU reports in one tar.gz.

No remote commands, installs, device clock changes, model data or full caches.
Failed captures still get a clearly INCOMPLETE diagnostic archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACU = Path("/sim/eec/shared/junfu.qx/asight/bin/acu")
sys.path.insert(0, str(ROOT))
from actlize_la.gdn_wy_interface import DELIVERIES
from actlize_la.gdn_residual_interface import PROFILE_VARIANTS, RESIDUAL_VARIANTS, RESIDUAL_CONTROLS, MATH_CONTRACT, WY_MATH_CONTRACT

RESIDUAL_DELIVERIES = set(RESIDUAL_VARIANTS) - {"residual"}


class CaptureArm(NamedTuple):
    label: str
    role: str
    delivery: str
    directory: Path


def capture_arms(bundle, implementation, delivery, control=None):
    """One FLA capture, optionally preceded by two explicitly selected WY arms."""
    if implementation not in ("original", "wy") or delivery not in PROFILE_VARIANTS:
        raise ValueError("unknown capture implementation or delivery")
    if implementation != "wy" and (delivery != "scalar" or control is not None):
        raise ValueError("WY control/delivery requires --wy-run")
    if delivery == "residual" and control != "state-pipeline":
        raise ValueError("residual requires the registered state-pipeline control")
    if delivery in RESIDUAL_DELIVERIES and control != RESIDUAL_CONTROLS[delivery]:
        raise ValueError(f"residual delivery requires same-geometry control {RESIDUAL_CONTROLS[delivery]}")
    arms = []
    if control is not None:
        if control not in PROFILE_VARIANTS or control == delivery:
            raise ValueError("WY control must be a different supported delivery")
        arms.append(CaptureArm("wy-control", "wy", control, bundle / "wy-control"))
    role = "wy" if implementation == "wy" else "ours"
    return tuple(arms + [CaptureArm(role, role, delivery, bundle),
                         CaptureArm("fla", "fla", delivery, bundle)])


def validate_residual_admission(arm, limit):
    """New arithmetic is explicit and independently checked, never fake RAW-BIT."""
    if (limit != 0.02 or arm.get("math_contract") != MATH_CONTRACT or
            "delivery_mask" not in arm or arm["delivery_mask"] is not None or
            "scalar_raw_bit_equal" not in arm or arm["scalar_raw_bit_equal"] is not None):
        raise ValueError("residual must declare its new rounding contract and non-RAW-BIT scope")
    errors = arm.get("errors", [])
    if len(errors) != 2 or any(type(e) not in (int, float) or not math.isfinite(e) or
                               e < 0 or e >= limit for e in errors):
        raise ValueError("residual lacks independent output/state 2% numerical admission")


def validate_residual_delivery_admission(arm, baseline, limit):
    validate_residual_admission(arm, limit)
    validate_residual_admission(baseline, limit)
    fingerprint = baseline.get("fingerprint")
    if (arm.get("residual_raw_bit_equal") is not True or not fingerprint or
            arm.get("fingerprint") != fingerprint or arm.get("residual_fingerprint") != fingerprint):
        raise ValueError("residual delivery lacks residual-control RAW-BIT admission")


def sha(path):
    # Python >=3.9 is supported by this project; file_digest arrived in3.11.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log, env, *, optional=False, console=True, timeout=None):
    """Save the precise argv and tool output; optional probes never masquerade as PASS."""
    command = [str(x) for x in command]
    log.with_suffix(log.suffix + ".command").write_text(shlex.join(command) + "\n")
    print(f"[GDN ACU bundle] {shlex.join(command)}", flush=True)
    with log.open("w") as stream:
        try:
            if timeout is not None:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
                rc = result.returncode
            else:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as proc:
                    for line in proc.stdout:
                        stream.write(line)
                        stream.flush()
                        if console:
                            print(line, end="", flush=True)
                    rc = proc.wait()
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(f"UNAVAILABLE: {type(exc).__name__}: {exc}\n")
            if not optional:
                raise
            return dict(status="UNAVAILABLE", reason=str(exc))
        if rc:
            stream.write(f"\n{'UNAVAILABLE' if optional else 'FAIL'}: command returncode={rc}\n")
            if not optional:
                raise RuntimeError(f"command failed (rc={rc}); see {log}")
            return dict(status="UNAVAILABLE", returncode=rc)
    return dict(status="COLLECTED", returncode=0)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy contents, never archive a link back into the operator's filesystem.
    shutil.copyfile(source, destination)


def report_file(base):
    candidates = [p for p in (base, Path(str(base) + ".acurep"))
                  if p.is_file() and p.stat().st_size > 0]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one nonempty ACU report, found {candidates}")
    return candidates[0]


def child_command(extension, role, gate, bundle, phase, implementation="original", delivery="scalar", sequence=2048):
    receipt = f"{role}-preflight.json" if phase == "preflight" else f"{role}.json"
    return [sys.executable, "-u", str(ROOT / "benchmarks/profile_ppu_gdn_fla.py"),
            "--extension", str(extension), "--role", role, "--phase", phase, "--gate", str(gate),
            "--implementation", implementation,
            "--wy-delivery", delivery,
            "--receipt", str(bundle / receipt),
            "--sources", str(bundle / "sources/reference"),
            *(["--sequence", str(sequence)] if sequence != 2048 else [])]


def acu_command(acu, report, extension, role, gate, bundle, implementation="original", delivery="scalar", sequence=2048):
    # Same direct CLI pattern as quactlize/tools/run_dense_marlin_m8_acu_box.sh:
    # preflight is a separate process; ACU owns profiling from process start.
    return [str(acu), "-f", "-o", str(report), "--set", "full",
            "--check-exit-code", "yes",
            *child_command(extension, role, gate, bundle, "subject", implementation, delivery, sequence)]


def read_wy_run(directory):
    """Bind a reused WY pair to the completed comparison, not today's checkout."""
    directory = directory.resolve()
    candidates = sorted((directory / "build").glob("_gdn_wy_ppu*.so"))
    if len(candidates) != 1:
        raise ValueError(f"expected one reused WY binding, found {candidates}")
    extension = candidates[0].resolve()
    library = extension.parent / "libgdn_wy_ppu.so"
    comparison = json.loads((directory / "comparison.json").read_text())
    if (comparison.get("protocol") not in ("full-public-api-event-span", "numeric-admission-for-acu")
            or not comparison.get("cases")):
        raise ValueError("WY run lacks complete-API comparison or explicit ACU numeric admission")
    if comparison["protocol"] == "numeric-admission-for-acu":
        if comparison.get("samples") != 0:
            raise ValueError("ACU numeric admission must not claim API timing")
        for case in comparison["cases"]:
            if case.get("timing") != "NOT_RUN" or not case.get("arms"):
                raise ValueError("ACU numeric admission lacks its explicit untimed scope")
            for role, arm in case["arms"].items():
                if arm.get("admitted_repeats") != 8 or arm.get("samples_us") != []:
                    raise ValueError("ACU numeric admission missing eight repeats or claiming timing")
                if role == "wy-residual":
                    validate_residual_admission(arm, comparison.get("limit"))
                elif role.removeprefix("wy-") in RESIDUAL_DELIVERIES:
                    validate_residual_delivery_admission(arm, case["arms"].get("wy-residual", {}), comparison.get("limit"))
                elif role.startswith("wy-") and arm.get("scalar_raw_bit_equal") is not True:
                    raise ValueError("ACU candidate lacks scalar RAW-BIT admission")
    source_sha = (directory / "sha.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("WY run has no complete source SHA")
    recorded = []
    for line in (directory / "binaries.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        recorded.append((Path(filename.lstrip("*")).name, digest))
    binary_hashes = {}
    for path in (extension, library):
        matches = [digest for name, digest in recorded if name == path.name]
        if matches != [sha(path)]:
            raise ValueError(f"reused binary differs from comparison manifest: {path.name}")
        binary_hashes[str(path)] = matches[0]
    matches = [digest for name, digest in comparison.get("binary_sha256", {}).items()
               if Path(name).name == extension.name]
    if matches != [sha(extension)]:
        raise ValueError("WY binding is not the one recorded by comparison.json")
    return extension, comparison, dict(source_sha=source_sha, binary_sha256=binary_hashes,
        source_diff_sha256=sha(directory / "source.diff"),
        source_diff_empty=(directory / "source.diff").stat().st_size == 0,
        comparison_sha256=sha(directory / "comparison.json"),
        binary_manifest_sha256=sha(directory / "binaries.sha256"))


def validate_comparison(comparison, subject, fla):
    """Do not rebind an old median to changed inputs, FLA code or arithmetic."""
    # Older receipts have no shape field and only measured S2048. They cannot
    # admit a newly requested tail shape, even if an arm ignores --sequence.
    cases = [case for case in comparison["cases"] if case.get("g") == subject["gate"] and
             (case["shape"] == subject["shape"] if "shape" in case else subject["shape"].get("S") == 2048)]
    if len(cases) != 1 or cases[0].get("input_sha") != subject["input_sha"]:
        raise ValueError("capture fixture differs from the selected WY comparison case")
    for key, value in (("initial_state", "zero"), ("final_state", True),
                       ("qk_norm", False), ("scale", "1/sqrt(128)"), ("dtype", "bf16")):
        if comparison.get(key) != value:
            raise ValueError(f"unsupported preceding comparison contract: {key}")
    for key in ("torch", "fla"):
        if key not in comparison or comparison[key] != fla.get(key):
            raise ValueError(f"capture differs from the preceding comparison: {key}")
    if comparison.get("limit") != subject["max_relative_error_limit"]:
        raise ValueError("comparison and capture use different numerical limits")
    # The older comparison stores properties but no device UUID. Preserve that
    # limitation: matching properties do NOT establish cross-run physical identity.
    if comparison.get("device") != subject["device"].get("properties"):
        raise ValueError("capture device properties differ from the comparison")
    delivery = subject.get("wy_delivery", "scalar")
    if delivery not in PROFILE_VARIANTS:
        raise ValueError(f"unknown WY delivery: {delivery}")
    role_name = "wy" if delivery == "scalar" else f"wy-{delivery}"
    # The actual selected arm is authoritative. Early AIU comparisons wrote
    # delivery_ab=false despite recording all four arms and their masks. Never
    # edit that old JSON or accept another raw-bit-identical arm in its place.
    arm = cases[0].get("arms", {}).get(role_name, {})
    if delivery != "scalar" and not arm:
        raise ValueError("selected delivery was not admitted in the preceding comparison")
    if delivery in RESIDUAL_VARIANTS:
        validate_residual_admission(arm, comparison.get("limit"))
        if delivery in RESIDUAL_DELIVERIES:
            validate_residual_delivery_admission(arm,cases[0]["arms"].get("wy-residual", {}),comparison.get("limit"))
        if subject.get("math_contract") != MATH_CONTRACT:
            raise ValueError("capture lost residual numerical-contract identity")
    elif delivery != "scalar" or "delivery_mask" in arm:
        if type(arm.get("delivery_mask")) is not int or arm["delivery_mask"] != DELIVERIES[delivery]:
            raise ValueError(f"{role_name}: recorded delivery mask differs or is missing")
    for role, record in ((role_name, subject), ("fla", fla)):
        arm = cases[0].get("arms", {}).get(role, {})
        if arm.get("fingerprint") != record["output_sha"]:
            raise ValueError(f"{role}: output differs from the compared binary/input")
        if arm.get("state_dtype") != record.get("state_dtype") or not record.get("state_dtype"):
            raise ValueError(f"{role}: final-state precision differs from the comparison")


def validate_preflight(preflight, subject):
    if preflight.get("math_contract") != subject.get("math_contract"):
        raise ValueError("subject differs from preflight: math_contract")
    if preflight.get("wy_delivery", "scalar") != subject.get("wy_delivery", "scalar"):
        raise ValueError("subject differs from independent preflight: wy_delivery")
    if (preflight.get("status") != "PASS" or preflight.get("phase") != "preflight"
            or preflight.get("warmup", 0) < 1
            or preflight.get("public_api_calls") != preflight["warmup"] + 1):
        raise ValueError("missing successful independent preflight")
    if subject.get("phase") != "subject" or subject.get("public_api_calls") != 1 or subject.get("warmup") != 0:
        raise ValueError("subject must contain exactly one API call and no warmup")
    for key in ("role", "gate", "shape", "input_sha", "reference_sha", "output_sha", "device",
                "torch", "torch_cuda", "fla", "extension_sha256", "library_sha256", "implementation", "protocol",
                "output_dtype", "state_dtype"):
        if key not in preflight or preflight[key] != subject.get(key):
            raise ValueError(f"subject differs from independent preflight: {key}")


def validate_pair(ours, fla, implementation="original"):
    if ours.get("wy_delivery", "scalar") != fla.get("wy_delivery", "scalar"):
        raise ValueError("capture pair delivery labels differ")
    if implementation not in ("original", "wy"):
        raise ValueError(f"unknown implementation: {implementation}")
    for record, role in ((ours, "wy" if implementation == "wy" else "ours"), (fla, "fla")):
        if (record.get("status") != "PASS" or record.get("role") != role
                or record.get("implementation") != implementation
                or record.get("phase") != "subject" or record.get("warmup") != 0
                or record.get("public_api_calls") != 1):
            raise ValueError(f"{role}: missing successful single-call capture receipt")
        for key in ("input_sha", "reference_sha", "extension_sha256", "library_sha256"):
            if not record.get(key):
                raise ValueError(f"{role}: empty {key}")
    for key in ("gate", "shape", "input_sha", "reference_sha", "device", "torch", "torch_cuda",
                "initial_state", "output_final_state", "fla_heads", "max_relative_error_limit",
                "protocol", "cache_control", "extension_sha256", "library_sha256"):
        if key not in ours or ours[key] != fla.get(key):
            raise ValueError(f"profile arms differ or lack identity: {key}")


def validate_wy_control(control, subject):
    """Delivery requires RAW-BIT; one explicit new algorithm uses its oracle."""
    for record in (control, subject):
        if (record.get("status") != "PASS" or record.get("role") != "wy"
                or record.get("implementation") != "wy"):
            raise ValueError("WY control requires two successful WY receipts")
    if control.get("wy_delivery") == subject.get("wy_delivery"):
        raise ValueError("WY control silently selected the candidate delivery")
    residual = subject.get("wy_delivery") == "residual"
    if subject.get("wy_delivery") in RESIDUAL_DELIVERIES:
        if (control.get("wy_delivery") != RESIDUAL_CONTROLS[subject["wy_delivery"]] or
                control.get("math_contract") != MATH_CONTRACT or subject.get("math_contract") != MATH_CONTRACT):
            raise ValueError("residual delivery must compare same-math residual control")
    if residual:
        if (control.get("wy_delivery") != "state-pipeline" or
                control.get("math_contract") != WY_MATH_CONTRACT or
                subject.get("math_contract") != MATH_CONTRACT):
            raise ValueError("residual comparison requires its explicit same-binary pipeline control")
        for record in (control, subject):
            errors = record.get("errors", [])
            if (record.get("max_relative_error_limit") != 0.02 or len(errors) != 2 or
                    any(type(e) not in (int,float) or not math.isfinite(e) or e < 0 or e >= .02 for e in errors)):
                raise ValueError("algorithm comparison lacks independent 2% numerical admission")
            if not record.get("output_sha"):
                raise ValueError("algorithm comparison lacks output fingerprint")
    elif not control.get("output_sha") or control.get("output_sha") != subject.get("output_sha"):
        raise ValueError("WY control differs from candidate: output_sha")
    for key in ("gate", "shape", "input_sha", "reference_sha", "device",
                "torch", "torch_cuda", "initial_state", "output_final_state", "fla_heads",
                "max_relative_error_limit", "protocol", "cache_control", "phase", "warmup",
                "public_api_calls", "extension_sha256", "library_sha256", "output_dtype", "state_dtype"):
        if key not in control or control[key] != subject.get(key):
            raise ValueError(f"WY control differs from candidate: {key}")


def validate_loaded_binary(receipt, extension, library):
    loaded = receipt.get("loaded_libraries", {})
    for binary in (extension, library):
        matches = [digest for path, digest in loaded.items() if Path(path).name == binary.name]
        if matches != [sha(binary)]:
            raise ValueError(f"loaded binary differs from archive or is absent: {binary.name}")


def validate_requested_sequence(receipt, sequence):
    if receipt.get("shape") != dict(B=1, S=sequence, Hk=16, Hv=32, K=128, V=128):
        raise ValueError("capture ignored the requested shape/sequence")


def pack(bundle, status):
    (bundle / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("bundle must not contain symlinks")
    checksums = bundle / "SHA256SUMS"
    checksums.write_text("".join(f"{sha(p)}  {p.relative_to(bundle).as_posix()}\n"
                                 for p in files if p != checksums))
    archive = bundle.parent / f"{bundle.parent.name}.tar.gz"
    with archive.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            tar.add(path, arcname=f"{bundle.parent.name}/{path.relative_to(bundle)}", recursive=False)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{sha(archive)}  {archive.name}\n")
    print(f"[GDN ACU bundle] {status['status']}; UPLOAD={archive} bytes={archive.stat().st_size}", flush=True)
    return archive


def find_acu():
    if os.environ.get("ACU"):
        return Path(os.environ["ACU"]).resolve()
    # Operator-selected site profiler (2026-09-24). Tool selection is not
    # capability discovery: a missing executable must fail at admission,
    # never silently choose the SDK or PATH version instead.
    return DEFAULT_ACU


def validate_backend_source_snapshot(root):
    # These are now transitive compile/load dependencies, not just documentation.
    # A reused binary still needs the actual moved source and target authority.
    required = (
        "csrc/backends/ppu_aiu/primitives.cuh",
        "csrc/backends/cuda_sm80/primitives.cuh",
        "csrc/backends/cuda_sm80/build.py",
        "include/gdn_qsa/backend_target.h",
        "cmake/GdnBackend.cmake", "cmake/backends/ppu_aiu.cmake",
        "actlize_la/backends/targets.json", "actlize_la/backends/loading.py",
    )
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"backend source snapshot incomplete: {missing}")


def collect(args, bundle, env):
    implementation = "wy" if args.wy_run else "original"
    delivery = getattr(args, "wy_delivery", "scalar")
    status = dict(status="INCOMPLETE", errors=[], probes={}, implementation=implementation, wy_delivery=delivery)
    try:
        arms = capture_arms(bundle, implementation, delivery, getattr(args, "wy_control", None))
        status["capture_order"] = [arm.label for arm in arms]
        status["capture_arms"] = {arm.label: dict(role=arm.role, wy_delivery=arm.delivery,
            directory=str(arm.directory.relative_to(bundle))) for arm in arms}
        for arm in arms:
            arm.directory.mkdir(exist_ok=True)
        for name, command in (
            ("git-head", ["git", "rev-parse", "HEAD"]),
            ("git-status", ["git", "status", "--porcelain=v1"]),
            ("git-diff", ["git", "diff", "HEAD", "--binary"]),
            ("submodules", ["git", "submodule", "status", "--recursive"]),
        ):
            run(command, bundle / f"{name}.txt", env, console=False)
        for directory in ("csrc/gdn_chunk", "csrc/backends", "include/gdn_qsa", "actlize_la", "cmake",
                          "benchmarks", "tests", "tools", "scripts", "dev/ppu", "dev/backends"):
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".hpp", ".cpp", ".h", ".sh", ".cmake", ".json"):
                    copy_file(path, bundle / "sources/ours" / path.relative_to(ROOT))
        for name in ("CMakeLists.txt", ".gitmodules", "setup.py", "pyproject.toml", "docs/PPU_BACKEND.md"):
            copy_file(ROOT / name, bundle / "sources/ours" / name)
        validate_backend_source_snapshot(bundle / "sources/ours")
        copy_file(ROOT / "docs/PPU_GDN_ACU.md", bundle / "README.md")
        if env.get("FLA_ROOT") and not (Path(env["FLA_ROOT"]) / "fla/__init__.py").is_file():
            raise RuntimeError(f"FLA_ROOT is not an FLA checkout: {env['FLA_ROOT']}")
        if not args.acu.is_file() or not os.access(args.acu, os.X_OK):
            raise RuntimeError(f"ACU unavailable: {args.acu}; set ACU=/path/to/acu")
        status["acu_identity"] = dict(path=str(args.acu), sha256=sha(args.acu),
            sdk_bundled=args.acu.resolve() == (args.sdk / "asight/bin/acu").resolve())
        print(f"[GDN ACU tool] sdk={args.sdk} acu={args.acu} "
              f"sdk_bundled={int(status['acu_identity']['sdk_bundled'])}", flush=True)
        status["probes"]["acu-version"] = run([args.acu, "--version"], bundle / "acu-version.txt", env)
        for name, command in (("acu-help", [args.acu, "--help"]),
                              ("hgcc-version", [args.sdk / "bin/hgcc", "--version"]),
                              ("host", ["uname", "-a"])):
            status["probes"][name] = run(command, bundle / f"{name}.txt", env,
                                          optional=True, console=False, timeout=20)
        smi = shutil.which("ppu-smi", path=env["PATH"])
        if smi:
            status["probes"]["device-before"] = run([smi, "-i", args.device, "-q"],
                bundle / "device-before.txt", env, optional=True, console=False, timeout=20)
        else:
            status["probes"]["ppu-smi"] = dict(status="UNAVAILABLE", reason="not on PATH")

        extension = args.extension
        comparison = None
        if args.wy_run:
            extension, comparison, origin = read_wy_run(args.wy_run)
            status["binary_source_binding"] = "reused-comparison; current sources are capture helpers, not binary origin"
            status["comparison_origin"] = origin
            status["prior_device_identity"] = "properties-only; previous comparison did not record a UUID"
            for name in ("sha.txt", "source.diff", "binaries.sha256", "comparison.json", "comparison.log", "wy-correctness.log", "residual-correctness.log"):
                if (args.wy_run / name).is_file():
                    copy_file(args.wy_run / name, bundle / "preceding-comparison" / name)
        elif extension is None:
            build = bundle.parent / "build"
            build_env = env | {"BUILD_DIR": str(build), "PPU_SDK": str(args.sdk)}
            run(["bash", ROOT / "scripts/build_ppu.sh"], bundle / "build.log", build_env)
            candidates = list(build.glob("_gdn_chunk_ppu*.so"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one built binding; found {candidates}")
            extension = candidates[0].resolve()
            status["binary_source_binding"] = "built-in-this-run (git head/diff and sources included)"
        else:
            status["binary_source_binding"] = "operator-supplied; current source not asserted as binary origin"
        library = extension.parent / ("libgdn_wy_ppu.so" if implementation == "wy" else "libgdn_qsa_ppu.so")
        arch_contract = extension.parent / "gdn_hgcc_arch.txt"
        if arch_contract.is_file():
            copy_file(arch_contract, bundle / arch_contract.name)
        for binary in (extension, library):
            if not binary.is_file():
                raise RuntimeError(f"PPU binary missing: {binary}")
            copy_file(binary, bundle / "binaries" / binary.name)
        status["binaries"] = (origin["binary_sha256"] if args.wy_run else
                              {str(p): sha(p) for p in (extension, library)})
        for path, expected in status["binaries"].items():
            if sha(path) != expected or sha(bundle / "binaries" / Path(path).name) != expected:
                raise ValueError(f"binary changed while preparing capture: {path}")
        for option, filename in (("--dump-resource-usage=all", "resources.txt"), ("--dump-isa", "isa.txt")):
            status["probes"][filename] = run(
                [args.sdk / "bin/hgobjdump", "--arch=ppu1.0", option, library],
                bundle / filename, env, optional=True, console=False, timeout=60)

        for arm in arms:
            try:
                run(child_command(extension, arm.role, args.gate, arm.directory, "preflight", implementation, arm.delivery, args.sequence),
                    arm.directory / f"{arm.role}-preflight.log", env)
            except Exception as exc:
                status["errors"].append(f"{arm.label} preflight: {type(exc).__name__}: {exc}")
        if status["errors"]:
            return status
        preflights = {arm.label: json.loads((arm.directory / f"{arm.role}-preflight.json").read_text())
                      for arm in arms}
        for arm in arms:
            validate_requested_sequence(preflights[arm.label], args.sequence)
            if preflights[arm.label].get("wy_delivery", "scalar") != arm.delivery:
                raise ValueError(f"{arm.label}: preflight ignored selected delivery")
        if comparison is not None:
            for arm in arms[:-1]:
                validate_comparison(comparison, preflights[arm.label], preflights["fla"])
        if "wy-control" in preflights:
            validate_wy_control(preflights["wy-control"], preflights["wy"])

        for arm in arms:
            try:
                base = arm.directory / f"{arm.role}-g{args.gate}.report"
                command = acu_command(args.acu, base, extension, arm.role, args.gate,
                                      arm.directory, implementation, arm.delivery, args.sequence)
                run(command, arm.directory / f"{arm.role}-acu.log", env)
                report = report_file(base)
                # Native reports remain the authority. Text exports make the tar
                # inspectable without the GUI; no CSV/copy-paste required.
                for page in ("details", "raw"):
                    status["probes"][f"{arm.label}-{page}"] = run(
                        [args.acu, "--import", report, "--page", page],
                        arm.directory / f"{arm.role}-{page}.txt", env,
                        optional=True, console=False, timeout=120)
            except Exception as exc:
                status["errors"].append(f"{arm.label}: {type(exc).__name__}: {exc}")
                print(f"[GDN ACU bundle] FAIL: {status['errors'][-1]}", flush=True)
        if smi:
            status["probes"]["device-after"] = run([smi, "-i", args.device, "-q"],
                bundle / "device-after.txt", env, optional=True, console=False, timeout=20)
        if status["errors"]:
            return status
        receipts = {arm.label: json.loads((arm.directory / f"{arm.role}.json").read_text()) for arm in arms}
        ours, fla = receipts[arms[-2].label], receipts["fla"]
        for arm in arms:
            validate_preflight(preflights[arm.label], receipts[arm.label])
        validate_pair(ours, fla, implementation)
        if "wy-control" in receipts:
            validate_wy_control(receipts["wy-control"], ours)
        if comparison is not None:
            for arm in arms[:-1]:
                validate_comparison(comparison, receipts[arm.label], fla)
        for arm in arms[:-1]:
            receipt = receipts[arm.label]
            validate_loaded_binary(receipt, extension, library)
            if (receipt["extension_sha256"] != status["binaries"][str(extension)]
                    or receipt["library_sha256"] != status["binaries"][str(library)]):
                raise ValueError("profiled binding/library differs from archived binary")
        for path, expected in status["binaries"].items():
            if sha(path) != expected:
                raise ValueError(f"binary changed during capture: {path}")
        status["status"] = "PASS"
    except Exception as exc:
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        (bundle / "failure.txt").write_text(traceback.format_exc())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=float, choices=(-0.1, -1.0), default=-0.1)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--extension", type=Path, default=os.environ.get("EXTENSION"))
    parser.add_argument("--wy-run", type=Path,
                        help="reuse this admitted WY comparison/numeric-receipt directory; never compile")
    parser.add_argument("--wy-delivery", choices=tuple(PROFILE_VARIANTS), default="scalar")
    parser.add_argument("--wy-control", choices=tuple(PROFILE_VARIANTS),
                        help="also capture this same-binary WY control before the candidate; FLA runs once")
    args = parser.parse_args()
    if args.sequence <= 0:
        parser.error("sequence must be positive")
    if args.wy_run and args.extension:
        parser.error("--wy-run and --extension/EXTENSION are mutually exclusive")
    if args.wy_delivery != "scalar" and not args.wy_run:
        parser.error("--wy-delivery requires --wy-run with an admitted WY receipt")
    if args.wy_control is not None and (not args.wy_run or args.wy_control == args.wy_delivery):
        parser.error("--wy-control requires --wy-run and a different --wy-delivery")
    if args.wy_delivery == "residual" and args.wy_control != "state-pipeline":
        parser.error("residual algorithm requires --wy-control state-pipeline")
    if args.wy_delivery in RESIDUAL_DELIVERIES and args.wy_control != RESIDUAL_CONTROLS[args.wy_delivery]:
        parser.error(f"residual delivery requires --wy-control {RESIDUAL_CONTROLS[args.wy_delivery]}")
    if args.wy_run:
        args.wy_run = args.wy_run.resolve()
    args.sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")).resolve()
    args.acu = find_acu()
    args.device = os.environ.get("DEVICE", "0")
    if not args.device.isdigit():
        parser.error("DEVICE must be one physical PPU's nonnegative integer index")
    if args.extension:
        args.extension = args.extension.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(os.environ.get("OUT", f"/workspace/gdn-qsa-acu-{stamp}-{os.getpid()}"))
    # Own a new directory; no stale reports, destructive cleanup, or mktemp.
    out.mkdir(parents=True, exist_ok=False)
    bundle = out.resolve() / "bundle"
    bundle.mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.device, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PPU_SDK=str(args.sdk),
               PATH=f"{args.sdk}/bin:" + os.environ.get("PATH", ""),
               LD_LIBRARY_PATH=f"{args.sdk}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
               PYTHONPATH=":".join(filter(None, [str(ROOT), os.environ.get("FLA_ROOT"),
                                                os.environ.get("PYTHONPATH")])))
    status = collect(args, bundle, env)
    status.update(created_utc=stamp, gate=args.gate, sequence=args.sequence, physical_device=args.device,
                  sdk=str(args.sdk), acu=str(args.acu),
                  scope="ACU kernel durations and counters; not complete public-API event timing")
    pack(bundle, status)
    if status["errors"]:
        print("\n".join(status["errors"]), file=sys.stderr)
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
