#!/usr/bin/env python3
"""Separate preflight and subject-only processes for direct ACU capture.

This is a counter capture, NOT another latency benchmark. Reuse the comparison
fixture, oracle, tolerance and FLA dispatch/compatibility code without forking
their numerical contract.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

import bench_ppu_gdn_fla as bench
from actlize_la.gdn_wy_interface import DELIVERIES
from actlize_la.gdn_residual_interface import PROFILE_VARIANTS, RESIDUAL_VARIANTS, math_contract


def subject_call(role, implementation, extension, inputs, delivery="scalar"):
    """Select an explicit API; never let 'ours' silently change meaning."""
    expected = "wy" if implementation == "wy" else "ours"
    if implementation not in ("original", "wy") or role not in (expected, "fla"):
        raise ValueError(f"role {role} does not belong to {implementation}/FLA comparison")
    if delivery not in PROFILE_VARIANTS or (implementation != "wy" and delivery != "scalar"):
        raise ValueError("delivery selection requires an explicit WY implementation")
    if role == "fla":
        fn, identity = bench.load_fla()
        return bench.fla_call(fn, inputs, "native"), identity
    if role == "wy":
        from actlize_la import gdn_chunk_wy
        os.environ["GDN_QSA_WY_EXTENSION"] = str(extension.resolve())
        if delivery in RESIDUAL_VARIANTS:
            from actlize_la import gdn_chunk_residual
            return lambda: gdn_chunk_residual(*inputs, output_final_state=True,
                                              delivery=RESIDUAL_VARIANTS[delivery]), {}
        if delivery == "scalar":
            return lambda: gdn_chunk_wy(*inputs, output_final_state=True), {}
        return lambda: gdn_chunk_wy(*inputs, output_final_state=True, delivery=delivery), {}
    os.environ["GDN_QSA_PPU_EXTENSION"] = str(extension.resolve())
    return lambda: bench.admission.gdn_chunk(*inputs, output_final_state=True), {}


def loaded_library_paths():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if path.is_file():
                paths.add(path.resolve())
    return paths


def loaded_library_hashes():
    # Read existing mappings only: no extra library is loaded and no profiler
    # API is invoked. Include the parser/injection DSOs on failures as well.
    prefixes = ("libhggc", "libcuda", "libgdn", "_gdn_chunk", "_gdn_wy",
                "libhg_wrapper", "libasight", "libhgpti", "libperfworks",
                "libhgBinaryAnalysis", "libhgdisassembler", "libcheckpoint")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in loaded_library_paths() if path.name.startswith(prefixes)}


def run_with_failure_receipt(call, receipt, identity):
    try:
        return call()
    except Exception as error:
        # Preserve the actual exception. A failed profiler must not disappear
        # before the loaded SDK/profiler identities can join the diagnostic tar.
        failure = identity | dict(status="FAIL", error_type=type(error).__name__, error=str(error))
        try:
            failure["loaded_libraries"] = loaded_library_hashes()
            receipt.write_text(json.dumps(failure, indent=2) + "\n")
        except Exception as diagnostic_error:
            print(f"[PPU GDN ACU failure receipt] UNAVAILABLE: {diagnostic_error}", file=sys.stderr)
        raise


def capture_one(call, synchronize):
    """No profiler API, warmup loop or alternate kernel in the subject process."""
    synchronize()
    result = call()
    synchronize()
    return result


def checked_cpu_pair(result, want):
    # bench.checked_pair/error casts to float. Copy first, so those casts and
    # all comparison kernels execute on CPU, not under ACU on the PPU.
    result_cpu = tuple(x.detach().cpu() for x in result)
    return result_cpu, bench.checked_pair(result_cpu, want)


def run_phase(call, synchronize, want, phase, warmup):
    if phase not in ("preflight", "subject"):
        raise ValueError(f"unknown capture phase: {phase}")
    first, errors = checked_cpu_pair(capture_one(call, synchronize), want)
    fingerprint = bench.admission.digest(first)
    warmups = warmup if phase == "preflight" else 0
    for _ in range(warmups):
        warm, _ = checked_cpu_pair(capture_one(call, synchronize), want)
        if bench.admission.digest(warm) != fingerprint:
            raise AssertionError("preflight output/state is not bit-stable")
    return dict(errors=errors, output_sha=fingerprint, warmup=warmups,
                output_dtype=str(first[0].dtype), state_dtype=str(first[1].dtype),
                public_api_calls=1 + warmups)


def save_fla_sources(destination):
    """Snapshot imported FLA Python sources, not caches, datasets or site-packages."""
    import fla
    package = Path(fla.__file__).resolve().parent
    paths = {Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
             if (name == "fla" or name.startswith("fla."))
             and getattr(module, "__file__", None)
             and Path(module.__file__).suffix == ".py"}
    from triton.backends.nvidia import compiler
    compiler_source = Path(compiler.__file__).resolve()
    paths.add(compiler_source)
    manifest = []
    for source in sorted(paths):
        if source.is_relative_to(package):
            relative = Path("fla") / source.relative_to(package)
        elif source == compiler_source:
            relative = Path("triton") / "nvidia_compiler.py"
        else:
            # A vendor may alias an external helper into the fla namespace.
            # Preserve it without colliding with the Triton compiler snapshot.
            relative = Path("external-fla") / hashlib.sha256(str(source).encode()).hexdigest()[:16] / source.name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append(dict(original=str(source), saved=str(relative),
                             sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    return manifest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("ours", "wy", "fla"))
    parser.add_argument("--implementation", choices=("original", "wy"), default="original")
    parser.add_argument("--wy-delivery", choices=tuple(PROFILE_VARIANTS), default="scalar")
    parser.add_argument("--phase", required=True, choices=("preflight", "subject"))
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--gate", type=float, choices=(-0.1, -1.0), default=-0.1)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    if args.sequence <= 0:
        parser.error("sequence must be positive")
    if not args.extension.is_file() or args.warmup < 1:
        parser.error("require an existing PPU extension and at least one warmup")
    library = args.extension.parent / ("libgdn_wy_ppu.so" if args.implementation == "wy" else "libgdn_qsa_ppu.so")
    prefix = "_gdn_wy_ppu" if args.implementation == "wy" else "_gdn_chunk_ppu"
    if not args.extension.name.startswith(prefix) or not library.is_file():
        parser.error(f"{args.implementation} requires {prefix}*.so and {library.name}")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)  # physical device is fixed by CUDA_VISIBLE_DEVICES
    props = torch.cuda.get_device_properties(0)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU device: {props.name}")

    cpu = bench.admission.fixture(1, args.sequence, 16, 32, args.gate)
    want = bench.admission.reference(cpu)
    inputs = tuple(x.cuda() for x in cpu)
    input_hash = bench.admission.digest(cpu)
    call, identity = subject_call(args.role, args.implementation, args.extension, inputs, args.wy_delivery)

    print(f"[PPU GDN ACU config] role={args.role} implementation={args.implementation} phase={args.phase} g={args.gate} "
          f"wy_delivery={args.wy_delivery} "
          f"shape=B1,S{args.sequence},Hk16,Hv32,D128 input_sha={input_hash} "
          "initial_state=zero final_state=1 GVA=native forward_only=1", flush=True)
    if args.phase == "subject":
        print(f"[PPU GDN ACU subject-only] role={args.role} public_api_calls=1 "
              "warmup=0 verification_device_kernels=0 profile_control=external-acu", flush=True)
    measured = run_with_failure_receipt(
        lambda: run_phase(call, torch.cuda.synchronize, want, args.phase, args.warmup),
        args.receipt, dict(role=args.role, implementation=args.implementation, wy_delivery=args.wy_delivery, phase=args.phase,
            gate=args.gate, input_sha=input_hash, device=str(props),
            extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest()))
    if bench.admission.digest(inputs) != input_hash:
        raise AssertionError(f"{args.phase} modified fixture inputs")
    print(f"[PPU GDN ACU check] role={args.role} phase={args.phase} "
          f"output_state_error={measured['errors']} output_sha={measured['output_sha']} "
          "verification=CPU NUMERIC/PASS", flush=True)

    source_manifest = save_fla_sources(args.sources) if args.role == "fla" else []
    # Actual mapped library paths/hashes expose a stale dependency even when the
    # extension filename itself looks current. Do not archive process env vars.
    loaded = loaded_library_hashes()
    receipt = dict(status="PASS", role=args.role, phase=args.phase, gate=args.gate,
                   implementation=args.implementation, wy_delivery=args.wy_delivery,
                   math_contract=math_contract(args.wy_delivery) if args.role == "wy" else "reference",
                   shape=dict(B=1, S=args.sequence, Hk=16, Hv=32, K=128, V=128),
                   input_sha=input_hash, reference_sha=bench.admission.digest(want),
                   fixture_seed=0x6A09E667, gate_bf16=float(cpu[3].flatten()[0]),
                   **measured,
                   max_relative_error_limit=bench.admission.MAX_RELATIVE_ERROR,
                   initial_state="zero", output_final_state=True, fla_heads="native",
                   protocol="ACU-direct-subject-process-v3",
                   timing_scope="PROFILED_DIAGNOSTIC_NOT_BENCHMARK",
                   cache_control="ACU default; not benchmark cache state",
                   capture_scope="whole subject process, including runtime/library setup if any",
                   autotune_scope="library-internal autotuning may repeat in the fresh process; not excluded by a range",
                   torch=torch.__version__, torch_cuda=torch.version.cuda,
                   torch_build_config=torch.__config__.show(),
                   python=sys.version, python_executable=sys.executable,
                   device=dict(name=props.name, cu=props.multi_processor_count,
                               total_memory=props.total_memory,
                               properties=str(props),
                               uuid=str(getattr(props, "uuid", "UNAVAILABLE")),
                               visible=os.environ.get("CUDA_VISIBLE_DEVICES")),
                   extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
                   library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                   loaded_libraries=loaded, fla=identity, fla_sources=source_manifest)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[PPU GDN ACU] PASS: role={args.role} phase={args.phase} "
          f"API_calls={measured['public_api_calls']} receipt={args.receipt}", flush=True)


if __name__ == "__main__":
    main()
