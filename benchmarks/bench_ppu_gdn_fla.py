#!/usr/bin/env python3
"""Same-input, forward-only PPU GDN versus installed FLA's Triton path.

Public-API event spans INCLUDE launch gaps and host dispatch synchronization.
Do not compare these to a sum of profiled device-kernel durations.
"""
import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# One fixture/reference/criterion authority, shared with box admission.
spec = importlib.util.spec_from_file_location(
    "ppu_admission", ROOT / "tests/test_ppu_gdn_backend.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def cuda13_ptx_compat(compiler):
    """Backport only the known CUDA 13.0 parser gap; never alter site-packages.

    Authority: Triton v3.5.0 third_party/nvidia/backend/compiler.py,
    ptx_get_version: CUDA 13.0 maps to PTX 9.0 (integer 90).
    Keep all successful vendor mappings and all unrelated failures unchanged.
    This does not select a backend/assembler or prove its PPU execution.
    """
    original = compiler.ptx_get_version
    try:
        version = original("13.0")
    except RuntimeError as exc:
        expected = "Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0"
        if str(exc) != expected:
            raise
    else:
        return dict(status="native-mapping-unchanged", cuda="13.0", ptx=version)

    @functools.wraps(original)
    def mapped(cuda_version):
        if cuda_version == "13.0":
            return 90
        return original(cuda_version)

    compiler.ptx_get_version = mapped
    return dict(status="process-local-upstream-backport", cuda="13.0", ptx=90,
                authority="triton-v3.5.0", scope="version-parser-only")


def load_fla():
    # New FLA can dispatch to FlashQLA etc. This baseline is explicitly Triton.
    if "fla" in sys.modules:
        raise RuntimeError("FLA must be imported after fixing its backend selection")
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    try:
        import triton
        from triton.backends.nvidia import compiler
        ptx_compat = cuda13_ptx_compat(compiler)
        print("[PPU GDN FLA PTX compatibility] " + json.dumps(ptx_compat, sort_keys=True), flush=True)
        import fla
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception as exc:
        raise RuntimeError(
            "FLA baseline unavailable; no comparison or speedup is valid. "
            "Use the box's working PPU-compatible FLA/Triton installation "
            "(or set FLA_ROOT to its checkout). Original import error: "
            f"{type(exc).__name__}: {exc}") from exc
    source = Path(inspect.getfile(inspect.unwrap(chunk_gated_delta_rule))).resolve()
    identity = dict(version=getattr(fla, "__version__", "UNKNOWN"),
                    triton_version=triton.__version__, source=str(source),
                    entry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    triton_compiler=str(Path(compiler.__file__).resolve()),
                    triton_compiler_sha256=hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest(),
                    cuda13_ptx_compat=ptx_compat,
                    backend_dispatch="disabled-before-import")
    print("[PPU GDN FLA identity] " + json.dumps(identity, sort_keys=True), flush=True)
    return chunk_gated_delta_rule, identity


def fla_call(fn, inputs, head_mode):
    q, k, v, g, beta = inputs
    if head_mode == "expanded":
        # Explicit compatibility mode, never a silent retry. Outside timing.
        ratio = v.shape[2] // q.shape[2]
        q, k = q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2)
    elif head_mode != "native":
        raise ValueError(f"unsupported FLA head mode: {head_mode}")
    params = inspect.signature(fn).parameters
    kwargs = dict(scale=q.shape[-1] ** -0.5, initial_state=None,
                  output_final_state=True, use_qk_l2norm_in_kernel=False)
    # Older FLA takes head_first; newer versions explicitly reject that kwarg.
    if "head_first" in params:
        kwargs["head_first"] = False
    for name in ("state_v_first", "use_beta_sigmoid_in_kernel", "use_gate_in_kernel"):
        if name in params:
            kwargs[name] = False
    if "chunk_size" in params or any(
            x.kind == inspect.Parameter.VAR_KEYWORD for x in params.values()):
        kwargs["chunk_size"] = 64
    # No gate cumsum, QK normalization, state transpose or dtype cast outside
    # the timed baseline. g is natural-log decay; beta is already activated.
    return lambda: fn(q, k, v, g, beta, **kwargs)


def checked_pair(got, want):
    if not isinstance(got, (tuple, list)) or len(got) != 2:
        raise AssertionError("both output AND final state are required")
    for name, actual, expected in zip(("output", "state"), got, want):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
            raise AssertionError(f"{name}: missing tensor or wrong shape")
    return admission.assert_pair(got, want)


def sample_order(index):
    return ("ours", "fla") if index % 2 == 0 else ("fla", "ours")


def verdict(ours, fla):
    if not ours or not fla or any(not math.isfinite(x) or x <= 0 for x in (*ours, *fla)):
        raise ValueError("timing samples must be finite, positive and nonempty")
    # An observed overlapping sample envelope is not a resolved winner.
    if max(ours) < min(fla):
        return "OURS-WINS"
    if max(fla) < min(ours):
        return "FLA-WINS"
    return "UNRESOLVED"


@torch.inference_mode()
def run_case(fn, gate, args, device):
    cpu = admission.fixture(1, 2048, 16, 32, gate)
    want = admission.reference(cpu)
    inputs = tuple(x.to(device) for x in cpu)
    source_hash = admission.digest(cpu)
    calls = dict(ours=lambda: admission.gdn_chunk(*inputs, output_final_state=True),
                 fla=fla_call(fn, inputs, args.fla_heads))
    print(f"[PPU GDN FLA config] shape=B1,S2048,Hk16,Hv32,D128 dtype=bf16 "
          f"g={gate} input_sha={source_hash} reference_sha={admission.digest(want)} "
          f"ours_expected_route={'reset-GC8' if gate == -1.0 else 'serial'} "
          f"initial_state=zero final_state=1 scale=1/sqrt(128) qk_norm=0 "
          f"gva={args.fla_heads} expansion_timed=0 chunk_ours=16 chunk_fla=default64 "
          f"protocol=full-public-api-event-span includes_host_dispatch_sync=1 "
          f"warmup={args.warmup} samples={args.samples} launches={args.launches}", flush=True)
    record = dict(gate=gate, input_sha=source_hash, arms={})
    for role, call in calls.items():
        torch.cuda.synchronize()
        before_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        first = call()  # Includes FLA JIT/autotune; explicitly NOT timed.
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - before_bytes
        try:
            errors = checked_pair(first, want)
        except AssertionError:
            if isinstance(first, (tuple, list)) and len(first) == 2 and all(
                    isinstance(x, torch.Tensor) and x.shape == y.shape
                    for x, y in zip(first, want)):
                admission.print_failure(f"fla-ab/g={gate}/{role}", first, want)
            raise
        fingerprint = admission.digest(first)
        for _ in range(3):
            if admission.digest(call()) != fingerprint:
                raise AssertionError(f"{role}: preflight not bit-stable")
        if admission.digest(inputs) != source_hash:
            raise AssertionError(f"{role}: modified the shared input")
        record["arms"][role] = dict(errors=errors, output_sha=fingerprint,
                                   output_dtype=str(first[0].dtype),
                                   state_dtype=str(first[1].dtype),
                                   peak_allocation_delta_bytes=peak_delta, samples_us=[])
        print(f"[PPU GDN FLA admission] g={gate} role={role} "
              f"output_state_error={errors} limit={admission.MAX_RELATIVE_ERROR} "
              f"state_dtype={first[1].dtype} repeat=4/4 RAW-BIT/STABLE NUMERIC/PASS", flush=True)
        del first
        for _ in range(args.warmup):
            call()
    for sample in range(args.samples):
        for role in sample_order(sample):
            torch.cuda.synchronize()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.launches):
                result = calls[role]()
            end.record()
            end.synchronize()
            elapsed_us = start.elapsed_time(end) * 1000 / args.launches
            if not math.isfinite(elapsed_us) or elapsed_us <= 0:
                raise RuntimeError(f"{role}: invalid event span {elapsed_us}")
            checked_pair(result, want)
            if admission.digest(result) != record["arms"][role]["output_sha"]:
                raise AssertionError(f"{role}: timed replay changed output")
            record["arms"][role]["samples_us"].append(elapsed_us)
            del result
    if admission.digest(inputs) != source_hash:
        raise AssertionError("timed calls modified the shared input")
    for role, arm in record["arms"].items():
        samples = arm["samples_us"]
        arm["median_us"] = statistics.median(samples)
        print(f"[PPU GDN FLA perf] g={gate} role={role} median_us={arm['median_us']:.3f} "
              f"range=[{min(samples):.3f},{max(samples):.3f}] "
              f"samples_us={','.join(f'{x:.3f}' for x in samples)} "
              f"peak_delta_bytes={arm['peak_allocation_delta_bytes']} "
              f"output_sha={arm['output_sha']}", flush=True)
    ours, fla = record["arms"]["ours"], record["arms"]["fla"]
    record["speedup_fla_over_ours"] = fla["median_us"] / ours["median_us"]
    record["verdict"] = verdict(ours["samples_us"], fla["samples_us"])
    print(f"[PPU GDN FLA compare] g={gate} ours_us={ours['median_us']:.3f} "
          f"fla_us={fla['median_us']:.3f} speedup={record['speedup_fla_over_ours']:.4f}x "
          f"verdict={record['verdict']} rule=disjoint-observed-envelopes", flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fla-heads", choices=("native", "expanded"), default="native")
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    if not args.extension.is_file():
        p.error("PPU extension not found")
    if args.warmup < 5 or args.samples < 3 or args.samples * args.launches < 50 or args.launches < 1:
        p.error("require warmup >= 5, samples >= 3, and >= 50 timed launches per arm")
    os.environ["GDN_QSA_PPU_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU device: {props.name}")
    fn, identity = load_fla()
    print(f"[PPU GDN FLA device] name={props.name} cu={props.multi_processor_count} "
          f"torch={torch.__version__} extension={args.extension}", flush=True)
    result = dict(protocol="full-public-api-event-span", fla=identity,
                  torch=torch.__version__, device=props.name, cu=props.multi_processor_count,
                  shape=dict(B=1, S=2048, Hk=16, Hv=32, K=128, V=128),
                  input_dtype="bf16", initial_state="zero", output_final_state=True,
                  max_relative_error_limit=admission.MAX_RELATIVE_ERROR,
                  warmup=args.warmup, samples=args.samples, launches=args.launches,
                  fla_heads=args.fla_heads, cases=[])
    for gate in (-1.0, -0.1):
        result["cases"].append(run_case(fn, gate, args, torch.device("cuda", args.device)))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[PPU GDN FLA] PASS: paired correctness and timing; results={args.results}")


if __name__ == "__main__":
    main()
