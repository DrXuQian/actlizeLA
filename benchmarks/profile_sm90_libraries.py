#!/usr/bin/env python3
"""Pinned upstream SM90 GDN references, same-input nsys kernel-sum measurement.

See docs/SM90_LIBRARY_NSYS_PLAN.md. No performance-dependent selection or
shipping dispatch edits. Reference dependency environments remain separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "benchmarks"), str(ROOT / "tools")]
import torch
from bench_sm90_hopper import DeviceWatch
from profile_sm90_cula import build_receipt, sha
from test_ppu_gdn_backend import fixture, assert_pair, digest
from actlize_la.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from actlize_la.gdn_sm90_interface import gdn_chunk_sm90
from analyze_sm90_nsys import LIBRARY_ROLES

PINS = {"flashqla": "a97c9783bbcc42fa8fbfe895dfc674131e376b5c",
        "flashinfer": "5d9f8c8d97fa53e22952ce8672f475d235f07478"}
SHAPE = (1, 2048, 16, 32)


def validate_archive(root, archive):
    """Bind executed sources to the official git archive, not an operator label."""
    hashes = {}
    with tarfile.open(archive) as source:
        if source.pax_headers.get("comment") not in PINS.values():
            raise RuntimeError("archive lacks registered git commit identity")
        pin = source.pax_headers["comment"]
        for member in source:
            if not member.isfile():
                continue
            path = (root / member.name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise RuntimeError("archive path escapes source root")
            expected = hashlib.sha256(source.extractfile(member).read()).hexdigest()
            if sha(path) != expected:
                raise RuntimeError(f"reference source changed: {member.name}")
            hashes[member.name] = expected
    return dict(pin=pin, archive_sha256=sha(archive), files=hashes)


def versions():
    names = ("torch", "triton", "tilelang", "apache-tvm-ffi", "nvidia-cutlass-dsl", "cuda-python")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def reference_binaries(family, out):
    """Hash the actual loaded JIT code, without changing compiler options."""
    result = []
    if family == "flashqla":
        cache = Path(os.environ["TILELANG_CACHE_DIR"]).resolve()
        mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                  if line.rstrip().endswith("/executable.so")}
        for name in sorted(mapped):
            path = Path(name).resolve()
            if path.is_relative_to(cache):
                result.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    else:
        from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import _in_mem_compile_cache
        from cutlass.base_dsl.jit_executor import walk_module_and_get_cubin_data
        output = out / "jit-binaries"
        output.mkdir(exist_ok=True)

        def record(symbol, function, data):
            digest = hashlib.sha256(data).hexdigest()
            path = output / f"{digest}.cubin"
            path.write_bytes(data)
            result.append(dict(symbol=symbol, function=function, path=str(path),
                               sha256=digest, bytes=len(data)))

        for compiled in _in_mem_compile_cache.values():
            before = len(result)
            for symbol in compiled.kernel_info:
                walk_module_and_get_cubin_data(compiled.ir_module, symbol, record)
            if len(result) == before:
                # TVM-FFI lowering embeds the GPU images into the host object;
                # kernel_info/gpu.binary need not survive that lowering.
                data = bytes(compiled.dump_to_object("gdn_reference"))
                digest = hashlib.sha256(data).hexdigest()
                path = output / f"{digest}.o"
                path.write_bytes(data)
                gpu_elfs = []
                start = data.find(b"\x7fELF", 1)
                while start != -1:
                    if start + 20 <= len(data) and int.from_bytes(data[start+18:start+20], "little") == 190:
                        gpu_elfs.append(start)
                    start = data.find(b"\x7fELF", start+4)
                if not gpu_elfs:
                    raise RuntimeError("exported JIT object contains no CUDA ELF image")
                result.append(dict(function=compiled.function_name, path=str(path),
                                   sha256=digest, bytes=len(data),
                                   kind="JIT_OBJECT_WITH_EMBEDDED_CUDA_ELF", cuda_elf_offsets=gpu_elfs))
    if not result:
        raise RuntimeError("reference JIT binary identity unavailable")
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=tuple(PINS), required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--cuda-extension", type=Path, required=True)
    p.add_argument("--ppu-source-extension", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate", type=float, choices=(-.1, -1.), required=True)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "receipt.json").exists():
        raise RuntimeError("refuse to overwrite a measurement receipt")
    if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("require the unremapped single-Hopper host")
    source = validate_archive(args.reference_root, args.source_archive)
    if source["pin"] != PINS[args.family]:
        raise RuntimeError("reference family/archive identity mismatch")
    builds = {
        "ours-cuda": build_receipt(args.cuda_extension, "cuda_sm90", "native"),
        "ours-ppu-source-check": build_receipt(args.ppu_source_extension, "ppu17", "source-check"),
    }
    watch = DeviceWatch(0)
    result = dict(status="INCOMPLETE", scope="H800_CONTROL_NOT_NATIVE_PPU17",
                  protocol="NSYS_KERNEL_SUM_PER_SYNCHRONIZED_NVTX_FORWARD",
                  comparison_family=args.family, samples=12, calls=[], shape=[*SHAPE, 128],
                  gate=args.gate, source=source, versions=versions(), incumbent_builds=builds,
                  harness_sha256=sha(__file__), utc=datetime.now(timezone.utc).isoformat())
    try:
        for _ in range(3):
            watch.sample(idle=True)
            time.sleep(.2)
        watch.thread.start()
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (9, 0) or "PPU" in props.name:
            raise RuntimeError("physical CUDA Hopper required; not a simulator runner")
        result.update(device=props.name, sms=props.multi_processor_count, cuda=torch.version.cuda)
        torch.set_num_threads(1)
        cpu = fixture(*SHAPE, args.gate)
        q, k, v, g, beta = cpu
        want = torch_recurrent_gated_delta_rule(
            q.repeat_interleave(2, 2), k.repeat_interleave(2, 2), v, g, beta,
            output_final_state=True)
        tensors = tuple(t.cuda() for t in cpu)
        result.update(input_sha256=digest(cpu), reference_sha256=digest(want))

        def ours(path, backend, source_check):
            os.environ["GDN_QSA_SM90_EXTENSION"] = str(path.resolve())
            role = "ours-ppu-source-check" if source_check else "ours-cuda"
            return gdn_chunk_sm90(*tensors, backend=backend, source_check=source_check,
                                 configuration=builds[role].get("configuration"))

        calls = {"ours-cuda": lambda: ours(args.cuda_extension, "cuda_sm90", False),
                 "ours-ppu-source-check": lambda: ours(args.ppu_source_extension, "ppu17", True)}
        sys.path.insert(0, str(args.reference_root))
        if args.family == "flashqla":
            from flash_qla import chunk_gated_delta_rule as ref
            from flash_qla.ops.gated_delta_rule.chunk.cp_context import _calc_cp_seqs
            cu = torch.tensor([0, 2048], dtype=torch.int32, device="cuda")
            cp = _calc_cp_seqs(cu, 64, 32)
            result["route"] = dict(auto_cp=bool(cp[0]), cp_offsets=cp[1].cpu().tolist() if cp[0] else None,
                                   state_layout="KV", backward_cache=False)
            for name, flag in (("auto", True), ("no-cp", False)):
                calls[f"flashqla-{name}"] = lambda flag=flag: ref(
                    *tensors, scale=128**-.5, initial_state=None, output_final_state=True,
                    use_qk_l2norm_in_kernel=False, state_v_first=False, auto_cp=flag,
                    enable_fwd_cp_cache=False)
        else:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule as ref
            from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
            alpha_cpu = g.float().exp().squeeze(0).contiguous()
            beta_cpu = beta.float().squeeze(0).contiguous()
            alpha, beta_fp32 = alpha_cpu.cuda(), beta_cpu.cuda()
            flat = tuple(t.squeeze(0) for t in tensors[:3])
            cu = torch.tensor([0, 2048], dtype=torch.int64, device="cuda")
            result["route"] = dict(auto_cp=should_use_cp_host(32, props.multi_processor_count, props.name, (9, 0)),
                                   backend="auto", state_layout="VK", native_gate="FP32 exp(log-gate)",
                                   gate_log_roundtrip_max=float((alpha_cpu.log()-g.float().squeeze(0)).abs().max()),
                                   adapter_inputs_sha256=digest((alpha_cpu, beta_cpu)))

            def infer(cp, adapter=False):
                a = tensors[3].float().exp().squeeze(0) if adapter else alpha
                b = tensors[4].float().squeeze(0) if adapter else beta_fp32
                return ref(*flat, g=a, beta=b, scale=128**-.5, initial_state=None,
                           output_final_state=True, cu_seqlens=cu,
                           use_qk_l2norm_in_kernel=False, use_cp=cp)

            calls.update({"flashinfer-auto": lambda: infer("auto"),
                          "flashinfer-no-cp": lambda: infer(False),
                          "flashinfer-auto-log-adapter": lambda: infer("auto", True)})
        if tuple(calls) != LIBRARY_ROLES[args.family]:
            raise AssertionError("role denominator changed")

        def checked(role, pair):
            actual = tuple(t.detach().cpu() for t in pair)
            if role.startswith("flashinfer"):
                actual = (actual[0].unsqueeze(0), actual[1].transpose(-2, -1).contiguous())
            if tuple(t.shape for t in actual) != tuple(t.shape for t in want):
                raise AssertionError("output/state ABI shape mismatch")
            if actual[0].dtype != torch.bfloat16 or actual[1].dtype != torch.float32:
                raise AssertionError("output/state dtype changed")
            return assert_pair(actual, want), digest(actual)

        # Independent CPU seam negatives: same shape, deliberately wrong semantics.
        wrong_gate = torch_recurrent_gated_delta_rule(
            q.repeat_interleave(2, 2), k.repeat_interleave(2, 2), v, g.float().exp(), beta,
            output_final_state=True)
        for name, bad in (("exp-treated-as-log", wrong_gate),
                          ("state-transpose", (want[0], want[1].transpose(-2, -1))),
                          ("zero-output", (torch.zeros_like(want[0]), want[1]))):
            try:
                assert_pair(bad, want)
            except AssertionError:
                print(f"[SM90 library negative] {name} EXPECTED_RED/PASS", flush=True)
            else:
                raise AssertionError(f"numeric seam negative escaped: {name}")

        admission = {}
        for role, call in calls.items():
            print(f"[SM90 library compile/admit] role={role}", flush=True)
            errors, fingerprint = checked(role, call())
            for _ in range(8):
                _, again = checked(role, call())
                if again != fingerprint:
                    raise AssertionError(f"{role}: repeated result unstable")
            admission[role] = dict(errors=errors, fingerprint=fingerprint, repeat="8/8 RAW-BIT")
            print(f"[SM90 library admission] role={role} {admission[role]}", flush=True)
        result["admission"] = admission
        result["reference_binaries"] = reference_binaries(args.family, args.out)
        torch.cuda.synchronize()
        if watch.errors:
            raise RuntimeError(f"concurrency monitor invalidates preflight: {watch.errors}")
        if not args.preflight_only:
            watch.sample()
            torch.cuda.cudart().cudaProfilerStart()
            outputs = []
            try:
                roles = tuple(calls)
                for sample in range(12):
                    order = roles[sample % len(roles):] + roles[:sample % len(roles)]
                    if sample % 2:
                        order = tuple(reversed(order))
                    for role in order:
                        label = f"GDN_FORWARD|{role}|{sample:03d}"
                        torch.cuda.nvtx.range_push(label)
                        try:
                            output = calls[role]()
                            torch.cuda.synchronize()
                        finally:
                            torch.cuda.nvtx.range_pop()
                        outputs.append((role, output))
                        result["calls"].append(label)
            finally:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            for role, output in outputs:
                _, fingerprint = checked(role, output)
                if fingerprint != admission[role]["fingerprint"]:
                    raise AssertionError(f"{role}: captured result changed")
        watch.sample()
        watch.stop.set()
        watch.thread.join(timeout=22)
        if watch.thread.is_alive() or watch.errors:
            raise RuntimeError(f"concurrency monitor invalidates measurement: {watch.errors}")
        result["status"] = "PREFLIGHT_PASS" if args.preflight_only else "CAPTURE_COMPLETE_AWAIT_NSYS_EXTRACTION"
        print(f"[SM90 library nsys] {result['status']} calls={len(result['calls'])}", flush=True)
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        watch.stop.set()
        if watch.thread.is_alive():
            watch.thread.join(timeout=22)
        result["device_watch"] = dict(records=watch.records, errors=watch.errors)
        (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
