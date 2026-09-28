#!/usr/bin/env python3
"""Same-input physical Hopper comparison; timing authority is the nsys trace.

No event/API latency substitutes for GPU kernel duration. cuLA is the original
C++ Hopper entry, including its original gate preprocessing and final-state
initialization. Warmup, input adaptation and CPU checking are outside capture.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "benchmarks")]
import torch
from bench_sm90_hopper import DeviceWatch
from test_ppu_gdn_backend import fixture, assert_pair, digest
from actlize_la.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from actlize_la.gdn_sm90_interface import gdn_chunk_sm90

PIN = "79be249e61453808e18e5cef7702b363239e7d8d"
SHAPE = (1, 2048, 16, 32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_receipt(path, target, mode):
    receipt = json.loads((path.parent / "build.json").read_text())
    if (not receipt.get("complete") or receipt.get("extension_sha256") != sha(path)
            or receipt.get("target") != target or receipt.get("mode") != mode):
        raise RuntimeError("incumbent binary/build identity mismatch")
    return receipt


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-extension", type=Path, required=True)
    p.add_argument("--ppu-source-extension", type=Path, required=True)
    p.add_argument("--cula", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--gate", type=float, choices=(-.1, -1.), required=True)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.samples < 3:
        p.error("need >=3 samples")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "receipt.json").exists():
        raise RuntimeError("refuse to overwrite a previous measurement receipt")
    if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("require the unremapped single-Hopper host")
    receipts = {
        "ours-cuda": build_receipt(args.cuda_extension, "cuda_sm90", "native"),
        "ours-ppu-source-check": build_receipt(args.ppu_source_extension, "ppu17", "source-check"),
    }
    if git(args.cula, "rev-parse", "HEAD") != PIN or git(args.cula, "diff", "HEAD", "--", "csrc", "cula"):
        raise RuntimeError("cuLA must be the pinned, unmodified upstream source")
    sys.path[:0] = [str(args.cula), str(args.cula / "third_party/flash-linear-attention")]
    watch = DeviceWatch(0)
    result = {"scope": "H800_CONTROL_NOT_NATIVE_PPU17", "status": "INCOMPLETE",
              "protocol": "NSYS_KERNEL_SUM_PER_SYNCHRONIZED_NVTX_FORWARD",
              "shape": [*SHAPE, 128], "gate": args.gate, "calls": [],
              "incumbent_builds": receipts, "harness_sha256": sha(__file__),
              "state_layout": "ours=KV; cuLA=VK; transpose CPU-side for comparison",
              "input_adapter": "CPU scalar-gate broadcast to contiguous BF16 [B,T,Hv,K], outside forward",
              "cula_pin": PIN,
              "cula_submodules": git(args.cula, "submodule", "status"),
              "utc": datetime.now(timezone.utc).isoformat()}
    try:
        for _ in range(3):
            watch.sample(idle=True)
            time.sleep(.2)
        watch.thread.start()
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (9, 0) or "PPU" in props.name:
            raise RuntimeError("this repeated runner is for physical CUDA Hopper only")
        result.update(device=props.name, sms=props.multi_processor_count,
                      torch=torch.__version__, cuda=torch.version.cuda)
        # Import only the explicitly selected original C++ wrapper, not auto-route/DSL.
        from cula.kda.hopper_fused_fwd import cula_kda_prefill
        import cula._cudac_sm90 as cula_binary
        result["cula_binary"] = dict(path=cula_binary.__file__, sha256=sha(cula_binary.__file__))
        result["cula_entry_sha256"] = sha(args.cula / "cula/kda/hopper_fused_fwd.py")
        torch.set_num_threads(1)
        cpu = fixture(*SHAPE, args.gate)
        q, k, v, g, beta = cpu
        vector_gate = g[..., None].expand(*g.shape, 128).contiguous()
        if not torch.equal(vector_gate[..., 0], g) or not torch.equal(
                vector_gate, g[..., None].expand_as(vector_gate)):
            raise AssertionError("gate adapter changed the input values")
        want = torch_recurrent_gated_delta_rule(
            q.repeat_interleave(2, 2), k.repeat_interleave(2, 2), v, g, beta,
            output_final_state=True)
        tensors = tuple(t.cuda() for t in cpu)
        gate_cuda = vector_gate.cuda()
        result["input_sha256"] = digest(cpu)
        result["cula_gate_sha256"] = digest((vector_gate,))
        result["reference_sha256"] = digest(want)

        def ours(path, backend, source_check):
            os.environ["GDN_QSA_SM90_EXTENSION"] = str(path.resolve())
            role = "ours-ppu-source-check" if source_check else "ours-cuda"
            return gdn_chunk_sm90(*tensors, backend=backend, source_check=source_check,
                                 configuration=receipts[role].get("configuration"))

        def cula():
            return cula_kda_prefill(
                tensors[0], tensors[1], tensors[2], gate_cuda, tensors[4],
                scale=128**-.5, initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=True)

        calls = {
            "ours-cuda": lambda: ours(args.cuda_extension, "cuda_sm90", False),
            "ours-ppu-source-check": lambda: ours(args.ppu_source_extension, "ppu17", True),
            "cula": cula,
        }

        def checked(role, pair):
            actual = tuple(t.detach().cpu() for t in pair)
            if role == "cula":
                actual = (actual[0], actual[1].transpose(-2, -1).contiguous())
            if tuple(t.shape for t in actual) != tuple(t.shape for t in want):
                raise AssertionError("output/state ABI shape mismatch")
            return assert_pair(actual, want), digest(actual)

        admission = {}
        for role, call in calls.items():
            errors, fingerprint = checked(role, call())
            for _ in range(8):
                _, again = checked(role, call())
                if again != fingerprint:
                    raise AssertionError(f"{role}: repeated result is not raw-bit stable")
            admission[role] = dict(errors=errors, fingerprint=fingerprint, repeat="8/8 RAW-BIT")
            print(f"[cuLA nsys admission] role={role} {admission[role]}", flush=True)
        # Each role has already run nine times: no compilation/warmup in capture.
        torch.cuda.synchronize()
        result["admission"] = admission
        if not args.preflight_only:
            watch.sample()
            torch.cuda.cudart().cudaProfilerStart()
            outputs = []
            try:
                roles = tuple(calls)
                for sample in range(args.samples):
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
                    raise AssertionError(f"{role}: profiled output differs from admitted output")
        watch.sample()
        watch.stop.set()
        watch.thread.join(timeout=22)
        if watch.thread.is_alive() or watch.errors:
            raise RuntimeError(f"concurrency monitor invalidates measurement: {watch.errors}")
        result["status"] = "PREFLIGHT_PASS" if args.preflight_only else "CAPTURE_COMPLETE_AWAIT_NSYS_EXTRACTION"
        print(f"[cuLA nsys] {result['status']} calls={len(result['calls'])}", flush=True)
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        watch.stop.set()
        if watch.thread.is_alive():
            watch.thread.join(timeout=22)
        result["device_watch"] = dict(records=watch.records, errors=watch.errors)
        (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
