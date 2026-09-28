#!/usr/bin/env python3
"""Physical Hopper-only timing. Never use this repeated runner in simulation.

CPU oracle and the existing 2% criterion precede measurement. GPU-process
sampling is fail-closed: observed concurrent work invalidates the entire run.
Sampling cannot exclude a foreign task shorter than the sampling interval;
reserve the device as well. No clock, power or compute-mode settings change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import torch
from test_ppu_gdn_backend import fixture, assert_pair, digest
from actlize_la.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from actlize_la.gdn_sm90_interface import gdn_chunk_sm90


def process_ids(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(not line.isdecimal() for line in lines):
        raise RuntimeError("unreadable GPU process list; no idle-device admission")
    return set(map(int, lines))


def own_pids():
    result = {os.getpid()}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("NSpid:"):
            result.update(map(int, line.split()[1:]))
    return result


class DeviceWatch:
    def __init__(self, device):
        self.device = device
        self.allowed = own_pids()
        self.records = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self, idle=False):
        cmd = ["nvidia-smi", "-i", str(self.device),
               "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        pids = process_ids(subprocess.check_output(cmd, text=True, timeout=10))
        fields = "uuid,utilization.gpu,clocks.sm,power.draw,memory.used"
        telemetry = subprocess.check_output(
            ["nvidia-smi", "-i", str(self.device), f"--query-gpu={fields}",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
        if len(telemetry.splitlines()) != 1:
            raise RuntimeError("expected exactly one physical GPU")
        row = dict(utc=datetime.now(timezone.utc).isoformat(), pids=sorted(pids),
                   telemetry_fields=fields, telemetry=telemetry)
        self.records.append(row)
        if pids - self.allowed or (idle and pids):
            raise RuntimeError(f"GPU is not exclusive to this run; observed PIDs={sorted(pids)}")
        if idle and float(telemetry.split(",")[1].strip()) != 0:
            raise RuntimeError("GPU utilization is nonzero before measurement")

    def loop(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as error:
                self.errors.append(str(error))
                return
            self.stop.wait(.2)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--backend", choices=("cuda_sm90", "ppu17"), required=True)
    p.add_argument("--source-check", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate", type=float, choices=(-.1, -1.), default=-.1)
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--calls", type=int, default=20)
    args = p.parse_args()
    if args.samples < 3 or not 1 <= args.calls <= 100:
        p.error("need samples>=3 and 1<=calls<=100")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.extension.parent / "build.json").read_text())
    binary_hash = hashlib.sha256(args.extension.read_bytes()).hexdigest()
    mode = "source-check" if args.source_check else "native"
    if (not manifest.get("complete") or manifest.get("extension_sha256") != binary_hash
            or manifest.get("target") != args.backend or manifest.get("mode") != mode):
        raise RuntimeError("build receipt/binary/target mismatch")
    os.environ["GDN_QSA_SM90_EXTENSION"] = str(args.extension.resolve())
    # Physical-device ordinal is explicit. Require the unremapped single-GPU host.
    if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("use the unremapped single-Hopper host for this runner")
    watch = DeviceWatch(0)
    try:
        for _ in range(3):
            watch.sample(idle=True)
            time.sleep(.2)
        watch.thread.start()
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor) != (9, 0) or "PPU" in props.name:
            raise RuntimeError("physical CUDA Hopper only; not PPU or a simulator")
        torch.set_num_threads(1)
        cpu = fixture(1, 2048, 16, 32, args.gate)
        q, k, v, g, beta = cpu
        want = torch_recurrent_gated_delta_rule(
            q.repeat_interleave(2, 2), k.repeat_interleave(2, 2), v, g, beta,
            output_final_state=True)
        tensors = tuple(t.cuda() for t in cpu)

        def call():
            return gdn_chunk_sm90(*tensors, backend=args.backend,
                                  source_check=args.source_check, configuration=manifest.get("configuration"))

        anchor = call()
        torch.cuda.synchronize()
        anchor = tuple(t.cpu() for t in anchor)
        errors = assert_pair(anchor, want)
        fingerprint = digest(anchor)
        for _ in range(8):
            actual = tuple(t.cpu() for t in call())
            assert_pair(actual, want)
            if digest(actual) != fingerprint:
                raise AssertionError("8-launch raw-bit repeat instability")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = [call() for _ in range(args.calls)]
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        raw = []
        for _ in range(args.samples):
            watch.sample()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            raw.append(start.elapsed_time(end) * 1000 / args.calls)
        for output in captured:
            actual = tuple(t.cpu() for t in output)
            assert_pair(actual, want)
            if digest(actual) != fingerprint:
                raise AssertionError("captured output differs from admitted eager output")
        watch.sample()
        watch.stop.set()
        watch.thread.join(timeout=22)
        if watch.thread.is_alive():
            raise RuntimeError("GPU process monitor did not terminate cleanly")
        if watch.errors:
            raise RuntimeError("invalid measurement window: " + "; ".join(watch.errors))
        result = dict(scope="HOPPER_CONTROL_NOT_PPU17", protocol="CUDA_GRAPH_EVENT_US_PER_FORWARD",
            shape=[1, 2048, 16, 32, 128], gate=args.gate, backend=args.backend,
            source_check=args.source_check, device=props.name, sms=props.multi_processor_count,
            errors=errors, repeat="8/8 RAW-BIT", fingerprint=fingerprint,
            samples_us=raw, median_us=statistics.median(raw), calls_per_graph=args.calls,
            observed_range_us=[min(raw), max(raw)], binary_sha256=binary_hash,
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256=manifest["source_sha256"],
            input_sha256=digest(cpu), torch=torch.__version__, cuda=torch.version.cuda,
            concurrency="NO_FOREIGN_PROCESS_OBSERVED; polling=0.2s, not scheduler exclusion",
            verdict="NUMERICS_AND_TIMING_COMPLETE_NOT_COMPARATIVE_SPEED_ADMISSION")
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[SM90 Hopper perf] " + json.dumps(result), flush=True)
    except Exception as error:
        watch.errors.append(str(error))
        raise
    finally:
        watch.stop.set()
        if watch.thread.is_alive():
            watch.thread.join(timeout=22)
        (args.out / "device-watch.json").write_text(json.dumps(
            dict(records=watch.records, errors=watch.errors), indent=2) + "\n")


if __name__ == "__main__":
    main()
