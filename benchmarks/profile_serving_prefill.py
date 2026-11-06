#!/usr/bin/env python3
"""One immutable serving/default/FLA call for external ACU; no profiler API.

Admission creates processed operands with the actual serving prepare, outside
capture. Core comparisons share those exact bytes and a KV FP32 initial state.
The serving call includes raw-logit preparation, packing and VK slot transport.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests", ROOT / "tools"):
    sys.path.insert(0, str(path))

from actlize_la import gdn_forward, get_gdn_route_info
from actlize_la.serving import NativeLibrary, PreparedGDN, gdn_workspace_specs
from actlize_la.serving.device import current_device_identity
from bench_ppu_gdn_fla import checked_pair, fla_call, load_fla
from collect_ppu_gdn_acu import sha
from collect_serving_prefill_acu import ROLES, validate_default_route
from profile_ppu_gdn_fla import save_fla_sources
from serving_device import close
from serving_reference import gdn_reference
from test_ppu_wy_backend import oracle as core_reference

def digest(tensors):
    result = hashlib.sha256()
    for value in tensors:
        value = value.detach().cpu().contiguous()
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


LENGTHS = (512, 2048, 4096, 8192, 16384)
VALUE_HEADS = (32, 48, 64)
BATCHES = (1, 2, 4)
GATES = (-.1, -1.)


def raw_fixture(length, hv, gate, batch=1):
    """B equal-length packed requests; slot 0 null, initial 1..B, final B+1..2B.

    B=1 keeps the historical seed and draw order, so its bytes are unchanged.
    """
    if (length < 1 or batch not in BATCHES or hv not in VALUE_HEADS or gate not in GATES):
        raise ValueError("unsupported synthetic fixture")
    generator = torch.Generator().manual_seed(1931 + length + hv + 100003 * (batch - 1))
    tokens = batch * length
    mixed = torch.randn(tokens, (32 + hv) * 128, generator=generator).bfloat16()
    a = torch.full((tokens, hv), math.log(math.expm1(-gate))).bfloat16()
    b = torch.randn(tokens, hv, generator=generator).bfloat16()
    state = torch.zeros(2 * batch + 1, hv, 128, 128, dtype=torch.float32)
    for row in range(batch):
        state[1 + row] = torch.randn(hv, 128, 128, generator=generator) * .02
    return dict(mixed=mixed, a=a, b=b, a_log=torch.zeros(hv, dtype=torch.float32),
                bias=torch.zeros(hv, dtype=torch.float32), state=state,
                starts=torch.arange(batch + 1, dtype=torch.int32) * length,
                live=torch.tensor([batch], dtype=torch.int32),
                initial_slots=torch.arange(1, batch + 1),
                final_slots=torch.arange(batch + 1, 2 * batch + 1))


def request_geometry(raw):
    """Requests and their common width from the actual packed metadata."""
    starts = [int(x) for x in raw["starts"]]
    widths = {hi - lo for lo, hi in zip(starts, starts[1:])}
    if len(starts) < 2 or len(widths) != 1 or starts[0] != 0 or int(raw["live"][0]) != len(starts) - 1:
        raise ValueError("performance fixture requires equal-length live packed requests")
    return len(starts) - 1, widths.pop()


def raw_reference(raw):
    starts = [int(x) for x in raw["starts"]]
    outputs, states = [], []
    for row, (lo, hi) in enumerate(zip(starts, starts[1:])):
        output, state, *_ = gdn_reference(raw["mixed"][lo:hi], raw["a"][lo:hi], raw["b"][lo:hi],
                                          raw["a_log"], raw["bias"],
                                          raw["state"][int(raw["initial_slots"][row])], 16)
        outputs.append(output)
        states.append(state.transpose(-1, -2).contiguous())
    return torch.stack(outputs), torch.stack(states)


def core_shape(packed):
    """Measured geometry from the actual processed tensors, never a CLI label."""
    q, k, v, log_g, beta = packed
    b, t, hk, kd = q.shape
    if (k.shape != q.shape or v.shape[:2] != (b, t) or v.shape[3] != 128 or kd != 128
            or log_g.shape != v.shape[:3] or beta.shape != v.shape[:3]):
        raise ValueError("inconsistent processed operand geometry")
    return dict(B=b, T=t, Hk=hk, Hv=v.shape[2], K=kd, V=v.shape[3])


def checked(result, reference, role):
    if len(result) != 2 or any(a.shape != b.shape for a, b in zip(result, reference)):
        raise AssertionError("output AND state geometry must match the reference")
    if result[0].dtype != torch.bfloat16 or result[1].dtype != torch.float32:
        raise AssertionError("require BF16 output and FP32 carried state")
    if role == "serving":
        return [close(a, b, chunk=True) for a, b in zip(result, reference)]
    return checked_pair(result, reference)


class ServingCall:
    def __init__(self, library, raw):
        self.raw_cpu = raw
        self.raw = {name: value.cuda() for name, value in raw.items()}
        self.requests, self.width = request_geometry(raw)
        hv = raw["a"].shape[1]
        self.final = slice(self.requests + 1, 2 * self.requests + 1)
        if [int(x) for x in raw["final_slots"]] != list(range(self.final.start, self.final.stop)):
            raise ValueError("final slots must be the contiguous B+1..2B block")
        self.output = torch.empty((self.requests * self.width, hv, 128), dtype=torch.bfloat16, device="cuda")
        self.work = {name: torch.empty(shape, dtype=dtype, device="cuda") for name, (shape, dtype)
                     in gdn_workspace_specs(self.requests, self.width, 16, hv, mode="prefill", target="ppu10").items()}
        self.plan = PreparedGDN(
            library, mixed=self.raw["mixed"], a=self.raw["a"], b=self.raw["b"],
            a_log=self.raw["a_log"], dt_bias=self.raw["bias"], state=self.raw["state"],
            starts=self.raw["starts"], live=self.raw["live"],
            initial_slots=self.raw["initial_slots"], final_slots=self.raw["final_slots"],
            output=self.output, workspace=self.work, width=self.width,
            mode="prefill", key_heads=16, value_heads=hv)

    def __call__(self):
        self.plan()
        # Views only; all transpose/materialization/checking is on CPU later.
        return self.output, self.raw["state"][self.final]

    def committed(self):
        """Null slot and every request's committed initial slot (never written)."""
        return self.raw["state"][:self.requests + 1]

    def cpu_pair(self, result):
        output, state = (value.detach().cpu() for value in result)
        if self.work["errors"].cpu().any():
            raise AssertionError("serving device validation rejected the request")
        if digest((self.committed(),)) != digest((self.raw_cpu["state"][:self.requests + 1],)):
            raise AssertionError("committed input state or null slot mutated")
        hv = output.shape[1]
        return (output.reshape(self.requests, self.width, hv, 128),
                state.transpose(-1, -2).contiguous())

    def packed_cpu(self):
        return tuple(self.work[name].cpu() for name in ("q", "k", "v", "log_g", "beta"))


class DefaultCall:
    """Measure the public default, not a hand-selected benchmark variant."""

    def __init__(self, library, packed, initial):
        self.inputs = tuple(value.cuda() for value in packed)
        self.initial = initial.cuda()
        if "GDN_QSA_WY_EXTENSION" in os.environ:
            raise ValueError("unset GDN_QSA_WY_EXTENSION; this measures the serving-backed public default")
        selected = os.environ.get("ACTLIZE_LA_SERVING_LIBRARY")
        if selected is not None and Path(selected).resolve() != library.path.resolve():
            raise ValueError("ACTLIZE_LA_SERVING_LIBRARY differs from the admitted library")
        os.environ["ACTLIZE_LA_SERVING_LIBRARY"] = str(library.path.resolve())
        self.route = get_gdn_route_info(*self.inputs, initial_state=self.initial,
                                        output_final_state=True, backend="ppu10")
        validate_default_route(self.route, library.path, sha(library.path))

    def __call__(self):
        # Deliberately omit algorithm/delivery. A changed default must fail the
        # route gate, not be hidden by an explicitly forced fast variant.
        return gdn_forward(*self.inputs, initial_state=self.initial,
                           output_final_state=True, backend="ppu10")

    def cpu_pair(self, result):
        return tuple(value.detach().cpu() for value in result)


class FlaCall:
    def __init__(self, packed, initial):
        fn, self.identity = load_fla()
        self.inputs = tuple(value.cuda() for value in packed)
        self.initial = initial.cuda()
        self.call = fla_call(fn, self.inputs, "native", initial_state=self.initial)

    def __call__(self):
        return self.call()

    def cpu_pair(self, result):
        return tuple(value.detach().cpu() for value in result)


def capture(call):
    torch.cuda.synchronize()
    result = call()
    torch.cuda.synchronize()
    return call.cpu_pair(result)


def input_fingerprint(call):
    if isinstance(call, ServingCall):
        return digest(tuple(value for name, value in call.raw.items() if name != "state")
                      + (call.committed(),))
    return digest((*call.inputs, call.initial))


def admit(library, length, hv, gate, directory, batch=1):
    raw = raw_fixture(length, hv, gate, batch)
    want_raw = raw_reference(raw)
    serving = ServingCall(library, raw)
    first = capture(serving)
    checked(first, want_raw, "serving")
    packed, initial = serving.packed_cpu(), serving.work["initial_kv"].cpu()
    measured = core_shape(packed)
    if tuple(initial.shape) != (measured["B"], measured["Hv"], 128, 128):
        raise AssertionError("initial KV state geometry differs from the processed operands")
    want_core = core_reference(packed, initial)
    calls = {"serving": serving, "default": DefaultCall(library, packed, initial),
             "fla": FlaCall(packed, initial)}
    records = {}
    for role, call in calls.items():
        source_hash, fingerprints, errors = input_fingerprint(call), [], []
        for _ in range(8):
            result = capture(call)
            errors = checked(result, want_raw if role == "serving" else want_core, role)
            fingerprints.append(digest(result))
        if len(set(fingerprints)) != 1 or input_fingerprint(call) != source_hash:
            raise AssertionError(f"{role}: input mutation or unstable repeated result")
        records[role] = dict(errors=errors, output_sha=fingerprints[0], repeats=8,
                             input_sha=source_hash, state_dtype="torch.float32")
    if records["serving"]["output_sha"] != records["default"]["output_sha"]:
        raise AssertionError("serving vs public default output/state RAW-BIT mismatch")
    payload = dict(raw=raw, packed=packed, initial=initial, want_raw=want_raw, want_core=want_core)
    torch.save(payload, directory / "operands.pt")
    sources = save_fla_sources(directory / "fla-sources")
    return dict(arms=records, operands_sha=sha(directory / "operands.pt"),
                packed_input_sha=digest((*packed, initial)), fla=calls["fla"].identity,
                fla_sources=sources, requested_gate=gate, measured_shape=measured,
                actual_log_g_range=[float(packed[3].min()), float(packed[3].max())],
                serving_default_bits="PASS", numeric_repeats=8,
                default_route=calls["default"].route,
                workspace_bytes=sum(value.numel() * value.element_size() for value in serving.work.values()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--phase", choices=("admit", "subject"), required=True)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--batch", type=int, choices=BATCHES, default=1)
    parser.add_argument("--length", type=int, choices=LENGTHS, required=True)
    parser.add_argument("--value-heads", type=int, choices=VALUE_HEADS, required=True)
    parser.add_argument("--gate", type=float, choices=GATES, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    record = dict(phase=args.phase, role=args.role, status="FAIL", api_timing="NOT_RUN",
                  measurement_owner="external-ACU-collector")
    requested = dict(B=args.batch, T=args.length, Hk=16, Hv=args.value_heads, K=128, V=128)
    try:
        if not torch.cuda.is_available():
            record.update(status="SKIP", reason="no device; no numerical/performance admission")
            print("[serving prefill] SKIP: no device; performance NOT_RUN", flush=True)
            return 3
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        record.update(device=current_device_identity("ppu10"), library_sha=sha(args.library),
                      shape=requested,
                      gate=args.gate, precision="BF16-QKV/beta; FP32-log-g/state",
                      initial_state="asymmetric-fp32", capture="external-ACU-no-profiler-API")
        record["device"]["uuid"] = str(getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE"))
        if args.phase == "admit":
            if args.role is not None:
                raise ValueError("admit checks all three roles")
            record.update(admit(NativeLibrary(args.library, target="ppu10"), args.length,
                                args.value_heads, args.gate, args.directory, args.batch))
            if record["measured_shape"] != requested:
                raise ValueError(f"measured operands {record['measured_shape']} differ from the requested cell")
        else:
            if args.role is None:
                raise ValueError("subject requires exactly one role")
            admission = json.loads((args.directory / "admission.json").read_text())
            for name in ("device", "library_sha", "shape", "gate"):
                if record[name] != admission[name]:
                    raise ValueError(f"subject/preflight identity changed: {name}")
            operands = args.directory / "operands.pt"
            if admission["status"] != "PASS" or sha(operands) != admission["operands_sha"]:
                raise ValueError("missing admission or changed operands")
            payload = torch.load(operands, map_location="cpu", weights_only=True)
            measured = core_shape(payload["packed"])
            if measured != requested or measured != admission.get("measured_shape"):
                raise ValueError(f"subject operands {measured} differ from the admitted cell")
            record["measured_shape"] = measured
            if args.role == "serving":
                call = ServingCall(NativeLibrary(args.library, target="ppu10"), payload["raw"])
            elif args.role == "default":
                call = DefaultCall(NativeLibrary(args.library, target="ppu10"), payload["packed"], payload["initial"])
                if call.route != admission["default_route"]:
                    raise ValueError("public default route or loaded image changed")
                record["default_route"] = call.route
            else:
                call = FlaCall(payload["packed"], payload["initial"])
                if call.identity != admission["fla"]:
                    raise ValueError("FLA source/compiler identity changed")
            before = input_fingerprint(call)
            # Exactly one measured public call; CPU checks never add device kernels.
            result = capture(call)
            errors = checked(result, payload["want_raw" if args.role == "serving" else "want_core"], args.role)
            expected = admission["arms"][args.role]
            if before != expected["input_sha"] or input_fingerprint(call) != before:
                raise AssertionError("subject input differs or was mutated")
            if digest(result) != expected["output_sha"]:
                raise AssertionError("subject result differs from the admitted eight repeats")
            record.update(public_calls=1, warmup=0, verification="CPU", errors=errors,
                          output_sha=digest(result), input_sha=before,
                          operands_sha=admission["operands_sha"])
        record["status"] = "PASS"
        print("[serving prefill admission] " + json.dumps(record, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        record.update(error=repr(error))
        raise
    finally:
        args.receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
