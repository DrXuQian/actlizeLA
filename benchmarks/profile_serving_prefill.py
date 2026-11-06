#!/usr/bin/env python3
"""One immutable serving/retained/FLA call for external ACU; no profiler API.

Admission creates processed operands with the actual serving prepare, outside
capture. Core comparisons share those exact bytes and a KV FP32 initial state.
The serving call includes raw-logit preparation, packing and VK slot transport.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests", ROOT / "tools"):
    sys.path.insert(0, str(path))

from actlize_la.serving import NativeLibrary, PreparedGDN, gdn_workspace_specs
from actlize_la.serving.device import current_device_identity
from bench_ppu_gdn_fla import checked_pair, fla_call, load_fla
from collect_ppu_gdn_acu import sha
from profile_ppu_gdn_fla import save_fla_sources
from serving_device import close
from serving_reference import gdn_reference
from test_ppu_wy_backend import oracle as core_reference

ROLES = ("serving", "retained", "fla")


def digest(tensors):
    result = hashlib.sha256()
    for value in tensors:
        value = value.detach().cpu().contiguous()
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def raw_fixture(length, hv, gate):
    if length < 1 or hv not in (32, 48) or gate not in (-.1, -1.):
        raise ValueError("unsupported synthetic fixture")
    generator = torch.Generator().manual_seed(1931 + length + hv)
    mixed = torch.randn(length, (32 + hv) * 128, generator=generator).bfloat16()
    a = torch.full((length, hv), math.log(math.expm1(-gate))).bfloat16()
    b = torch.randn(length, hv, generator=generator).bfloat16()
    state = torch.zeros(3, hv, 128, 128, dtype=torch.float32)
    state[1] = torch.randn(hv, 128, 128, generator=generator) * .02
    return dict(mixed=mixed, a=a, b=b, a_log=torch.zeros(hv, dtype=torch.float32),
                bias=torch.zeros(hv, dtype=torch.float32),
                state=state, starts=torch.tensor([0, length], dtype=torch.int32),
                live=torch.tensor([1], dtype=torch.int32),
                initial_slots=torch.tensor([1]), final_slots=torch.tensor([2]))


def raw_reference(raw):
    output, state, *_ = gdn_reference(raw["mixed"], raw["a"], raw["b"],
                                    raw["a_log"], raw["bias"], raw["state"][1], 16)
    return output[None], state.transpose(-1, -2).contiguous()[None]


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
        length, hv = raw["a"].shape
        self.output = torch.empty((length, hv, 128), dtype=torch.bfloat16, device="cuda")
        self.work = {name: torch.empty(shape, dtype=dtype, device="cuda") for name, (shape, dtype)
                     in gdn_workspace_specs(1, length, 16, hv, mode="prefill", target="ppu10").items()}
        self.plan = PreparedGDN(
            library, mixed=self.raw["mixed"], a=self.raw["a"], b=self.raw["b"],
            a_log=self.raw["a_log"], dt_bias=self.raw["bias"], state=self.raw["state"],
            starts=self.raw["starts"], live=self.raw["live"],
            initial_slots=self.raw["initial_slots"], final_slots=self.raw["final_slots"],
            output=self.output, workspace=self.work, width=length,
            mode="prefill", key_heads=16, value_heads=hv)

    def __call__(self):
        self.plan()
        # Views only; all transpose/materialization/checking is on CPU later.
        return self.output, self.raw["state"][2]

    def cpu_pair(self, result):
        output, state = (value.detach().cpu() for value in result)
        if self.work["errors"].cpu().any():
            raise AssertionError("serving device validation rejected the request")
        if digest((self.raw["state"][:2],)) != digest((self.raw_cpu["state"][:2],)):
            raise AssertionError("committed input state or null slot mutated")
        return output[None], state.transpose(-1, -2).contiguous()[None]

    def packed_cpu(self):
        return tuple(self.work[name].cpu() for name in ("q", "k", "v", "log_g", "beta"))


class RetainedCall:
    def __init__(self, library, packed, initial):
        self.inputs = tuple(value.cuda() for value in packed)
        self.initial = initial.cuda()
        batch, length, hv, _ = packed[2].shape
        self.output = torch.empty_like(self.inputs[2])
        self.final = torch.empty_like(self.initial)
        specs = gdn_workspace_specs(batch, length, 16, hv, mode="prefill", target="ppu10")
        self.work = {name: torch.empty(*specs[name][0], dtype=specs[name][1], device="cuda")
                     for name in ("inverse", "snapshots", "vnew", "gates")}
        self.fn = library.lib.actlize_la_ppu10_forward_prepared
        self.fn.argtypes = [C.c_void_p] * 12 + [C.c_int] * 4 + [C.c_bool, C.c_void_p]
        self.fn.restype = C.c_int
        library.lib.actlize_la_ppu10_configure.restype = C.c_int
        rc = library.lib.actlize_la_ppu10_configure()
        if rc:
            raise RuntimeError(f"retained configure failed: {rc}")
        self.library = library
        tensors = (*self.inputs, self.initial, self.output, self.final,
                   *(self.work[name] for name in ("inverse", "snapshots", "vnew", "gates")))
        self.args = [value.data_ptr() for value in tensors] + [batch, length, 16, hv, True]

    def __call__(self):
        rc = self.fn(*self.args, torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"retained prepared launch failed: {rc}")
        return self.output, self.final

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
                      + (call.raw["state"][:2],))
    return digest((*call.inputs, call.initial))


def admit(library, length, hv, gate, directory):
    raw = raw_fixture(length, hv, gate)
    want_raw = raw_reference(raw)
    serving = ServingCall(library, raw)
    first = capture(serving)
    checked(first, want_raw, "serving")
    packed, initial = serving.packed_cpu(), serving.work["initial_kv"].cpu()
    want_core = core_reference(packed, initial)
    calls = {"serving": serving, "retained": RetainedCall(library, packed, initial),
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
    if records["serving"]["output_sha"] != records["retained"]["output_sha"]:
        raise AssertionError("serving vs retained output/state RAW-BIT mismatch")
    payload = dict(raw=raw, packed=packed, initial=initial, want_raw=want_raw, want_core=want_core)
    torch.save(payload, directory / "operands.pt")
    sources = save_fla_sources(directory / "fla-sources")
    return dict(arms=records, operands_sha=sha(directory / "operands.pt"),
                packed_input_sha=digest((*packed, initial)), fla=calls["fla"].identity,
                fla_sources=sources, requested_gate=gate,
                actual_log_g_range=[float(packed[3].min()), float(packed[3].max())],
                serving_core_bits="PASS", numeric_repeats=8,
                workspace_bytes=sum(value.numel() * value.element_size() for value in serving.work.values()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--phase", choices=("admit", "subject"), required=True)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--length", type=int, choices=(2048, 8192), required=True)
    parser.add_argument("--value-heads", type=int, choices=(32, 48), required=True)
    parser.add_argument("--gate", type=float, choices=(-.1, -1.), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    record = dict(phase=args.phase, role=args.role, status="FAIL", api_timing="NOT_RUN",
                  measurement_owner="external-ACU-collector")
    try:
        if not torch.cuda.is_available():
            record.update(status="SKIP", reason="no device; no numerical/performance admission")
            print("[serving prefill] SKIP: no device; performance NOT_RUN", flush=True)
            return 3
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        record.update(device=current_device_identity("ppu10"), library_sha=sha(args.library),
                      shape=dict(B=1, T=args.length, Hk=16, Hv=args.value_heads, K=128, V=128),
                      gate=args.gate, precision="BF16-QKV/beta; FP32-log-g/state",
                      initial_state="asymmetric-fp32", capture="external-ACU-no-profiler-API")
        record["device"]["uuid"] = str(getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE"))
        if args.phase == "admit":
            if args.role is not None:
                raise ValueError("admit checks all three roles")
            record.update(admit(NativeLibrary(args.library, target="ppu10"), args.length,
                                args.value_heads, args.gate, args.directory))
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
            if args.role == "serving":
                call = ServingCall(NativeLibrary(args.library, target="ppu10"), payload["raw"])
            elif args.role == "retained":
                call = RetainedCall(NativeLibrary(args.library, target="ppu10"), payload["packed"], payload["initial"])
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
