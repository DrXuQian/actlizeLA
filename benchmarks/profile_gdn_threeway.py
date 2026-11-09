#!/usr/bin/env python3
"""One default/PLA/FLA core call using the previous ZW810 run's exact operands."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tools", ROOT / "tests", ROOT / "benchmarks"):
    sys.path.insert(0, str(directory))

from actlize_la.serving import NativeLibrary
from actlize_la.serving.device import current_device_identity
from collect_ppu_gdn_acu import sha
from gdn_threeway_contract import CHUNKS, ROLES, cells, geometry, validate_pla
from check_ppu810_gdn_recheck import validate_810
from profile_serving_prefill import (
    DefaultCall, FlaCall, capture, checked, core_shape, digest, input_fingerprint,
)
from profile_ppu_gdn_fla import save_fla_sources


def pla_operands(packed, initial):
    """CPU-only lossless ABI adaptation; never reapply sigmoid/normalization."""
    if (len(packed) != 5 or any(x.device.type != "cpu" for x in (*packed, initial))
            or [x.dtype for x in packed] != [torch.bfloat16] * 3 + [torch.float32, torch.bfloat16]
            or initial.dtype != torch.float32):
        raise ValueError("expected saved CPU BF16 QKV/beta and FP32 natural-log g/KV state")
    shape = core_shape(packed)
    if shape["B"] != 1 or tuple(initial.shape) != (1, shape["Hv"], 128, 128):
        raise ValueError("PLA legacy state must be [1,Hv,K,V], not an indexed VK pool")
    beta = packed[4].float().contiguous()
    if not torch.equal(beta.bfloat16().view(torch.int16), packed[4].contiguous().view(torch.int16)):
        raise AssertionError("beta promotion was not lossless")
    return (*packed[:4], beta), initial


def pla_call(fn, inputs, initial, offsets):
    # Host offsets are the exact same metadata already copied to the device.
    # Keep upstream auto_cp default; do not turn the reference into a forced arm.
    return lambda: fn(*inputs, scale=128 ** -.5, initial_state=initial,
                      initial_state_indices=None, cu_seqlens=offsets,
                      output_final_state=True, normalize_qk=False,
                      raw_cu_seqlens=(0, inputs[0].shape[1]), final_state_dtype=torch.float32)


class PlaCall:
    def __init__(self, manifest, packed, initial):
        validate_pla(manifest)
        if any(name == "pla" or name.startswith("pla.") for name in sys.modules):
            raise ValueError("PLA must not have been imported from another installation")
        runtime = Path(manifest["runtime"]).resolve()
        sys.path.insert(0, str(runtime))
        ops = importlib.import_module("pla.prefill.flashqla.ops")
        image = importlib.import_module("pla.prefill.flashqla._C")
        for module in (ops, image):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(runtime) or manifest["files"].get(str(path.relative_to(runtime))) != sha(path):
                raise ValueError("PLA imported an unbound source or extension")
        if ops.CHUNK_SIZE != CHUNKS["pla"]:
            raise ValueError("PLA default chunk changed")
        physical, state = pla_operands(packed, initial)
        self.inputs = tuple(x.cuda() for x in physical)
        self.initial = state.cuda()
        self.offsets = torch.tensor([0, packed[0].shape[1]], dtype=torch.int32).cuda()
        self.call = pla_call(ops.chunk_gated_delta_rule_fwd, self.inputs, self.initial, self.offsets)
        self.route = dict(api="pla.prefill.flashqla.chunk_gated_delta_rule_fwd", chunk=32,
                          auto_cp="upstream-default", qk_normalization=False,
                          state_layout="B,Hv,K,V", beta="BF16-values-losslessly-promoted-to-FP32",
                          scale=128 ** -.5, output_final_state=True,
                          extension=str(Path(image.__file__).resolve()))

    def __call__(self):
        return self.call()

    def cpu_pair(self, result):
        if not isinstance(result, tuple) or len(result) != 5:
            raise AssertionError("PLA changed its (prefix,A,O,None,ht) return ABI")
        return result[2].detach().cpu(), result[4].detach().cpu()


def physical_fingerprint(call):
    values = (*call.inputs, call.initial)
    if isinstance(call, PlaCall):
        values += (call.offsets,)
    return digest(values)


def construct(role, library, packed, initial, manifest):
    if role == "default":
        return DefaultCall(NativeLibrary(library, target="ppu10"), packed, initial)
    if role == "fla":
        call = FlaCall(packed, initial)
        call.route = call.identity
        return call
    return PlaCall(manifest, packed, initial)


def validate_payload(payload, previous, cell):
    packed, initial = payload["packed"], payload["initial"]
    if core_shape(packed) != geometry(cell) or previous["gate"] != cell["gate"]:
        raise ValueError("saved operands do not match registered geometry/gate")
    pla_operands(packed, initial)  # The same dtype/layout admission applies to all arms.
    fingerprint = digest((*packed, initial))
    if fingerprint != previous["packed_input_sha"]:
        raise ValueError("canonical processed input changed")
    if torch.equal(initial, initial.transpose(-1, -2)):
        raise ValueError("require nonzero asymmetric state to expose KV/VK mistakes")
    return fingerprint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pla-manifest", type=Path, required=True)
    parser.add_argument("--cell", choices=[c["id"] for c in cells()], required=True)
    parser.add_argument("--phase", choices=("admit", "subject"), required=True)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    record = dict(status="FAIL", phase=args.phase, role=args.role, chunks=CHUNKS)
    try:
        if not torch.cuda.is_available():
            record.update(status="SKIP", reason="no device; no numerical/performance admission")
            return 3
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        device = current_device_identity("ppu10")
        device["uuid"] = str(getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE"))
        validate_810(device)
        previous_path = args.inputs / "admission.json"
        previous = json.loads(previous_path.read_text())
        cell = next(c for c in cells() if c["id"] == args.cell)
        operands = args.inputs / "operands.pt"
        if (previous.get("status") != "PASS" or previous["device"] != device
                or previous["library_sha"] != sha(args.library)
                or previous["operands_sha"] != sha(operands)):
            raise ValueError("previous input/device/environment/binary admission no longer matches")
        # Verify all previously saved FLA sources, not only its public entry.
        for source in previous["fla_sources"]:
            if sha(Path(source["original"])) != source["sha256"]:
                raise ValueError(f"FLA source changed: {source['original']}")
        payload = torch.load(operands, map_location="cpu", weights_only=True)
        canonical = validate_payload(payload, previous, cell)
        manifest = json.loads(args.pla_manifest.read_text())
        validate_pla(manifest)
        record.update(device=device, shape=geometry(cell), gate=cell["gate"],
                      library_sha=previous["library_sha"], operands_sha=previous["operands_sha"],
                      input_admission_sha=sha(previous_path), canonical_input_sha=canonical, pla=manifest)
        packed, initial = payload["packed"], payload["initial"]
        if args.phase == "admit":
            if args.role is not None:
                raise ValueError("admit checks all three cores")
            records = {}
            for role in ROLES:
                call = construct(role, args.library, packed, initial, manifest)
                if role == "default" and call.route != previous["default_route"]:
                    raise ValueError("public default no longer selects the same admitted route")
                if role == "fla" and call.identity != previous["fla"]:
                    raise ValueError("FLA entry/compiler/backend changed")
                before, outputs = physical_fingerprint(call), []
                if role != "pla" and input_fingerprint(call) != canonical:
                    raise ValueError("core operands are not the canonical values")
                for _ in range(8):
                    result = capture(call)
                    errors = checked(result, payload["want_core"], role)
                    outputs.append(digest(result))
                if len(set(outputs)) != 1 or physical_fingerprint(call) != before:
                    raise AssertionError(f"{role}: result unstable or inputs/initial state mutated")
                records[role] = dict(input_sha=before, output_sha=outputs[0], errors=errors,
                                     route=call.route, repeats=8)
                if role == "fla":
                    record["fla_sources"] = save_fla_sources(args.directory / "fla-sources")
            record.update(arms=records, numeric_repeats=8)
        else:
            if args.role is None:
                raise ValueError("subject requires exactly one role")
            admission = json.loads((args.directory / "admission.json").read_text())
            if admission["status"] != "PASS":
                raise ValueError("three-core numerical admission did not pass")
            for name in ("device", "shape", "gate", "library_sha", "operands_sha", "input_admission_sha",
                         "canonical_input_sha", "pla", "chunks"):
                if record[name] != admission[name]:
                    raise ValueError(f"subject identity changed: {name}")
            call = construct(args.role, args.library, packed, initial, manifest)
            expected = admission["arms"][args.role]
            before = physical_fingerprint(call)
            if before != expected["input_sha"] or call.route != expected["route"]:
                raise ValueError("subject inputs/route differ from numerical admission")
            # No device warmup/verification kernels. The external profiler sees
            # all work of exactly one public call, including reference helpers.
            result = capture(call)
            errors = checked(result, payload["want_core"], args.role)
            output_sha = digest(result)
            if physical_fingerprint(call) != before or output_sha != expected["output_sha"]:
                raise AssertionError("subject differs from eight-repeat numerical gate")
            record.update(input_sha=before, output_sha=output_sha, route=call.route, errors=errors,
                          public_calls=1, warmup=0, verification="CPU")
        record["status"] = "PASS"
        print("[GDN threeway subject] " + json.dumps(dict(
            phase=args.phase, role=args.role, cell=args.cell, status="PASS", chunks=CHUNKS)), flush=True)
        return 0
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        args.receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
