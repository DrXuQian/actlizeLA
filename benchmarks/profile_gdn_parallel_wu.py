#!/usr/bin/env python3
"""Exact saved inputs: old public default / opt-in WU / unchanged FLA."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests", ROOT / "tools", ROOT / "benchmarks"):
    sys.path.insert(0, str(path))

from actlize_la.serving import NativeLibrary
from actlize_la.serving.gdn_core import PreparedGDNCore, gdn_core_workspace_specs
from actlize_la.serving.device import current_device_identity
from check_ppu810_gdn_recheck import validate_810
from collect_ppu_gdn_acu import sha
from gdn_kda_contract import ENTRY, ROLES, cells, geometry
from profile_gdn_threeway import validate_payload
from profile_serving_prefill import DefaultCall, FlaCall, capture, checked, core_shape, digest, input_fingerprint
from profile_ppu_gdn_fla import save_fla_sources


class PreparedCall:
    """Benchmark-only prepared owner; no public API/selector changes."""
    def __init__(self, library, packed, initial, candidate=True):
        shape = core_shape(packed)
        if shape["T"] % 64:
            raise ValueError("parallel W/U is admitted only for complete C64 chunks")
        self.inputs, self.initial = tuple(x.cuda() for x in packed), initial.cuda()
        self.output = torch.empty_like(self.inputs[2])
        self.final = torch.empty_like(self.initial)
        self.work = {name: torch.empty(extent, dtype=dtype, device="cuda") for name, (extent, dtype) in
                     gdn_core_workspace_specs(shape["B"], shape["T"], shape["Hk"], shape["Hv"]).items()}
        self.owner = PreparedGDNCore(library, **dict(zip(("q", "k", "v", "g", "beta"), self.inputs)),
                                     initial_state=self.initial, final_state=self.final,
                                     output=self.output, workspace=self.work)
        if candidate:
            configure = library.lib.actlize_la_ppu10_parallel_wu_configure
            configure.argtypes, configure.restype = [], C.c_int
            self.owner.check(configure())
            fn = getattr(library.lib, ENTRY)
            fn.argtypes, fn.restype = self.owner.fn.argtypes, C.c_int
            self.owner.fn = fn
        self.route = dict(entry=ENTRY if candidate else "actlize_la_ppu10_forward_prepared", chunk=64,
                          arithmetic="materialized-WU-BF16" if candidate else "retained-residual",
                          public_default_replaced=False, final_state=True,
                          library_sha=sha(library.path), workspace_bytes=sum(x.numel() * x.element_size() for x in self.work.values()))

    def __call__(self):
        self.owner()
        return self.output, self.final

    def cpu_pair(self, result):
        return tuple(x.detach().cpu() for x in result)


def construct(role, baseline, candidate, packed, initial):
    if role == "default":
        return DefaultCall(NativeLibrary(baseline, target="ppu10"), packed, initial)
    if role == "parallel-wu":
        return PreparedCall(NativeLibrary(candidate, target="ppu10"), packed, initial)
    if role == "fla":
        call = FlaCall(packed, initial)
        call.route = call.identity
        return call
    raise ValueError("unknown role")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "candidate", "inputs", "directory", "receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cell", choices=[c["id"] for c in cells()], required=True)
    p.add_argument("--phase", choices=("admit", "subject"), required=True)
    p.add_argument("--role", choices=ROLES)
    a = p.parse_args()
    record = dict(status="FAIL", phase=a.phase, role=a.role)
    try:
        if not torch.cuda.is_available():
            record.update(status="SKIP", reason="no device; no numerical/performance admission")
            return 3
        torch.set_num_threads(1)
        torch.cuda.set_device(0)
        device = current_device_identity("ppu10")
        device["uuid"] = str(getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE"))
        validate_810(device)
        previous_path = a.inputs / "admission.json"
        previous = json.loads(previous_path.read_text())
        operands = a.inputs / "operands.pt"
        if (previous.get("status") != "PASS" or previous["device"] != device
                or previous["library_sha"] != sha(a.library) or previous["operands_sha"] != sha(operands)):
            raise ValueError("baseline device/input/library identity changed")
        for source in previous["fla_sources"]:
            if sha(Path(source["original"])) != source["sha256"]:
                raise ValueError("FLA source changed: " + source["original"])
        payload = torch.load(operands, map_location="cpu", weights_only=True)
        cell = next(c for c in cells() if c["id"] == a.cell)
        canonical = validate_payload(payload, previous, cell)
        record.update(device=device, shape=geometry(cell), gate=cell["gate"],
                      libraries=dict(default=sha(a.library), candidate=sha(a.candidate)),
                      operands_sha=sha(operands), input_admission_sha=sha(previous_path), canonical_input_sha=canonical)
        packed, initial = payload["packed"], payload["initial"]

        def make(role):
            call = construct(role, a.library, a.candidate, packed, initial)
            if input_fingerprint(call) != canonical:
                raise AssertionError("physical inputs differ from the canonical tensors")
            if role == "default" and call.route != previous["default_route"]:
                raise ValueError("public default no longer selects the admitted route")
            if role == "fla" and call.identity != previous["fla"]:
                raise ValueError("FLA entry/compiler/backend changed")
            return call

        if a.phase == "admit":
            if a.role:
                raise ValueError("admit requires all three roles")
            arms = {}
            for role in ROLES:
                call = make(role)
                fingerprints = []
                for _ in range(8):
                    if role == "parallel-wu":
                        for scratch in call.work.values():
                            scratch.fill_(float("nan"))  # admission only, never in profiled subject
                    result = capture(call)
                    errors = checked(result, payload["want_core"], role)
                    fingerprints.append(digest(result))
                if len(set(fingerprints)) != 1 or input_fingerprint(call) != canonical:
                    raise AssertionError("unstable output/state or mutated inputs")
                arms[role] = dict(input_sha=canonical, output_sha=fingerprints[0], route=call.route, errors=errors, repeats=8)
                if role == "fla":
                    record["fla_sources"] = save_fla_sources(a.directory / "fla-sources")
            # The opt-in object must not change the shipped four-stage default.
            control = PreparedCall(NativeLibrary(a.candidate, target="ppu10"), packed, initial, candidate=False)
            for _ in range(8):
                if digest(capture(control)) != arms["default"]["output_sha"]:
                    raise AssertionError("unchanged default in candidate library differs from old baseline bits")
            if input_fingerprint(control) != canonical:
                raise AssertionError("default control mutated its inputs")
            record.update(arms=arms, numeric_repeats=8, unchanged_public_default_bits="PASS")
        else:
            if not a.role:
                raise ValueError("subject requires a role")
            admission = json.loads((a.directory / "admission.json").read_text())
            if admission.get("status") != "PASS":
                raise ValueError("numerical admission did not pass")
            for name in ("device", "shape", "gate", "libraries", "operands_sha", "input_admission_sha", "canonical_input_sha"):
                if record[name] != admission[name]:
                    raise ValueError("subject identity changed: " + name)
            call = make(a.role)
            result = capture(call)  # Exactly one full core call; external ACU owns capture.
            errors, fingerprint = checked(result, payload["want_core"], a.role), digest(result)
            expected = admission["arms"][a.role]
            if (fingerprint != expected["output_sha"] or call.route != expected["route"]
                    or input_fingerprint(call) != canonical):
                raise AssertionError("subject differs from eight-repeat gate")
            record.update(input_sha=canonical, output_sha=fingerprint, route=call.route, errors=errors,
                          public_calls=1, warmup=0, verification="CPU")
        record["status"] = "PASS"
        print(f"[GDN KDA] cell={a.cell} phase={a.phase} role={a.role} NUMERIC/PASS public-default=UNCHANGED", flush=True)
        return 0
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        a.receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
