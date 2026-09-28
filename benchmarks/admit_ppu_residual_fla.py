#!/usr/bin/env python3
"""Untimed same-input scalar/pipeline/residual/FLA admission for ACU capture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from bench_ppu_gdn_fla import admission, checked_pair, fla_call, load_fla
from actlize_la import gdn_chunk_wy, gdn_chunk_residual
from actlize_la.gdn_residual_interface import RESIDUAL_ENTRYPOINTS, MATH_CONTRACT, WY_MATH_CONTRACT
from actlize_la.gdn_wy_interface import DELIVERIES


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--deliveries", nargs="+", choices=tuple(k for k in RESIDUAL_ENTRYPOINTS if k != "scalar"), default=[])
    args = p.parse_args()
    if not args.extension.is_file():
        p.error("WY extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "PPU" not in props.name.upper():
        raise RuntimeError("admission requires PPU")
    fn, identity = load_fla()
    result = dict(protocol="numeric-admission-for-acu", samples=0, cases=[],
                  torch=torch.__version__, device=str(props), fla=identity,
                  initial_state="zero", final_state=True, qk_norm=False, scale="1/sqrt(128)",
                  dtype="bf16", limit=admission.MAX_RELATIVE_ERROR,
                  binary_sha256={str(args.extension.resolve()): hashlib.sha256(args.extension.read_bytes()).hexdigest()})
    for gate in (-.1, -1.):
        cpu = admission.fixture(1, 2048, 16, 32, gate)
        inputs = tuple(t.cuda() for t in cpu)
        want = admission.reference(cpu)
        calls = {"wy": lambda: gdn_chunk_wy(*inputs),
                 "wy-state-pipeline": lambda: gdn_chunk_wy(*inputs, delivery="state-pipeline"),
                 "wy-residual": lambda: gdn_chunk_residual(*inputs),
                 "fla": fla_call(fn, inputs, "native")}
        for delivery in args.deliveries:
            calls[f"wy-residual-{delivery}"] = lambda delivery=delivery: gdn_chunk_residual(*inputs, delivery=delivery)
        residual_pair = None
        record = dict(g=gate, input_sha=admission.digest(cpu), timing="NOT_RUN", arms={})
        for role, call in calls.items():
            got = call()
            torch.cuda.synchronize()
            errors = checked_pair(got, want)
            fingerprint = admission.digest(got)
            for _ in range(7):
                repeated = call()
                checked_pair(repeated, want)
                if admission.digest(repeated) != fingerprint:
                    raise AssertionError(f"{role}: replay changed")
            arm = dict(errors=errors, fingerprint=fingerprint, state_dtype=str(got[1].dtype),
                       samples_us=[], admitted_repeats=8)
            if role == "wy-residual" or role.startswith("wy-residual-"):
                arm.update(math_contract=MATH_CONTRACT, delivery_mask=None, scalar_raw_bit_equal=None,
                           scalar_fingerprint_equal=fingerprint == record["arms"]["wy"]["fingerprint"])
                if role == "wy-residual":
                    residual_pair = got
                else:
                    if residual_pair is None or not all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                                                        for a,b in zip(got,residual_pair)):
                        raise AssertionError(f"{role} changed residual output/state bits")
                    arm.update(residual_raw_bit_equal=True,
                               residual_fingerprint=record["arms"]["wy-residual"]["fingerprint"])
            elif role.startswith("wy"):
                if role != "wy" and fingerprint != record["arms"]["wy"]["fingerprint"]:
                    raise AssertionError("old pipeline delivery lost scalar RAW-BIT equality")
                arm.update(math_contract=WY_MATH_CONTRACT, scalar_raw_bit_equal=True,
                           delivery_mask=DELIVERIES["scalar" if role == "wy" else "state-pipeline"])
            record["arms"][role] = arm
            print(f"[residual admission] g={gate} role={role} errors={errors} fingerprint={fingerprint} "
                  f"contract={arm.get('math_contract','FLA')} repeat=8/8 NUMERIC/PASS", flush=True)
        if admission.digest(inputs) != record["input_sha"]:
            raise AssertionError("comparison mutated fixture")
        result["cases"].append(record)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[residual admission] PASS API_TIMING=NOT_RUN routing=UNCHANGED results={args.results}")


if __name__ == "__main__":
    main()
