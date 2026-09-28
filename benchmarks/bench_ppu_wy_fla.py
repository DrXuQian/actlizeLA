#!/usr/bin/env python3
"""Same-input original/WY/FLA comparison. No routing promotion or reset retuning."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_ppu_gdn_fla import admission, checked_pair, fla_call, load_fla, verdict
from actlize_la import gdn_chunk_wy
from actlize_la.gdn_wy_interface import (DELIVERIES, PACKED_DELIVERIES, TILED_DELIVERIES,
                                          STATE_DELIVERIES, STAGE_DELIVERIES, PREPARE_ROWS_DELIVERIES,
                                          AIU_DELIVERIES, SPLIT_PREPARE_DELIVERIES, STATE_PIPELINE_DELIVERIES)


DELIVERY_ROLES = ("original", "wy", "wy-prepare", "wy-state", "wy-output", "wy-all", "fla")
TILE_ROLES = ("original", "wy", *(f"wy-{name}" for name in TILED_DELIVERIES), "fla")
STATE_ROLES = ("original", "wy", *(f"wy-{name}" for name in STATE_DELIVERIES), "fla")
STAGE_ROLES = ("original", "wy", *(f"wy-{name}" for name in STAGE_DELIVERIES), "fla")
PREPARE_ROWS_ROLES = ("original", "wy", *(f"wy-{name}" for name in PREPARE_ROWS_DELIVERIES), "fla")
AIU_ROLES = ("original", "wy", *(f"wy-{name}" for name in AIU_DELIVERIES), "fla")
SPLIT_PREPARE_ROLES = ("original", "wy", *(f"wy-{name}" for name in SPLIT_PREPARE_DELIVERIES), "fla")
STATE_PIPELINE_ROLES = ("original", "wy", *(f"wy-{name}" for name in STATE_PIPELINE_DELIVERIES), "fla")


def experiment(delivery_ab=False, tile_ab=False, state_ab=False, stage_ab=False, prepare_rows_ab=False, aiu_ab=False,
               split_prepare_ab=False, state_pipeline_ab=False):
    if sum((delivery_ab, tile_ab, state_ab, stage_ab, prepare_rows_ab, aiu_ab, split_prepare_ab, state_pipeline_ab)) > 1:
        raise ValueError("choose one balanced candidate family")
    names = (STATE_PIPELINE_DELIVERIES if state_pipeline_ab else SPLIT_PREPARE_DELIVERIES if split_prepare_ab else AIU_DELIVERIES if aiu_ab else PREPARE_ROWS_DELIVERIES if prepare_rows_ab else STAGE_DELIVERIES if stage_ab else STATE_DELIVERIES if state_ab else
             TILED_DELIVERIES if tile_ab else PACKED_DELIVERIES if delivery_ab else ())
    return names, ("original", "wy", *(f"wy-{name}" for name in names), "fla")


def resolve_samples(requested, roles):
    """A changed role inventory must not inherit a partial timing-order cycle."""
    candidates = roles != ("original", "wy", "fla")
    cycle = 2 * len(roles)
    samples = requested if requested is not None else cycle if candidates else 12
    if candidates and (samples < cycle or samples % cycle):
        raise ValueError(f"{len(roles)}-role A/B needs a multiple of {cycle} samples "
                         "for complete balanced orders")
    return samples


def order(sample, roles=("original", "wy", "fla")):
    if roles != ("original", "wy", "fla"):
        # A complete 2*N cycle puts each role twice in every timing position;
        # reverse the traversal on the second half without concurrent launches.
        n = len(roles)
        shift = sample % n
        row = roles[shift:] + roles[:shift]
        return row if (sample // n) % 2 == 0 else row[::-1]
    rows = (roles, roles[::-1], (roles[1], roles[2], roles[0]),
            (roles[0], roles[2], roles[1]), (roles[2], roles[0], roles[1]),
            (roles[1], roles[0], roles[2]))
    return rows[sample % len(rows)]


def delivery_comparisons(arms, delivery, state_ab=False, stage_ab=False, prepare_rows_ab=False, aiu_ab=False,
                        split_prepare_ab=False, state_pipeline_ab=False):
    """Compare the new pair directly, without subtracting isolated stage costs."""
    controls = ("wy", "fla")
    if state_pipeline_ab:
        controls += tuple(role for role in STATE_PIPELINE_ROLES if role not in controls and role != f"wy-{delivery}")
    elif split_prepare_ab:
        controls += tuple(role for role in SPLIT_PREPARE_ROLES if role not in controls and role != f"wy-{delivery}")
    elif aiu_ab:
        controls += tuple(role for role in AIU_ROLES if role not in controls and role != f"wy-{delivery}")
    elif prepare_rows_ab:
        controls += tuple(role for role in ("original", "wy-tiled-state-output", "wy-tiled-state-output-both",
                                            "wy-stage-address-prepare") if role != f"wy-{delivery}")
        if delivery.startswith("prepare-rows-"):
            other = "warp" if delivery.endswith("shared") else "shared"
            controls += (f"wy-prepare-rows-{other}",)
    elif stage_ab:
        controls += tuple(role for role in ("original", "wy-tiled-state-output", "wy-tiled-state-output-both")
                          if role != f"wy-{delivery}")
        if delivery == "stage-address-both":
            controls += ("wy-stage-address-prepare", "wy-stage-address-output")
    elif state_ab:
        controls += tuple(role for role in ("original", "wy-tiled-state-output", "wy-tiled-all")
                          if role != f"wy-{delivery}")
        if delivery == "tiled-state-output-both":
            controls += ("wy-tiled-state-output-address", "wy-tiled-state-output-gates")
    elif delivery == "tiled-state-output":
        controls += ("wy-tiled-state", "wy-tiled-all", "original")
    candidate = arms[f"wy-{delivery}"]["samples_us"]
    result = {}
    for control in controls:
        baseline = arms[control]["samples_us"]
        label = verdict(candidate, baseline).replace("OURS", "CANDIDATE").replace("FLA", "CONTROL")
        result[control] = dict(verdict=label,
                               descriptive_speedup=statistics.median(baseline) / statistics.median(candidate))
    return result


def comparison_summary(arms):
    """Keep descriptive medians separate from the unchanged envelope rule."""
    result = dict(ratio_scope="DESCRIPTIVE_MEDIANS_NOT_ADMISSION")
    medians = {role: statistics.median(arm["samples_us"]) for role, arm in arms.items()}
    for control in ("original", "fla"):
        label = verdict(arms["wy"]["samples_us"], arms[control]["samples_us"])
        result[f"wy_vs_{control}"] = label.replace("OURS", "WY").replace("FLA", control.upper())
    result["wy_over_fla"] = medians["wy"] / medians["fla"]
    result["speedup_over_original"] = medians["original"] / medians["wy"]
    return result


@torch.inference_mode()
def compare(fn, gate, args, device):
    cpu = admission.fixture(1, 2048, 16, 32, gate)
    inputs = tuple(x.to(device) for x in cpu)
    want = admission.reference(cpu)
    calls = dict(original=lambda: admission.gdn_chunk(*inputs),
                 wy=lambda: gdn_chunk_wy(*inputs), fla=fla_call(fn, inputs, "native"))
    state_ab = getattr(args, "state_ab", False)
    stage_ab = getattr(args, "stage_ab", False)
    prepare_rows_ab = getattr(args, "prepare_rows_ab", False)
    names, roles = experiment(args.delivery_ab, args.tile_ab, state_ab, stage_ab, prepare_rows_ab,
                              getattr(args, "aiu_ab", False), getattr(args, "split_prepare_ab", False),
                              getattr(args, "state_pipeline_ab", False))
    for delivery in names:
        calls[f"wy-{delivery}"] = lambda delivery=delivery: gdn_chunk_wy(*inputs, delivery=delivery)
    record = dict(g=gate, shape="B1/S2048/Hk16/Hv32/D128", input_sha=admission.digest(cpu), arms={})
    for role in roles:
        call = calls[role]
        first = call()
        torch.cuda.synchronize()
        errors = checked_pair(first, want)
        fingerprint = admission.digest(first)
        if role.startswith("wy-") and fingerprint != record["arms"]["wy"]["fingerprint"]:
            raise AssertionError(f"{role} output/state bits differ from scalar WY")
        for _ in range(7):
            if admission.digest(call()) != fingerprint:
                raise AssertionError(f"{role} replay changed")
        record["arms"][role] = dict(errors=errors, fingerprint=fingerprint,
                                   state_dtype=str(first[1].dtype), samples_us=[], admitted_repeats=8)
        if role == "wy" or role.startswith("wy-"):
            delivery = "scalar" if role == "wy" else role.removeprefix("wy-")
            record["arms"][role]["delivery_mask"] = DELIVERIES[delivery]
            record["arms"][role]["scalar_raw_bit_equal"] = True
        del first
        for _ in range(args.warmup):
            call()
        print(f"[WY compare admission] g={gate} role={role} errors={errors} "
              f"delivery_mask={record['arms'][role].get('delivery_mask', 'NA')} "
              "repeat=8/8 NUMERIC/PASS", flush=True)
    if getattr(args, "admission_only", False):
        if admission.digest(inputs) != record["input_sha"]:
            raise AssertionError("a compared call changed input")
        record.update(timing="NOT_RUN", scope="NUMERICS_ONLY_PERFORMANCE_NOT_MEASURED")
        print(f"[WY ACU admission] g={gate} roles={roles} NUMERIC/PASS API_TIMING=NOT_RUN", flush=True)
        return record
    for sample in range(args.samples):
        for role in order(sample, roles):
            torch.cuda.synchronize()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.launches):
                result = calls[role]()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / args.launches
            if not math.isfinite(us) or us <= 0:
                raise AssertionError("invalid device event timing")
            checked_pair(result, want)
            if admission.digest(result) != record["arms"][role]["fingerprint"]:
                raise AssertionError(f"{role} timed replay changed")
            record["arms"][role]["samples_us"].append(us)
            del result
    if admission.digest(inputs) != record["input_sha"]:
        raise AssertionError("a compared call changed input")
    for role, arm in record["arms"].items():
        times = arm["samples_us"]
        arm["median_us"] = statistics.median(times)
        print(f"[WY compare] g={gate} role={role} median_us={arm['median_us']:.3f} "
              f"range=[{min(times):.3f},{max(times):.3f}] samples_us={times}", flush=True)
    record.update(comparison_summary(record["arms"]))
    for control in ("original", "fla"):
        label = record[f"wy_vs_{control}"]
        print(f"[WY verdict] g={gate} subject=wy control={control} verdict={label} rule=disjoint-observed-envelopes")
    print(f"[WY ratios] g={gate} WY/FLA={record['wy_over_fla']:.4f} "
          f"original/WY={record['speedup_over_original']:.4f} scope={record['ratio_scope']} "
          f"WY_vs_FLA={record['wy_vs_fla']} routing=UNCHANGED")
    if names:
        for delivery in names:
            role = f"wy-{delivery}"
            candidate = record["arms"][role]
            candidate["versus"] = delivery_comparisons(record["arms"], delivery, state_ab, stage_ab, prepare_rows_ab,
                                                        getattr(args, "aiu_ab", False), getattr(args, "split_prepare_ab", False),
                                                        getattr(args, "state_pipeline_ab", False))
            for control, comparison in candidate["versus"].items():
                label, speedup = comparison["verdict"], comparison["descriptive_speedup"]
                print(f"[WY delivery verdict] g={gate} candidate={role} control={control} "
                      f"speedup={speedup:.4f}x verdict={label} rule=disjoint-observed-envelopes "
                      "raw-bit-vs-scalar=PASS routing=UNCHANGED", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", required=True, type=Path)
    p.add_argument("--wy-extension", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--samples", type=int,
                   help="default: 12 without candidates, otherwise twice the role count")
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--admission-only", action="store_true",
                   help="write input/binary/numeric receipt for ACU; no API event timing or speed verdict")
    family = p.add_mutually_exclusive_group()
    family.add_argument("--delivery-ab", action="store_true", help="paired legacy packed-delivery controls")
    family.add_argument("--tile-ab", action="store_true",
                        help="paired compute-tile prepare/state/output/state+output/all controls")
    family.add_argument("--state-ab", action="store_true",
                        help="independent state address/gate reuse and combined arms; retain pair/all controls")
    family.add_argument("--stage-ab", action="store_true",
                        help="prepare/output address ablations on frozen state-both; retain old/new pair controls")
    family.add_argument("--prepare-rows-ab", action="store_true",
                        help="shared/warp exact row-factor reuse on frozen prepare-address; output unchanged")
    family.add_argument("--aiu-ab", action="store_true",
                        help="matched AIU/SWZL state/output/both on the shared-row incumbent")
    family.add_argument("--split-prepare-ab", action="store_true",
                        help="separate prefix/solve/WU resource budgets on frozen AIU state/output")
    family.add_argument("--state-pipeline-ab", action="store_true",
                        help="current K/U and next W overlap; frozen split prepare/AIU output")
    args = p.parse_args()
    _, roles = experiment(args.delivery_ab, args.tile_ab, args.state_ab, args.stage_ab, args.prepare_rows_ab, args.aiu_ab,
                          args.split_prepare_ab, args.state_pipeline_ab)
    if args.admission_only:
        if args.samples is not None:
            p.error("--admission-only does not accept --samples")
        args.samples = 0
        if args.warmup < 5:
            p.error("ACU admission retains >=5 warmups")
    else:
        try:
            args.samples = resolve_samples(args.samples, roles)
        except ValueError as exc:
            p.error(str(exc))
        if args.samples < 3 or args.launches < 1 or args.samples * args.launches < 50 or args.warmup < 5:
            p.error("need >=5 warmups, >=3 samples and >=50 timed launches per arm")
    for key, path in (("GDN_QSA_PPU_EXTENSION", args.extension), ("GDN_QSA_WY_EXTENSION", args.wy_extension)):
        if not path.is_file():
            p.error(f"missing {key}: {path}")
        os.environ[key] = str(path.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU: {props.name}")
    fn, identity = load_fla()
    result = dict(protocol="numeric-admission-for-acu" if args.admission_only else "full-public-api-event-span",
                  includes_allocation_and_launch_gaps=not args.admission_only,
                  includes_original_host_dispatch_sync=True, initial_state="zero", final_state=True,
                  qk_norm=False, scale="1/sqrt(128)", dtype="bf16", device=str(props),
                  torch=torch.__version__, fla=identity, samples=args.samples, launches=args.launches,
                  warmup=args.warmup, limit=admission.MAX_RELATIVE_ERROR,
                  delivery_ab=roles != ("original", "wy", "fla"),
                  tile_ab=args.tile_ab, state_ab=args.state_ab, stage_ab=args.stage_ab,
                  prepare_rows_ab=args.prepare_rows_ab,
                  aiu_ab=args.aiu_ab,
                  split_prepare_ab=args.split_prepare_ab,
                  state_pipeline_ab=args.state_pipeline_ab,
                  roles=roles, order_cycle_samples=2 * len(roles),
                  binary_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                                 for x in (args.extension, args.wy_extension)}, cases=[])
    for gate in (-.1, -1.):
        result["cases"].append(compare(fn, gate, args, torch.device("cuda", args.device)))
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(result, indent=2) + "\n")
    scope = "NUMERICS_ONLY_PERFORMANCE_NOT_MEASURED" if args.admission_only else "NUMERICS+MEASUREMENT_COMPLETED_NOT_SPEED_ADMISSION"
    print(f"[WY compare] PASS scope={scope} "
          f"results={args.results} routing=UNCHANGED")


if __name__ == "__main__":
    main()
