#!/usr/bin/env python3
"""Read-only ACU geometry audit. No imports/launches of measured device code."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

from collect_ppu_gdn_acu import validate_pair, validate_preflight, validate_wy_control
from sweep_ppu10_geometry import CONTROL, DELIVERIES, cells, check_comparison

HEADER = re.compile(r"^  (\S.*) \((\d+),(\d+),(\d+)\)x\((\d+),(\d+),(\d+)\), Device (\d+)\s*$")
TIME_US = {"nsecond": .001, "usecond": 1, "msecond": 1000, "second": 1000000}
STATE_METRICS = {
    "registers_per_thread": "launch__registers_per_thread",
    "shared_bytes": "launch__shared_mem_per_block",
    "stack_bytes_reported_per_thread": "launch__stack_size_per_thread",
    "resident_blocks_ceiling": "launch__occupancy_blocks_per_cu",
    "register_block_limit": "launch__occupancy_limit_registers",
    "shared_block_limit": "launch__occupancy_limit_shared_mem",
    "warp_block_limit": "launch__occupancy_limit_warps",
    "waves_per_cu": "launch__waves_per_cu",
    "active_warps_per_cu": "cu__warps_active.avg.per_cycle_active",
    "achieved_occupancy_pct": "cu__warps_active.avg.pct_of_peak_sustained_active",
    "executed_instructions_pu": "pu__inst_executed.sum",
    "ce_hz": "ce__cycles_elapsed.avg.per_second",
    "bank_conflicts": "cu__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "bank_conflicts_load": "cu__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "bank_conflicts_store": "cu__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "shared_load_transactions": "cu__data_pipe_lsu_transactions_mem_shared_op_ld.sum",
    "shared_store_transactions": "cu__data_pipe_lsu_transactions_mem_shared_op_st.sum",
    "global_load_bytes": "kvd__transactions_pipe_lsu_mem_global_op_ld_bytes.sum",
    "global_store_bytes": "kvd__transactions_pipe_lsu_mem_global_op_st_bytes.sum",
    "global_to_shared_bytes": "kvd__transactions_pipe_lsu_mem_global_op_ldgsts_bytes.sum",
    **{f"{s}_per_issue": f"pu__we_average_warps_issue_stalled_{s}_per_issue_active.ratio"
       for s in ("memory_dependency", "compute_dependency", "inst_fetch", "amc", "sync", "not_selected")},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksums(data):
    result = {}
    for line in data.decode().splitlines():
        digest, name = line.split("  ", 1)
        path = PurePosixPath(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in result
                and not path.is_absolute() and ".." not in path.parts
                and str(path) == name, "invalid checksum entry")
        result[name] = digest
    return result


def load(archive):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        require(len(members) == len({m.name for m in members}), "duplicate tar entry")
        for member in members:
            p = PurePosixPath(member.name)
            require(member.isfile() and not p.is_absolute() and ".." not in p.parts
                    and str(p) == member.name, "nonregular/unsafe tar entry")
        files = {m.name: tar.extractfile(m).read() for m in members}
    sums = checksums(files["LIGHT_SHA256SUMS"])
    require(set(sums) == set(files) - {"LIGHT_SHA256SUMS"}, "upload checksum denominator")
    for name, digest in sums.items():
        require(hashlib.sha256(files[name]).hexdigest() == digest, "changed upload: " + name)
    return files


def parse(data, raw=True):
    kernels = []
    for line in data.decode().splitlines():
        h = HEADER.match(line)
        if h:
            kernels.append(dict(symbol=h[1], grid=list(map(int, h.groups()[1:4])),
                                block=list(map(int, h.groups()[4:7])), device=int(h[8]), metrics={}))
        elif kernels and line.startswith("    "):
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 2 or parts[0] == "Metric Name" or parts[0].startswith("-"):
                continue
            # ACU repeats device attributes with different units/limits (e.g.
            # max-registers 255 vs256). These are NOT executed launch metrics.
            if raw and parts[0].startswith("device__"):
                continue
            if not raw and parts[0] != "Duration":
                continue
            value = parts[-1]
            try:
                value = float(value.replace(",", ""))
            except ValueError:
                pass
            item = dict(unit=parts[1] if len(parts) == 3 else "", value=value)
            metrics = kernels[-1]["metrics"]
            require(parts[0] not in metrics or metrics[parts[0]] == item,
                    "conflicting metric: " + parts[0])
            metrics[parts[0]] = item
    require(kernels, "no kernels")
    for k in kernels:
        duration = k["metrics"]["ppu__time_duration.sum" if raw else "Duration"]
        require(duration["unit"] in TIME_US and math.isfinite(duration["value"])
                and duration["value"] > 0, "invalid/missing duration")
        k["us"] = duration["value"] * TIME_US[duration["unit"]]
    return kernels


def numeric(k, key):
    value = k["metrics"][key]["value"]
    require(isinstance(value, (float, int)) and math.isfinite(value), "invalid " + key)
    return value


def state_counters(k):
    return {alias: numeric(k, key) for alias, key in STATE_METRICS.items()}


def audit(files):
    index = json.loads(files["LIGHT_INDEX.json"])
    source = files["sha.txt"].decode().strip()
    shapes = json.loads(files["geometry-shapes.json"])
    require(len(shapes) == 6 and len({tuple(s[k] for k in ("B", "S", "Hk", "Hv")) for s in shapes}) == 6,
            "shape denominator")
    comparison = json.loads(files["comparison.json"])
    check_comparison(comparison, shapes)
    require(index["cells"] == 12 and index["arms"] == 60 and len(index["reports"]) == 60
            and index["measured_source_sha"] == source, "index denominator/source")
    outer = checksums(files["original-geometry.SHA256SUMS"])
    for name in files:
        if "/" not in name and name in outer:
            require(hashlib.sha256(files[name]).hexdigest() == outer[name], "changed original root: " + name)
    result = dict(source_sha=source, performance_scope="ONE_CAPTURE_PER_ARM_CELL_DESCRIPTIVE_ONLY",
                  decision="NO_DEFAULT_PROMOTION_REPEATS_REQUIRED", registration_sha256=outer["geometry-registration.md"],
                  cells=[], state_metric_names=STATE_METRICS,
                  counters_scope="raw text metrics; no per-opcode/per-PC reconstruction",
                  originals_scope="capture-time report digests retained; native reports/binaries not uploaded")
    identity = None
    count = 0
    expected_cells = {name for name, _, _ in cells(shapes)}
    require({n.rsplit("/", 1)[0] for n in files if n.endswith("/STATUS.json")} == expected_cells,
            "capture cell denominator")
    for name, shape, gate in cells(shapes):
        status = json.loads(files[name + "/STATUS.json"])
        labels = ["wy-control", "wy", *("wy-" + d for d in DELIVERIES[1:]), "fla"]
        require(status["status"] == "PASS" and not status["errors"] and status["capture_order"] == labels
                and set(status["capture_arms"]) == set(labels)
                and status["comparison_origin"]["source_sha"] == source, "capture identity: " + name)
        sums = checksums(files[name + "/SHA256SUMS"])
        for path, data in files.items():
            if path.startswith(name + "/") and path != name + "/SHA256SUMS":
                relative = path[len(name) + 1:]
                require(hashlib.sha256(data).hexdigest() == sums.get(relative), "changed original capture: " + path)
        row = dict(cell=name, shape=shape, gate=gate, arms={})
        receipts = {}
        for label, delivery in zip(labels, (CONTROL, *DELIVERIES, DELIVERIES[0])):
            role = "fla" if label == "fla" else "wy"
            directory = "." if label in ("wy", "fla") else label
            require(status["capture_arms"][label] == dict(role=role, wy_delivery=delivery, directory=directory),
                    "arm mapping: " + label)
            prefix = name + "/" + (directory + "/" if directory != "." else "") + role
            receipt = json.loads(files[prefix + ".json"])
            preflight = json.loads(files[prefix + "-preflight.json"])
            validate_preflight(preflight, receipt)
            require(receipt["shape"] == dict(**shape, K=128, V=128) and receipt["gate"] == gate
                    and receipt["wy_delivery"] == delivery, "receipt shape/arm: " + label)
            current = tuple(receipt[k] for k in ("device", "torch", "torch_cuda", "extension_sha256", "library_sha256"))
            if identity is None:
                identity = current
                result["identity"] = dict(zip(("device", "torch", "torch_cuda", "extension_sha256", "library_sha256"), current))
            require(current == identity, "mixed device/runtime/binary")
            errors = receipt["errors"]
            require(len(errors) == 2 and receipt["max_relative_error_limit"] == .02
                    and all(0 <= e < .02 for e in errors), "numeric receipt")
            receipts[label] = receipt
            raw, details = (parse(files[prefix + "-" + page + ".txt"], page == "raw") for page in ("raw", "details"))
            require(len(raw) == len(details) == (7 if role == "fla" else 4), "kernel denominator: " + label)
            for rk, dk in zip(raw, details):
                require(all(rk[x] == dk[x] for x in ("symbol", "grid", "block", "device")) and rk["us"] == dk["us"],
                        "raw/details disagree: " + label)
            if role == "fla":
                markers = ("chunk_local_cumsum_scalar_kernel", "FillFunctor<c10::BFloat16>",
                           "chunk_gated_delta_rule_fwd_kkt_solve_kernel", "recompute_w_u_fwd_kernel",
                           "FillFunctor<float>", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "chunk_fwd_kernel_o")
                state = raw[5]
            else:
                v, w = (32, 8) if label in ("wy-control", "wy") else ((32, 4) if label.endswith("v32-w4") else (16, 4))
                full = shape["S"] % 64 == 0
                state_marker = ("gdn_wy_residual_full_chunk_state" if full else "gdn_wy_residual_gate_cache_state") if label == "wy-control" else f"gdn_wy_geometry_state<{v}, {w}, {'true' if full else 'false'}>"
                markers = ("gdn_wy_split_prefix", "gdn_wy_split_solve_static", state_marker,
                           "gdn_wy_residual_warps8_hvlayout_output")
                state = raw[2]
                require(state["grid"] == [shape["B"] * shape["Hv"] * (128 // v), 1, 1]
                        and state["block"] == [w * 32, 1, 1], "wrong state geometry")
            require(all(marker in k["symbol"] for marker, k in zip(markers, raw)), "wrong/missing kernel symbol")
            row["arms"][label] = dict(total_us=sum(k["us"] for k in raw), state_us=state["us"],
                                     kernels=[dict(**{n: k[n] for n in ("symbol", "grid", "block", "us")},
                                                   instructions_pu=numeric(k, "pu__inst_executed.sum")) for k in raw],
                                     state_counters=state_counters(state), errors=errors,
                                     clocks_hz=[numeric(k, "ce__cycles_elapsed.avg.per_second") for k in raw])
            count += len(raw)
        for label in labels[:-1]:
            validate_pair(receipts[label], receipts["fla"], "wy", shared_reference=label != "wy")
            if label != "wy-control":
                validate_wy_control(receipts["wy-control"], receipts[label])
        for label, arm in row["arms"].items():
            control = row["arms"]["wy-control"]
            arm["delta_vs_control_pct"] = 100 * (arm["total_us"] / control["total_us"] - 1)
            if label != "fla":
                for pos in (0, 1, 3):
                    require(all(arm["kernels"][pos][k] == control["kernels"][pos][k]
                                for k in ("symbol", "grid", "block", "instructions_pu")),
                            "unregistered change in a nonstate stage")
                arm["unchanged_stages_delta_us"] = arm["total_us"] - control["total_us"] - (arm["state_us"] - control["state_us"])
        result["cells"].append(row)
    require(count == 276, "total kernel denominator")
    result["kernel_count"] = count
    return result


def negative_controls(files):
    """Plant semantic defects after refreshing checksums, not hash-only tests."""
    cell = "acu-B1-S2048-Hk16-Hv32-g-1.0"
    candidate = cell + "/wy-residual-geometry-v16-w4/wy"
    cases = []
    missing_cell = {k: v for k, v in files.items() if not k.startswith(cell + "/")}
    cases.append(("missing-cell", missing_cell))
    changed = files.copy()
    del changed[candidate + "-raw.txt"]
    cases.append(("missing-arm-export", changed))
    changed = files.copy()
    for page in ("raw", "details"):
        key = cell + "/fla-" + page + ".txt"
        sections = re.split(r"(?=^  \S.*\), Device \d+$)", changed[key].decode(), flags=re.M)
        require(sum("FillFunctor<c10::BFloat16>" in s.splitlines()[0] for s in sections) == 1,
                "negative target missing")
        changed[key] = "".join(s for s in sections if "FillFunctor<c10::BFloat16>" not in s.splitlines()[0]).encode()
    cases.append(("omitted-fla-fill", changed))
    for plant, before, after in (("wrong-geometry", b"gdn_wy_geometry_state<16, 4, true>", b"gdn_wy_geometry_state<32, 8, true>"),
                                ("wrong-duration-unit", b"nsecond", b"usecond")):
        changed = files.copy()
        key = candidate + "-raw.txt"
        require(before in changed[key], "negative target missing: " + plant)
        changed[key] = changed[key].replace(before, after, 1)
        # The wrong symbol must survive raw/details agreement to reach the
        # exact-instance check. The unit plant must fail that agreement.
        if plant == "wrong-geometry":
            key = candidate + "-details.txt"
            changed[key] = changed[key].replace(before, after, 1)
        cases.append((plant, changed))
    for plant, field in (("mixed-device", "device"), ("wrong-input-shape", "shape")):
        changed = files.copy()
        for suffix in (".json", "-preflight.json"):
            key = candidate + suffix
            record = json.loads(changed[key])
            if field == "device":
                record[field]["uuid"] = "planted-other-device"
            else:
                record[field]["S"] += 1
            changed[key] = json.dumps(record).encode()
        cases.append((plant, changed))
    for name, changed in cases:
        # Retained capture bytes are rehashed so the numerical/timing/identity
        # checker, not merely the checksum checker, must catch the defect.
        for path in (k for k in changed if k.endswith("/SHA256SUMS")):
            root = path.rsplit("/", 1)[0] + "/"
            sums = checksums(changed[path])
            for relative in sums:
                if root + relative in changed:
                    sums[relative] = hashlib.sha256(changed[root + relative]).hexdigest()
            changed[path] = "".join(f"{digest}  {relative}\n" for relative, digest in sums.items()).encode()
        try:
            audit(changed)
        except (ValueError, KeyError):
            print("EXPECTED-RED/PASS:", name)
        else:
            raise ValueError("negative control escaped: " + name)
    return len(cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    files = load(args.archive)
    result = audit(files)
    if args.self_test:
        result["negative_controls_passed"] = negative_controls(files)
    result["archive_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if args.output:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print("cell | control | V32/8 | V32/4 | V16/4 | FLA; microseconds, all kernels")
    for row in result["cells"]:
        print(row["cell"], "|", " | ".join(f"{a['total_us']:.5f}" for a in row["arms"].values()))
    print("PASS: 12 cells / 60 arms / 276 kernels; receipts+raw/details reconciled; no default promotion")


if __name__ == "__main__":
    main()
