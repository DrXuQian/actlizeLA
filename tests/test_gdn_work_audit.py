"""Work/inventory negatives; synthetic counters are never device evidence."""
import copy
import csv
import io
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "dev/ppu/zw810_fla_kda")]
import analyze_work as work
import acu_report_details as reader
import export_gdn_kda_details as exporter
import gdn_kda_contract as contract


def test_matrix_work_is_not_executed_instruction_count():
    model = work.logical_work()
    bf16 = model["bf16_macs"]
    assert bf16["kkt_lower_10_blocks"] == 327680
    assert 2 * sum(bf16["residual_state"].values()) == 5242880
    assert 2 * sum(bf16["fla_wu_and_state"].values()) == 6291456
    assert 2 * sum(bf16["output"].values()) == 4194304
    assert model["inverse_merge_products"] == 16
    assert model["inverse_merge_macs"] == 65536
    assert sum(bf16["residual_state"].values()) / sum(bf16["fla_wu_and_state"].values()) == 5/6


def csv_rows():
    fields = ["ID", "Process ID", "Kernel Name", "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Grid Size", "Block Size"]
    names = [*work.retained.FLA, "vectorized_elementwise_kernel<FillFunctor<float>>", "vectorized_elementwise_kernel<FillFunctor<BFloat16>>"]
    rows = [dict(zip(fields, [str(i), "1", n, work.SECTION, work.METRIC, "inst", str(100*(i+1)), "(32,1,1)", "(256,1,1)"])) for i,n in enumerate(names)]
    kernels = [dict(id=r["ID"], process="1", kernel=r["Kernel Name"]) for r in rows]
    return fields, rows, kernels


def encode_csv(fields, rows):
    out = io.StringIO()
    w = csv.DictWriter(out, fields, quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def test_counts_include_both_fills_and_wu_before_normalization():
    fields, rows, kernels = csv_rows()
    got = work.instruction_rows(encode_csv(fields, rows), kernels)
    assert sum(r["count"] for r in got) == 2800
    assert sum(r["count"] for r in got if work.work_phase("fla", r["kernel"]) == "state_equivalent") == 700
    assert sum(r["count"] for r in got if work.work_phase("fla", r["kernel"]) == "other_included") == 1300


@pytest.mark.parametrize("plant", ["missing", "duplicate", "wrong-unit", "wrong-section", "nonfinite", "fraction", "wrong-kernel"])
def test_instruction_count_negatives(plant):
    fields, rows, kernels = csv_rows()
    if plant == "missing":
        rows.pop()
    elif plant == "duplicate":
        rows.append(rows[0])
    elif plant == "wrong-unit":
        rows[0]["Metric Unit"] = "inst/thread"
    elif plant == "wrong-section":
        rows[0]["Section Name"] = "Other counter"
    elif plant == "nonfinite":
        rows[0]["Metric Value"] = "NaN"
    elif plant == "fraction":
        rows[0]["Metric Value"] = "1.5"
    else:
        rows[0]["Kernel Name"] = "not_the_captured_kernel"
    with pytest.raises(ValueError):
        work.instruction_rows(encode_csv(fields, rows), kernels)


def summary_fixture():
    captures = {}
    for cell in contract.cells():
        arms = {}
        for role in contract.roles("solve-tf32x1"):
            names = work.retained.FLA if role == "fla" else work.retained.CORE if role == "default" else contract.EXPERIMENTS[role]["kernels"]
            kernels = [dict(id=str(i), process="1", kernel=n, us=float(i+1), grid="(1,1,1)", block="(32,1,1)") for i,n in enumerate(names)]
            cap = contract.inventory(role, kernels)
            cap["report_sha256"] = "1"*64
            arms[role] = [copy.deepcopy(cap) for _ in range(3)]
        captures[cell["id"]] = dict(cell=cell, captures=arms, summary=contract.summarize(arms, "solve-tf32x1"))
    return dict(status="COLLECTED", errors=[], experiment="solve-tf32x1", cells=contract.cells(), captures=captures)


def test_detail_subset_is_fixed_not_fastest():
    s = summary_fixture()
    selected = exporter.select_reports(s)
    assert len(selected) == 12
    assert {r["cell"] for r in selected} == {c["id"] for c in contract.cells() if c["gate"] == -.1}
    assert {r["repeat"] for r in selected} == {0}


@pytest.mark.parametrize("plant", ["missing-cell", "missing-role", "missing-repeat", "missing-report-hash"])
def test_export_denominator_negative(plant):
    s = summary_fixture()
    key = contract.cells()[0]["id"]
    if plant == "missing-cell":
        del s["captures"][key]
    elif plant == "missing-role":
        del s["captures"][key]["captures"]["fla"]
    elif plant == "missing-repeat":
        s["captures"][key]["captures"]["fla"].pop()
    else:
        del s["captures"][key]["captures"]["fla"][0]["report_sha256"]
    with pytest.raises(ValueError):
        exporter.select_reports(s)


def detail_fixture():
    expected = dict(report_sha256="1"*64, kernels=[dict(id="0", kernel="kernel(int)", grid="(32,1,1)", block="(128,1,1)", us=4.2)])
    action = dict(name="kernel", demangled="kernel(int)", capture_id=0, grid="32,1,1", block="128,1,1",
                  metrics={"ppu__time_duration.sum": dict(value=4200, unit="nsecond", has_value=True)},
                  sass=[dict(sass_code="v.mma.f32.bf16.m16n16k16")], coverage=dict(warp_executions=10))
    return dict(report_sha256="1"*64, actions=[action]), expected


def test_native_duration_and_geometry_match():
    detail, expected = detail_fixture()
    exporter.check_detail(detail, expected)


@pytest.mark.parametrize("plant", ["different-report", "missing-kernel", "wrong-symbol", "wrong-grid", "wrong-duration", "wrong-unit", "empty-sass", "no-counts"])
def test_native_detail_negatives(plant):
    d, expected = detail_fixture()
    a = d["actions"][0]
    if plant == "different-report":
        d["report_sha256"] = "2"*64
    elif plant == "missing-kernel":
        d["actions"] = []
    elif plant == "wrong-symbol":
        a["name"] = a["demangled"] = "other"
    elif plant == "wrong-grid":
        a["grid"] = "64,1,1"
    elif plant == "wrong-duration":
        a["metrics"]["ppu__time_duration.sum"]["value"] = 1
    elif plant == "wrong-unit":
        a["metrics"]["ppu__time_duration.sum"]["unit"] = "cycle"
    elif plant == "empty-sass":
        a["sass"] = []
    else:
        a["coverage"]["warp_executions"] = 0
    with pytest.raises(ValueError):
        exporter.check_detail(d, expected)


def test_report_hash_fails_before_import_or_loading(tmp_path):
    report = tmp_path / "test.acurep"
    report.write_bytes(b"not an admitted report")
    with pytest.raises(ValueError, match="SHA256"):
        reader.export_report(tmp_path / "no-api", report, "0"*64)


class Metric:
    def __init__(self, values, ids=None, kind=6):
        self.values, self.ids, self.kind_value = values, ids, kind
    def name(self): return "inst_executed"
    def kind(self): return self.kind_value
    def unit(self): return "inst"
    def has_value(self): return False
    def value(self): raise AssertionError("absent value must not become zero")
    def num_instances(self): return len(self.values)
    def as_uint64(self, i): return self.values[i]
    def as_double(self, i): return float(self.values[i])
    def as_string(self, i): return str(self.values[i])
    def has_correlation_ids(self): return self.ids is not None
    def correlation_ids(self): return Metric(self.ids)


def test_per_pc_correlation_ids_are_not_lost():
    record = reader.metric_record(Metric([2, 3], [4096, 4104]))
    assert record["value"] is None and record["has_value"] is False
    assert record["instances"] == [2, 3] and record["correlation_ids"] == [4096, 4104]
    with pytest.raises(ValueError, match="denominator"):
        reader.metric_record(Metric([2, 3], [4096]))
    with pytest.raises(ValueError, match="unsupported"):
        reader.metric_record(Metric([2], kind=0))


def test_work_audit_cannot_hide_a_missing_cell(monkeypatch):
    s = summary_fixture()
    del s["captures"][contract.cells()[-1]["id"]]
    monkeypatch.setattr(work, "files", lambda _: {"summary.json": json.dumps(s).encode()})
    with pytest.raises(ValueError, match="eight-cell"):
        work.analyze(b"synthetic")
