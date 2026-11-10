"""Work/inventory negatives; synthetic counters are never device evidence."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_wide_and_long_csv_instruction_counts_match_only_with_explicit_binding():
    fields, rows, kernels = csv_rows()
    wanted = work.instruction_rows(encode_csv(fields, rows), kernels)
    metric = "captured_instruction_counter.sum"
    wide_fields = [k for k in fields if k not in ("Section Name", "Metric Name", "Metric Unit", "Metric Value")] + [metric]
    wide_rows = [{**{k: row[k] for k in wide_fields[:-1]}, metric: row["Metric Value"]} for row in rows]
    wide = encode_csv(wide_fields, wide_rows)
    assert work.instruction_rows(wide, kernels, metric) == wanted
    for binding in (None, "different.sum"):
        with pytest.raises(ValueError, match="binding"):
            work.instruction_rows(wide, kernels, binding)


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
    reader.check_detail(detail, expected)


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
        reader.check_detail(d, expected)


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


def test_instruction_counter_binding_comes_from_capture_sections(tmp_path):
    section = tmp_path / "InstructionStatistics.section"
    section.write_text('Header { Metrics { Label: "Executed Instructions" Name: "actual_capture_counter.sum" } }')
    layout = dict(sections=str(tmp_path), section_sha256={section.name: reader.sha(section)})
    assert exporter.instruction_metric_id(layout) == "actual_capture_counter.sum"
    section.write_text('Header { Metrics { Label: "Other" Name: "wrong.sum" } }')
    with pytest.raises(ValueError, match="changed"):
        exporter.instruction_metric_id(layout)
    layout["section_sha256"][section.name] = reader.sha(section)
    with pytest.raises(ValueError, match="missing or ambiguous"):
        exporter.instruction_metric_id(layout)


@pytest.mark.parametrize("plant", [None, "count", "unit", "binding", "kernel", "denominator"])
def test_separate_reader_cannot_change_instruction_counts(plant):
    d, expected = detail_fixture()
    metric_id = "captured_counter.sum"
    metric = dict(value=1234, has_value=True, unit="inst")
    d["actions"][0]["metrics"][metric_id] = metric
    counts = [dict(kernel="kernel(int)", count=1234)]
    if plant == "count":
        metric["value"] = 1235
    elif plant == "unit":
        metric["unit"] = "inst/thread"
    elif plant == "binding":
        metric_id = "another_counter.sum"
    elif plant == "kernel":
        counts[0]["kernel"] = "different"
    elif plant == "denominator":
        counts = []
    if plant is None:
        reader.check_detail(d, expected, counts, metric_id)
    else:
        with pytest.raises(ValueError):
            reader.check_detail(d, expected, counts, metric_id)


def source_fixture():
    fields = ["Range Id", "Function Name", "Function Id", "Address", "Source",
              "Instructions Executed", "Thread Instructions Executed"]
    rows = [dict(zip(fields, ["0", "kernel", "0", "0x100", "v.mma.f32.bf16.m16n16k16 vreg0", "8", "256"])),
            dict(zip(fields, ["0", "helper", "1", "0x200", "s.wait 0x0", "4", "128"]))]
    kernels = [dict(id="0", kernel="void example::kernel<int>(int)")]
    return fields, rows, kernels


def test_source_parser_preserves_helpers_and_counter_namespace():
    fields, rows, kernels = source_fixture()
    parsed = exporter.source_counts(encode_csv(fields, rows), kernels)
    assert parsed["0"]["pc_warp_executions"] == 12
    assert parsed["0"]["pc_thread_executions"] == 384
    assert parsed["0"]["functions"] == ["helper", "kernel"]
    assert parsed["0"]["opcode_warp_counts"] == {"v.mma.f32.bf16.m16n16k16": 8, "s.wait": 4}


@pytest.mark.parametrize("plant", ["no-header", "missing-action", "extra-action", "duplicate-pc", "wrong-symbol", "empty-op", "negative-count", "empty-count", "zero-execution"])
def test_cli_source_negatives(plant):
    fields, rows, kernels = source_fixture()
    if plant == "no-header":
        text = "not a source table"
    else:
        if plant == "missing-action":
            kernels.append(dict(id="1", kernel="other(int)"))
        elif plant == "extra-action":
            rows[0]["Range Id"] = "2"
        elif plant == "duplicate-pc":
            rows.append(rows[0])
        elif plant == "wrong-symbol":
            rows[0]["Function Name"] = "different"
        elif plant == "empty-op":
            rows[0]["Source"] = ""
        elif plant == "negative-count":
            rows[0]["Instructions Executed"] = "-1"
        elif plant == "empty-count":
            rows[0]["Instructions Executed"] = ""
        elif plant == "zero-execution":
            for row in rows:
                row["Instructions Executed"] = "0"
        text = encode_csv(fields, rows)
    with pytest.raises(ValueError):
        exporter.source_counts(text, kernels)


def test_cli_import_works_without_companion_api(tmp_path, monkeypatch):
    capture_home = tmp_path / "capture"
    capture_home.mkdir()
    with pytest.raises(ValueError, match="API unavailable"):
        reader.api_identity(capture_home)  # exact failure in the uploaded490B package
    fields, rows, kernels = source_fixture()
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert command == ["acu", "--section-folder", "sections", "--import", "existing.acurep",
                           "--csv", "--page", "source"]
        kwargs["stdout"].write(encode_csv(fields, rows))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(exporter.subprocess, "run", run)
    output = tmp_path / "native.csv"
    exporter.export_page(["acu", "--section-folder", "sections"], Path("existing.acurep"), "source", output, {})
    assert len(commands) == 1
    assert exporter.source_counts(output.read_text(), kernels)["0"]["pc_warp_executions"] == 12


@pytest.mark.parametrize("plant", ["rc", "empty", "protobuf-stdout", "protobuf-stderr"])
def test_cli_failures_cannot_become_empty_success(tmp_path, monkeypatch, plant):
    message = "Error parsing text-format HG.Profiler.ProfilerSection"
    def run(command, **kwargs):
        if plant != "empty":
            kwargs["stdout"].write(message if plant == "protobuf-stdout" else "data")
        if plant == "protobuf-stderr":
            kwargs["stderr"].write(message)
        return SimpleNamespace(returncode=1 if plant == "rc" else 0)
    monkeypatch.setattr(exporter.subprocess, "run", run)
    with pytest.raises((ValueError, RuntimeError)):
        exporter.export_page(["acu"], Path("existing.acurep"), "source", tmp_path / "page.csv", {})


def test_old_cli_identical_metric_pages_are_not_native_instructions():
    # Uploaded9940b34b: raw/source/details were byte-identical aggregate rows.
    fields, rows, kernels = csv_rows()
    pages = {p: encode_csv(fields, rows) for p in ("raw", "source", "details")}
    assert len(set(pages.values())) == 1
    with pytest.raises(ValueError, match="metric rows are not native instructions"):
        exporter.source_counts(pages["source"], kernels)


def native_rule_fixture(tmp_path, monkeypatch, *, modern=False, broken=False):
    """Run the actual embedded rule with a fake native context; no SDK/device."""
    path = tmp_path / "native.csv"
    names = SimpleNamespace(NameBase_DEMANGLED=1, NameBase_MANGLED=2)
    metrics = {
        "ppu__time_duration.sum": SimpleNamespace(has_value=lambda: True, value=lambda: 4200., unit=lambda: "nsecond"),
        "captured_counter.sum": SimpleNamespace(has_value=lambda: True, value=lambda: 1234., unit=lambda: "inst"),
    }
    insn = SimpleNamespace(address=256, function_id=8675309, sass_code="v.mma.f32.bf16.m16n16k16 vreg0",
                           warp_executed_count=8, thread_executed_count=256)
    helper = SimpleNamespace(address=512, function_id=123, sass_code="s.wait 0x0",
                             warp_executed_count=4, thread_executed_count=128)
    native = {256: insn, 512: helper}
    def action(i):
        return SimpleNamespace(
            sass=lambda: {} if broken and i == 1 else native,
            functions=lambda: {8675309: "kernel", 123: "helper"},
            name=lambda base=0: ("kernel", "kernel(int)", "_Zkernel_i")[base],
            metric_by_name=lambda key: metrics[key], capture_sequence_id=lambda: i,
            grid_size=lambda: "32,1,1", block_size=lambda: "128,1,1")
    context = SimpleNamespace(num_ranges=lambda: 2, range_by_idx=lambda i: SimpleNamespace(
        num_actions=lambda: 1, action_by_idx=lambda j: action(i)))
    module = SimpleNamespace(get_output_file_path=lambda h: str(path), get_context=lambda h: context,
                             IAction=names if modern else SimpleNamespace(NameBase=names))
    monkeypatch.setitem(sys.modules, "HgRules", module)
    monkeypatch.setenv("ACTLIZE_ACU_INSTRUCTION_METRIC", "captured_counter.sum")
    spec = importlib.util.spec_from_file_location("test_embedded_work_rule", ROOT / "tools/acu_work_rule.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    kernels = [dict(id=str(i), kernel="kernel(int)", grid="(32,1,1)", block="(128,1,1)", us=4.2) for i in range(2)]
    counts = [dict(kernel="kernel(int)", count=1234) for i in range(2)]
    return rule, path, kernels, counts


@pytest.mark.parametrize("modern", [False, True])
def test_embedded_rule_old_and_new_name_api_all_actions_helpers_and_blank_optional(tmp_path, monkeypatch, modern):
    rule, path, kernels, counts = native_rule_fixture(tmp_path, monkeypatch, modern=modern)
    rule.apply(None)
    groups = exporter.rule_counts(path, kernels, counts, "captured_counter.sum")
    assert set(groups) == {"0", "1"}
    assert all(g["pc_warp_executions"] == 12 for g in groups.values())
    assert all(g["functions"] == ["helper", "kernel"] for g in groups.values())
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 4 and rows[0]["Function Id"] == "8675309"
    assert rows[0]["achieved_access_data_bytes"] == rows[0]["stall_total"] == ""


@pytest.mark.parametrize("plant", ["no-marker", "denominator", "dropped-row", "grid", "symbol", "duration", "unit", "count", "binding"])
def test_rule_output_cannot_hide_partial_or_changed_native_data(tmp_path, monkeypatch, plant):
    rule, path, kernels, counts = native_rule_fixture(tmp_path, monkeypatch)
    rule.apply(None)
    marker = Path(str(path) + ".complete")
    rows = list(csv.DictReader(path.open()))
    fields = list(rows[0])
    if plant == "no-marker":
        marker.unlink()
    elif plant == "denominator":
        marker.write_text("1,4\n")
    elif plant == "dropped-row":
        rows.pop()
    else:
        label, value = {
            "grid": ("Grid Size", "64,1,1"), "symbol": ("Kernel Name", "other(int)"),
            "duration": ("Duration", "4201"), "unit": ("Duration Unit", "cycle"),
            "count": ("Aggregate Instructions", "1235"), "binding": ("Instruction Metric", "other.sum"),
        }[plant]
        rows[0][label] = value
    path.write_text(encode_csv(fields, rows))
    with pytest.raises(ValueError):
        exporter.rule_counts(path, kernels, counts, "captured_counter.sum")


def test_rule_failure_after_valid_action_does_not_publish_completion(tmp_path, monkeypatch):
    rule, path, kernels, counts = native_rule_fixture(tmp_path, monkeypatch, broken=True)
    with pytest.raises(ValueError, match="missing native"):
        rule.apply(None)
    assert path.is_file() and not Path(str(path) + ".complete").exists()
    with pytest.raises(ValueError, match="did not complete"):
        exporter.rule_counts(path, kernels, counts, "captured_counter.sum")


@pytest.mark.parametrize("swallowed_error", [False, True])
def test_native_export_is_offline_and_rejects_rc0_rule_failure(tmp_path, monkeypatch, swallowed_error):
    rule, path, kernels, counts = native_rule_fixture(tmp_path, monkeypatch)
    def run(command, **kwargs):
        assert command == ["acu", "--section-folder", "sections", "--import", "existing.acurep",
                           "--stats-rule", str(ROOT / "tools/acu_work_rule.py"), "--stats-rule-output", str(path)]
        assert kwargs["env"]["ACTLIZE_ACU_INSTRUCTION_METRIC"] == "captured_counter.sum"
        if swallowed_error:
            kwargs["stdout"].write("[Asight]: load python script failed\n")
        else:
            rule.apply(None)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(exporter.subprocess, "run", run)
    if swallowed_error:
        with pytest.raises(ValueError, match="load python script failed"):
            exporter.export_rule(["acu", "--section-folder", "sections"], Path("existing.acurep"),
                                 path, {}, kernels, counts, "captured_counter.sum")
    else:
        assert len(exporter.export_rule(["acu", "--section-folder", "sections"], Path("existing.acurep"),
                                       path, {}, kernels, counts, "captured_counter.sum")) == 2


def test_existing_native_evidence_is_not_overwritten(tmp_path, monkeypatch):
    rule, path, kernels, counts = native_rule_fixture(tmp_path, monkeypatch)
    receipt = path.with_suffix(".command.json")
    receipt.write_text("original identity")
    def unexpected(*args, **kwargs):
        raise AssertionError("must reject before running the CLI")
    monkeypatch.setattr(exporter.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="already exists"):
        exporter.export_rule(["acu"], Path("existing.acurep"), path, {}, kernels, counts, "captured_counter.sum")
    assert receipt.read_text() == "original identity"
