"""Measurement-boundary negatives; no fabricated PPU performance evidence."""
import csv
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "benchmarks"):
    sys.path.insert(0, str(path))
collect = importlib.import_module("collect_serving_prefill_acu")
profile = importlib.import_module("profile_serving_prefill")


def raw_csv(kernels, durations=None):
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(("ID", "Process ID", "Kernel Name", "Grid Size", "Block Size", "ppu__time_duration.sum"))
    for i, name in enumerate(kernels):
        writer.writerow((i, 123, name, "(128,1,1)", "(256,1,1)",
                         durations[i] if durations else (i + 1) * 1000))
    return output.getvalue()


def test_real_format_duration_all_kernels_and_units():
    names = (*collect.FLA, "void vectorized_elementwise_kernel<FillFunctor<float>>(int)")
    kernels = collect.parse_raw("tool preamble\n" + raw_csv(names))
    assert [row["us"] for row in kernels] == [1, 2, 3, 4, 5, 6]
    measured = collect.inventory("fla", kernels)
    assert measured["status"] == "ADMITTED" and measured["total_us"] == 21
    assert measured["core_in_same_call_us"] == 21, "fill kernels must not be dropped"


@pytest.mark.parametrize("plant", ["missing-stage", "duplicate-math", "unknown-kernel"])
@pytest.mark.parametrize("role", collect.ROLES)
def test_unknown_missing_or_replayed_work_never_gets_a_speed_verdict(plant, role):
    names = list(collect.FLA if role == "fla" else collect.CORE)
    if role == "serving":
        names += collect.TRANSPORT
    if plant == "missing-stage":
        names.pop(0)
    elif plant == "duplicate-math":
        names.append(names[0])
    else:
        names.append("unexpected_setup_or_autotune_kernel")
    measured = collect.inventory(role, collect.parse_raw(raw_csv(names)))
    assert measured["status"] == "UNADJUDICATED"
    assert len(measured["kernels"]) == len(names), "unknown work must remain visible"


@pytest.mark.parametrize("bad", ["nan", "inf", "0", "-1", ""])
def test_bad_or_missing_duration_is_red(bad):
    with pytest.raises(ValueError):
        collect.parse_raw(raw_csv(["kernel"], [bad]))


def test_wrong_counter_and_duplicate_launch_id_are_red():
    text = raw_csv(["a", "b"])
    with pytest.raises(ValueError, match="duration"):
        collect.parse_raw(text.replace("ppu__time_duration.sum", "ce__cycles_active.max"))
    with pytest.raises(ValueError, match="duplicate"):
        collect.parse_raw(text.replace('"1","123"', '"0","123"'))


def metric_csv(rows):
    stream = io.StringIO()
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(("ID", "Process ID", "Kernel Name", "Grid Size", "Block Size",
                     "Device", "Context", "Stream", "Section Name", "Metric Name",
                     "Metric Unit", "Metric Value"))
    writer.writerows(rows)
    return stream.getvalue()


def metric_row(index, metric="Duration", unit="nsecond", value="1000"):
    return [str(index), "1739642", "kernel" + str(index), "(1,32,1)", "(128,1,1)",
            "PPU-ZW810", "1", "7", "PPU Speed Of Light Throughput", metric, unit, value]


def test_long_metric_rows_are_one_launch_not_one_launch_per_counter():
    text = metric_csv([metric_row(0), metric_row(0, "Active Cycles", "cycle", "100000"),
                       metric_row(1, unit="usecond", value="2")])
    assert [row["us"] for row in collect.parse_raw(text)] == [1, 2]


def test_uploaded_site_csv_regression_not_a_self_generated_schema():
    # Actual 2026-11-06 upload; Duration/Active Cycles rows retained verbatim.
    # Archive SHA256 f31c5788f57057c246a030e0d11e666551e21c2930f94987a5c47e2061c06b7a.
    path = ROOT / "tests/data/serving_acu_long_duration.csv"
    measured = collect.inventory("serving", collect.parse_raw(path.read_text()))
    assert len(measured["kernels"]) == 9 and measured["status"] == "ADMITTED"
    assert measured["total_us"] == pytest.approx(3303.09823)
    assert measured["core_in_same_call_us"] == pytest.approx(169.76353)
    assert measured["transport_us"] == pytest.approx(3133.33470)


@pytest.mark.parametrize("plant", ["missing", "duplicate", "cycles", "wrong-section", "wrong-kernel", "nonfinite"])
def test_long_duration_fail_closed(plant):
    rows = [metric_row(0), metric_row(0, "Active Cycles", "cycle", "100000")]
    if plant == "missing":
        rows.append(metric_row(1, "Active Cycles", "cycle", "100000"))
    elif plant == "duplicate":
        rows.append(metric_row(0))
    elif plant == "cycles":
        rows[0][-2] = "cycle"
    elif plant == "wrong-section":
        rows[0][-4] = "unrelated Duration label"
    elif plant == "wrong-kernel":
        rows[1][2] = "different kernel under the same launch ID"
    else:
        rows[0][-1] = "NaN"
    with pytest.raises(ValueError):
        collect.parse_raw(metric_csv(rows))


def test_both_winners_overlap_and_scope():
    def captures(core, fla):
        return {"serving": [dict(total_us=200, transport_us=100, status="ADMITTED")] * 3,
                "default": [dict(total_us=t, status="ADMITTED") for t in core],
                "fla": [dict(total_us=t, status="ADMITTED") for t in fla]}
    for core, fla, expected in (([1, 2, 3], [4, 5, 6], "DEFAULT_CORE_WINS"),
                                ([4, 5, 6], [1, 2, 3], "FLA_CORE_WINS"),
                                ([1, 2, 4], [3, 4, 5], "UNRESOLVED")):
        result = collect.summarize(captures(core, fla), 3)
        assert result["core_verdict"] == expected
        assert result["serving_vs_fla"].startswith("NOT_COMPARABLE")
        assert result["serving_transport_us"] == [100] * 3
    incomplete = captures([1, 2, 3], [4, 5, 6])
    incomplete["fla"].pop()
    with pytest.raises(ValueError, match="missing"):
        collect.summarize(incomplete, 3)
    ambiguous = captures([1, 2, 3], [4, 5, 6])
    ambiguous["fla"][0]["status"] = "UNADJUDICATED"
    assert collect.summarize(ambiguous, 3)["core_verdict"] == "UNADJUDICATED"


def make_metadata_run(root, name):
    """Synthetic metadata, NOT an executable library or device admission."""
    directory = root / name
    (directory / "build").mkdir(parents=True)
    library = directory / "build/libactlize_la_serving.so"
    library.write_bytes(b"test metadata only, not native code")
    digest = collect.sha(library)
    binary = dict(library=str(library), target="ppu10", sha256=digest, source_sha="6" * 40)
    cases = [dict(case="ppu10-prefill-retained-pair", length=n, heads=f"16:{h}", output_state="RAW-BIT/PASS")
             for n in (64, 65, 129, 8192) for h in (32, 48)]
    native = dict(target="ppu10", status="PASS", library_sha256=digest, cases=cases)
    aux = dict(target="ppu10", status="PASS", binary_sha256=digest,
               cases=[dict(case="gdn-checked-prefill", checked_vs_unchecked="RAW-BIT/PASS")])
    for filename, content in (("binary.json", binary), ("results.json", native), ("aux-results.json", aux)):
        (directory / filename).write_text(json.dumps(content))
    (directory / "native-graph.log").write_text(
        "metadata fixture only\n[FlashNext native] PASS replays=8 fixed-addresses kernel_nodes=1\n")
    return directory


def test_unique_passed_run_and_ambiguity(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        collect.select_run(workspace=tmp_path)
    first = make_metadata_run(tmp_path, "actlize-la-flare-ppu10-first")
    assert collect.select_run(workspace=tmp_path)["directory"] == str(first)
    make_metadata_run(tmp_path, "actlize-la-flare-ppu10-second")
    with pytest.raises(ValueError, match="exactly one"):
        collect.select_run(workspace=tmp_path)
    assert collect.select_run(first)["directory"] == str(first)


@pytest.mark.parametrize("plant", ["library", "aux-fail", "missing-cell", "missing-checked-bits", "native-graph"])
def test_old_or_changed_library_cannot_reuse_pass_receipt(tmp_path, plant):
    directory = make_metadata_run(tmp_path, "fixture")
    if plant == "library":
        (directory / "build/libactlize_la_serving.so").write_bytes(b"changed")
    elif plant == "native-graph":
        (directory / "native-graph.log").write_text("failed before completion")
    else:
        path = directory / ("results.json" if plant == "missing-cell" else "aux-results.json")
        receipt = json.loads(path.read_text())
        if plant == "aux-fail":
            receipt["status"] = "FAIL"
        elif plant == "missing-cell":
            receipt["cases"].pop()
        else:
            del receipt["cases"][0]["checked_vs_unchecked"]
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        collect.read_run(directory)


def test_archive_excludes_big_operands_reports_and_never_follows_links(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "operands.pt").write_bytes(b"do not upload")
    (tmp_path / "role.report.acurep").write_bytes(b"keep original on box")
    archive = collect.small_archive(tmp_path)
    import tarfile
    with tarfile.open(archive) as tar:
        assert tar.getnames() == ["summary.json"]
    (tmp_path / "planted-link").symlink_to(tmp_path / "operands.pt")
    with pytest.raises(ValueError, match="source link"):
        collect.small_archive(tmp_path)


def test_fla_preserves_exact_nonzero_kv_state_and_activations():
    packed = tuple(torch.empty(shape, dtype=dtype) for shape, dtype in (
        ((1, 64, 16, 128), torch.bfloat16), ((1, 64, 16, 128), torch.bfloat16),
        ((1, 64, 32, 128), torch.bfloat16), ((1, 64, 32), torch.float32),
        ((1, 64, 32), torch.bfloat16)))
    initial = torch.arange(32 * 128 * 128).reshape(1, 32, 128, 128).float()
    def fla(q, k, v, g, beta, state_v_first=True, use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True, **kwargs):
        assert all(a is b for a, b in zip((q, k, v, g, beta), packed))
        assert kwargs["initial_state"] is initial
        assert kwargs["scale"] == 128 ** -.5 and kwargs["chunk_size"] == 64
        assert not kwargs["use_qk_l2norm_in_kernel"]
        assert not (state_v_first or use_beta_sigmoid_in_kernel or use_gate_in_kernel)
    profile.fla_call(fla, packed, "native", initial_state=initial)()


def test_capture_is_one_call_then_cpu_observation(monkeypatch):
    operations = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: operations.append("sync"))
    class Call:
        def __call__(self):
            operations.append("call")
            return 42
        def cpu_pair(self, result):
            assert result == 42
            operations.append("cpu-check")
            return result
    assert profile.capture(Call()) == 42
    assert operations == ["sync", "call", "sync", "cpu-check"]


def test_actual_public_default_and_route_without_device_claim(tmp_path, monkeypatch):
    from actlize_la._gdn_ppu10 import PPU10Backend
    from actlize_la.backends.loading import _file_image
    from actlize_la import gdn_residual_interface

    library = SimpleNamespace(path=tmp_path / "metadata-only.so")
    library.path.write_bytes(b"metadata fixture; no device code")
    backend = PPU10Backend.__new__(PPU10Backend)
    backend.library = library
    backend.image = _file_image(library.path)
    backend.manifest_sha256 = None
    backend.provider = "explicit-serving"
    backend.residual_full_chunk = Mock(return_value=("output", "final"))
    monkeypatch.setattr(gdn_residual_interface, "_backend", lambda: backend)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.delenv("GDN_QSA_WY_EXTENSION", raising=False)
    # Record this key with monkeypatch even when absent: DefaultCall sets it.
    # A no-op delenv on an absent key would leave that later assignment behind.
    monkeypatch.setenv("ACTLIZE_LA_SERVING_LIBRARY", str(library.path))
    packed = tuple(torch.empty(shape, dtype=dtype) for shape, dtype in (
        ((1, 64, 16, 128), torch.bfloat16), ((1, 64, 16, 128), torch.bfloat16),
        ((1, 64, 32, 128), torch.bfloat16), ((1, 64, 32), torch.float32),
        ((1, 64, 32), torch.bfloat16)))
    initial = torch.empty(1, 32, 128, 128)
    call = profile.DefaultCall(library, packed, initial)
    backend.residual_full_chunk.assert_not_called()  # Route audit launches nothing.
    assert call() == ("output", "final")
    assert backend.residual_full_chunk.call_args.args == (*packed, initial, True)
    assert call.route["delivery"] == "full-chunk"
    assert call.route["native_images"][0]["sha256"] == collect.sha(library.path)

    # Moving the production default to scalar must fail BEFORE a launch; the
    # benchmark must not silently force full-chunk and hide this regression.
    route = dict(call.route, delivery="scalar")
    monkeypatch.setattr(profile, "get_gdn_route_info", lambda *a, **kw: route)
    with pytest.raises(ValueError, match="default"):
        profile.DefaultCall(library, packed, initial)
    assert backend.residual_full_chunk.call_count == 1


def default_route(library="/synthetic/library.so", digest="binary"):
    return dict(api="gdn_forward", backend="ppu10", algorithm="residual",
                delivery="full-chunk", native_entry="actlize_la_ppu10_forward_prepared",
                native_provider="explicit-serving", output_final_state=True,
                state_layout="B,Hv,K,V", final_state_dtype="fp32",
                native_status="LOADED_FILES_VERIFIED_NOT_DEVICE_ADMITTED",
                native_images=[dict(path=library, sha256=digest)])


@pytest.mark.parametrize("field", ["algorithm", "delivery", "native_entry", "native_provider",
                                   "native_status", "state_layout", "final_state_dtype"])
def test_wrong_default_route_is_red(field):
    route = default_route()
    route[field] = "wrong"
    with pytest.raises(ValueError):
        collect.validate_default_route(route, "/synthetic/library.so", "binary")


@pytest.mark.parametrize("field", ["path", "sha256"])
def test_different_loaded_binary_is_red(field):
    route = default_route()
    route["native_images"][0][field] = "wrong"
    with pytest.raises(ValueError):
        collect.validate_default_route(route, "/synthetic/library.so", "binary")


def admission_receipt():
    return dict(status="PASS", numeric_repeats=8, serving_default_bits="PASS", device={"uuid": "fixture-only"},
                library_sha="binary", shape={"T": 2048}, gate=-.1, operands_sha="payload",
                default_route=default_route(),
                arms={role: dict(input_sha="raw" if role == "serving" else "packed",
                                 output_sha="same-native" if role != "fla" else "fla-output")
                      for role in collect.ROLES})


def subject_receipt(admission, role):
    return {**admission, **admission["arms"][role], "phase": "subject", "role": role,
            "public_calls": 1, "warmup": 0}


@pytest.mark.parametrize("plant", ["device", "library_sha", "operands_sha", "input_sha", "output_sha",
                                  "status", "public_calls", "warmup"])
def test_a_fast_report_cannot_replace_a_missing_or_changed_numerical_receipt(plant):
    admitted = admission_receipt()
    subject = subject_receipt(admitted, "default")
    collect.validate_receipts(admitted, subject, "default")
    subject[plant] = "wrong"
    with pytest.raises(ValueError):
        collect.validate_receipts(admitted, subject, "default")


def fake_profiler(tmp_path):
    home = tmp_path / "site-asight"
    (home / "bin").mkdir(parents=True)
    (home / "sections").mkdir()
    (home / "sections/fixture.section").write_text("synthetic section, not PPU evidence\n")
    tool = home / "bin/acu"
    tool.write_bytes(b"not executable; test replaces child process transport")
    return tool


def test_profiler_layout_hashes_sections_and_rejects_missing(tmp_path):
    tool = fake_profiler(tmp_path)
    before = collect.profiler_layout(tool)
    sections = tool.parent.parent / "sections"
    (sections / "fixture.section").write_text("changed schema")
    assert collect.profiler_layout(tool) != before
    broken = tmp_path / "broken/bin/acu"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"no companion sections")
    with pytest.raises(ValueError, match="companion sections"):
        collect.profiler_layout(broken)


def test_failed_process_prints_original_parser_error(tmp_path, capsys):
    log = tmp_path / "parse.log"
    with pytest.raises(RuntimeError, match="rc=101"):
        collect.run([sys.executable, "-c", "import sys; print('MemoryMetricTable schema mismatch'); sys.exit(101)"],
                    log, {}, console=False)
    assert "MemoryMetricTable schema mismatch" in capsys.readouterr().err
    assert "MemoryMetricTable schema mismatch" in log.read_text()


@pytest.mark.parametrize("plant", [None, "wrong-receipt", "duplicate-math", "section-schema"])
def test_collector_control_flow_not_device_execution(tmp_path, monkeypatch, plant):
    origin = make_metadata_run(tmp_path, "admitted")
    tool = fake_profiler(tmp_path)
    out = tmp_path / "output"
    monkeypatch.setenv("ASIGHT_HOME", "/different-sdk/asight")
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.setattr(collect, "DEFAULT_ACU", tool)
    monkeypatch.setattr(collect.shutil, "which", lambda name: None)
    monkeypatch.setattr(collect.subprocess, "check_output", lambda command, **kwargs:
                        "" if "diff" in command else "a" * 40)
    commands = []

    def transport(command, log, env, **kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        assert env["ASIGHT_HOME"] == str(tool.parent.parent)
        if "--list-sections" in command or "--set" in command or "--import" in command:
            assert command[command.index("--section-folder") + 1] == str(tool.parent.parent / "sections")
        log.write_text("synthetic transport; no device ran\n")
        if plant == "section-schema" and "--list-sections" in command:
            log.write_text("MemoryMetricTable schema mismatch; synthetic negative\n")
            raise RuntimeError("command failed (rc=101); synthetic schema negative")
        if "--phase" in command:
            phase = command[command.index("--phase") + 1]
            path = Path(command[command.index("--receipt") + 1])
            admission = admission_receipt()
            if phase == "admit":
                path.write_text(json.dumps(admission))
            else:
                role = command[command.index("--role") + 1]
                receipt = subject_receipt(admission, role)
                if plant == "wrong-receipt":
                    receipt["output_sha"] = "corrupt"
                path.write_text(json.dumps(receipt))
                report = Path(command[command.index("-o") + 1] + ".acurep")
                report.write_bytes(b"synthetic report identity only")
        elif "--import" in command:
            role = Path(command[command.index("--import") + 1]).name.split("-")[0]
            names = list(collect.FLA if role == "fla" else collect.CORE)
            if role == "serving":
                names += collect.TRANSPORT
            if plant == "duplicate-math":
                names += names[:1]
            log.write_text(raw_csv(names))
        return dict(status="COLLECTED", returncode=0)

    monkeypatch.setattr(collect, "run", transport)
    monkeypatch.setattr(sys, "argv", ["collect", "--serving-run", str(origin), "--lengths", "2048"])
    result = collect.main()
    assert result == (1 if plant in ("wrong-receipt", "section-schema") else 3 if plant == "duplicate-math" else 0)
    assert (out / "results.tar.gz").is_file(), "failure must preserve the small handoff"
    if plant == "section-schema":
        assert not any("--phase" in command for command in commands), "bad profiler ran numerical/device work"
    if plant is None:
        subjects = [command for command in commands if "--phase" in command
                    and command[command.index("--phase") + 1] == "subject"]
        assert len(subjects) == 9
        for position in range(3):
            assert {command[command.index("--role") + 1] for command in subjects[position::3]} == set(collect.ROLES)
