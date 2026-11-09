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


@pytest.mark.parametrize('role', ['default', 'serving'])
def test_frozen_value64_default_inventory_and_wrong_thread_negative(role):
    names = [*collect.CORE[:2], 'gdn_wy_value64_state', collect.CORE[3]]
    if role == 'serving':
        names += list(collect.TRANSPORT)
    rows = collect.parse_raw(raw_csv(names))
    for row in rows:
        row['grid'] = '(192,1,1)' if 'gdn_wy_value64_state' in row['kernel'] else '(12288,1,1)'
    assert collect.inventory(role, rows)['status'] == 'ADMITTED'
    wrong = [dict(row, block='(512,1,1)') if 'gdn_wy_value64_state' in row['kernel'] else row for row in rows]
    assert collect.inventory(role, wrong)['status'] == 'UNADJUDICATED'
    wrong = [dict(row, grid='(384,1,1)') if 'gdn_wy_value64_state' in row['kernel'] else row for row in rows]
    assert collect.inventory(role, wrong)['status'] == 'UNADJUDICATED'


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
             for n in (64, 65, 129, 8192) for h in collect.RETAINED_PAIR_HEADS]
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


def admission_receipt(measured=None):
    return dict(status="PASS", numeric_repeats=8, serving_default_bits="PASS", device={"uuid": "fixture-only"},
                library_sha="binary", shape={"T": 2048}, gate=-.1, operands_sha="payload",
                measured_shape=measured or dict(B=1, T=2048, Hk=16, Hv=32, K=128, V=128),
                default_route=default_route(),
                arms={role: dict(input_sha="raw" if role == "serving" else "packed",
                                 output_sha="same-native" if role != "fla" else "fla-output")
                      for role in collect.ROLES})


def subject_receipt(admission, role):
    return {**admission, **admission["arms"][role], "phase": "subject", "role": role,
            "public_calls": 1, "warmup": 0}


def saved_core_fixture(root, name):
    """Metadata-only fixture; neither a real DSO nor numerical evidence."""
    directory = root / name
    directory.mkdir()
    library = directory / "metadata.so"
    library.write_bytes(b"synthetic library identity only")
    (directory / "operands.pt").write_bytes(b"synthetic payload identity only")
    admission = admission_receipt()
    admission.update(shape=dict(B=1, T=8192, Hk=16, Hv=48, K=128, V=128),
                     library_sha=collect.sha(library),
                     operands_sha=collect.sha(directory / "operands.pt"),
                     default_route=default_route(str(library), collect.sha(library)))
    (directory / "admission.json").write_text(json.dumps(admission))
    (directory / "core.json").write_text(json.dumps(subject_receipt(admission, "default")))
    return directory


def test_saved_flashnext_operands_unique_selection_and_ambiguity(tmp_path):
    first = saved_core_fixture(tmp_path, "flashnext-gdn-core-first")
    assert collect.select_core_input_run(workspace=tmp_path)["directory"] == str(first)
    saved_core_fixture(tmp_path, "flashnext-gdn-core-second")
    with pytest.raises(ValueError, match="exactly one"):
        collect.select_core_input_run(workspace=tmp_path)
    assert collect.select_core_input_run(first)["directory"] == str(first)


@pytest.mark.parametrize("plant", ["shape", "gate", "payload", "library", "no-subject", "subject-device"])
def test_saved_flashnext_run_cannot_silently_drift(tmp_path, plant):
    directory = saved_core_fixture(tmp_path, "core")
    if plant in ("payload", "library"):
        name = "operands.pt" if plant == "payload" else "metadata.so"
        (directory / name).write_bytes(b"changed bytes")
    elif plant == "no-subject":
        (directory / "core.json").unlink()
    else:
        name = "core.json" if plant == "subject-device" else "admission.json"
        path = directory / name
        record = json.loads(path.read_text())
        if plant == "shape":
            record["shape"]["T"] = 2048
        elif plant == "gate":
            record["gate"] = -1.
        else:
            record["device"] = {"uuid": "different physical device"}
        path.write_text(json.dumps(record))
    with pytest.raises((ValueError, OSError)):
        collect.core_input_run(directory)


def test_core_and_fla_shell_embedded_python_parses():
    import ast
    import subprocess
    script = ROOT / "tools/run_ppu_flashnext_core_acu_box.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "tools/run_ppu_flashnext_fla_acu_box.sh")], check=True)
    bodies = script.read_text().split("<<'PY'\n")[1:]
    assert len(bodies) == 2
    for body in bodies:
        ast.parse(body.split("\nPY\n", 1)[0])


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


@pytest.mark.parametrize("plant", [None, "wrong-receipt", "duplicate-math", "section-schema",
                                   "relabeled-shape", "confirm"])
def test_collector_control_flow_not_device_execution(tmp_path, monkeypatch, plant):
    origin = make_metadata_run(tmp_path, "admitted")
    tool = fake_profiler(tmp_path)
    out = tmp_path / "output"
    monkeypatch.setenv("ASIGHT_HOME", "/different-sdk/asight")
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.delenv("ACU", raising=False)
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
            hv = int(command[command.index("--value-heads") + 1])
            measured = dict(B=int(command[command.index("--batch") + 1]),
                            T=int(command[command.index("--length") + 1]), Hk=16,
                            Hv=48 if plant == "relabeled-shape" else hv, K=128, V=128)
            admission = admission_receipt(measured)
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
            # default core = 10 us; FLA = 12.5 us puts the ratio at 0.80 exactly.
            durations = [2500] * len(names) if plant == "confirm" and role == "fla" else None
            log.write_text(raw_csv(names, durations))
        return dict(status="COLLECTED", returncode=0)

    monkeypatch.setattr(collect, "run", transport)
    argv = ["collect", "--serving-run", str(origin), "--lengths", "2048"]
    if plant == "relabeled-shape":
        argv += ["--value-heads", "64"]
    monkeypatch.setattr(sys, "argv", argv)
    result = collect.main()
    failing = ("wrong-receipt", "section-schema", "relabeled-shape")
    assert result == (1 if plant in failing else 3 if plant == "duplicate-math" else 0)
    assert (out / "results.tar.gz").is_file(), "failure must preserve the small handoff"
    if plant == "section-schema":
        assert not any("--phase" in command for command in commands), "bad profiler ran numerical/device work"
    if plant == "relabeled-shape":
        assert not any("--phase" in command and command[command.index("--phase") + 1] == "subject"
                       for command in commands), "relabeled cell reached device timing"
    subjects = [command for command in commands if "--phase" in command
                and command[command.index("--phase") + 1] == "subject"]
    if plant is None:
        assert len(subjects) == 9
        for position in range(3):
            assert {command[command.index("--role") + 1] for command in subjects[position::3]} == set(collect.ROLES)
    if plant == "confirm":
        summary = json.loads((out / "summary.json").read_text())
        cell = summary["captures"]["T2048-Hk16-Hv32-g-0.1"]["summary"]
        assert len(subjects) == 18 and cell["repeats"] == 6 and cell["confirmation_round"]
        assert cell["median_ratio_default_over_fla"] == pytest.approx(0.8)


matrix = importlib.import_module("summarize_gdn_matrix")
REGISTERED = ROOT / "benchmarks/gdn_prefill_matrix.json"
REGISTERED_SHA = "48a7aa907067cf2a966594c899da9b9882316950ddd09d39a292e4810f5d7402"


def legacy_raw_fixture(length, hv, gate):
    """Verbatim pre-extension fixture (fece3fd), kept to pin B=1 bytes."""
    import math
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


@pytest.mark.parametrize("hv,gate", [(32, -.1), (48, -1.)])
def test_b1_fixture_bytes_unchanged_by_the_extension(hv, gate):
    new, old = profile.raw_fixture(64, hv, gate), legacy_raw_fixture(64, hv, gate)
    assert new.keys() == old.keys()
    for name in old:
        assert new[name].dtype == old[name].dtype and torch.equal(new[name], old[name]), name


@pytest.mark.parametrize("batch", [2, 4])
def test_batched_fixture_is_really_batched_not_a_label(batch):
    raw = profile.raw_fixture(16, 64, -.1, batch)
    assert raw["mixed"].shape == (batch * 16, (32 + 64) * 128)
    assert raw["starts"].tolist() == [16 * r for r in range(batch + 1)]
    assert raw["live"].tolist() == [batch] and profile.request_geometry(raw) == (batch, 16)
    assert raw["initial_slots"].tolist() == list(range(1, batch + 1))
    assert raw["final_slots"].tolist() == list(range(batch + 1, 2 * batch + 1))
    assert raw["state"].shape[0] == 2 * batch + 1 and not raw["state"][0].any()
    initial = raw["state"][1:batch + 1]
    assert all(not torch.equal(initial[0], initial[r]) for r in range(1, batch)), "distinct request states"
    assert not raw["state"][batch + 1:].any()


def test_batched_raw_reference_is_independent_per_request():
    raw = profile.raw_fixture(9, 32, -1., 2)
    output, state = profile.raw_reference(raw)
    assert output.shape == (2, 9, 32, 128) and state.shape == (2, 32, 128, 128)
    for row in range(2):
        lo, hi = 9 * row, 9 * row + 9
        want, final, *_ = profile.gdn_reference(raw["mixed"][lo:hi], raw["a"][lo:hi], raw["b"][lo:hi],
                                               raw["a_log"], raw["bias"], raw["state"][1 + row], 16)
        assert torch.equal(output[row], want)
        assert torch.equal(state[row], final.transpose(-1, -2).contiguous()), "explicit VK->KV transpose"


def test_unsupported_or_mislabeled_fixture_is_red():
    for args in ((64, 40, -.1), (64, 32, -.5), (64, 32, -.1, 3), (0, 32, -.1)):
        with pytest.raises(ValueError):
            profile.raw_fixture(*args)
    raw = profile.raw_fixture(8, 32, -.1, 2)
    raw["starts"] = torch.tensor([0, 8, 15], dtype=torch.int32)
    with pytest.raises(ValueError, match="equal-length"):
        profile.request_geometry(raw)


def test_core_shape_is_measured_from_tensors():
    packed = (torch.empty(2, 64, 16, 128), torch.empty(2, 64, 16, 128), torch.empty(2, 64, 64, 128),
              torch.empty(2, 64, 64), torch.empty(2, 64, 64))
    assert profile.core_shape(packed) == dict(B=2, T=64, Hk=16, Hv=64, K=128, V=128)
    with pytest.raises(ValueError):
        profile.core_shape((*packed[:2], torch.empty(2, 64, 48, 128), *packed[3:]))


def test_collector_and_fixture_registered_axes_agree():
    for name in ("BATCHES", "LENGTHS", "VALUE_HEADS", "GATES"):
        assert getattr(collect, name) == getattr(profile, name), name


@pytest.mark.parametrize("default,fla,want", [
    ([70, 71, 72], [100, 101, 102], "TARGET_MET"),        # 0.70 and disjoint
    ([79, 80, 101], [100, 101, 102], "UNRESOLVED"),      # passing median, overlapping envelope
    ([81, 82, 83], [100, 101, 102], "TARGET_NOT_MET"),   # 1.23x: faster than FLA, not 20% less time
    ([83, 83.3, 83.5], [100, 100, 100], "TARGET_NOT_MET"),  # the 1.20x speed-ratio trap
    ([110, 111, 112], [100, 101, 102], "TARGET_NOT_MET"),
    ([90, 110, 120], [100, 101, 102], "UNRESOLVED"),   # 未过中位数且包络重叠，仍不得确定排名
])
def test_registered_target_is_time_reduction_not_speed_ratio(default, fla, want):
    assert collect.target_verdict(default, fla, True)[0] == want
    assert collect.target_verdict(default, fla, False)[0] == "UNADJUDICATED"


@pytest.mark.parametrize("default,fla,confirm", [
    ([80, 80, 80], [100, 100, 100], True),     # ratio 0.80 is inside the pre-registered band
    ([70, 70, 70], [100, 106, 100], True),     # FLA spread 6% > 5%
    ([70, 71, 70], [100, 101, 100], False),
])
def test_pre_registered_confirmation_rule(default, fla, confirm):
    rows = dict(serving=[1, 1, 1], default=default, fla=fla)
    ratio = collect.target_verdict(default, fla, True)[1]
    assert collect.needs_confirmation(rows, ratio)[0] is confirm


def test_registered_manifest_denominator_and_hash():
    assert collect.sha(REGISTERED) == REGISTERED_SHA
    data = json.loads(REGISTERED.read_text())
    cells = data["cells"]
    assert len(cells) == 54 and len({c["id"] for c in cells}) == 54
    assert len({(c["geometry"], c["B"], c["T"]) for c in cells}) == 27
    assert {c["Hv"] for c in cells} == {32, 48, 64} and {c["B"] for c in cells} == {1, 2, 4}
    phase1, _ = collect.registered_cells(REGISTERED, "1", expected_sha=REGISTERED_SHA)
    assert len(phase1) == 12 and all(c["B"] == 1 and c["T"] in (2048, 8192) for c in phase1)
    assert len(collect.registered_cells(REGISTERED, "all")[0]) == 54


def test_edited_or_unknown_denominator_is_red(tmp_path):
    with pytest.raises(ValueError, match="changed"):
        collect.registered_cells(REGISTERED, "1", expected_sha="0" * 64)
    with pytest.raises(ValueError):
        collect.registered_cells(REGISTERED, ids=["G64-B1-T2048-g-0.1", "G99-B1-T1-g-0.1"])
    edited = tmp_path / "edited.json"
    data = json.loads(REGISTERED.read_text())
    data["cells"] = [c for c in data["cells"] if c["id"] != "G64-B1-T8192-g-1.0"]   # drop a losing cell
    edited.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        matrix.matrix(edited, REGISTERED_SHA, [])


def collected(tmp_path, entries, name="run.json"):
    """Synthetic collector summary for matrix logic only; not device evidence."""
    captures = {}
    for key, measured, verdict in entries:
        cell = next(c for c in json.loads(REGISTERED.read_text())["cells"] if c["id"] == key)
        spec = {k: cell[k] for k in ("B", "T", "Hk", "Hv", "K", "V", "gate")}
        rows = {role: dict(median_us=v, min_us=v, max_us=v, samples_us=[v] * 3)
                for role, v in (("serving", 300.), ("default", 70.), ("fla", 100.))}
        captures[key] = dict(cell=spec, device_uuid="uuid-fixture",
                             measured_shape=measured or {k: cell[k] for k in ("B", "T", "Hk", "Hv", "K", "V")},
                             summary=dict(target_verdict=verdict, rows=rows, repeats=3, core_verdict="DEFAULT_CORE_WINS",
                                          median_ratio_default_over_fla=.7, speedup_fla_over_default=1 / .7,
                                          gap_to_target_us=-10.))
    path = tmp_path / name
    path.write_text(json.dumps(dict(manifest_sha256=REGISTERED_SHA, captures=captures)))
    return path


def test_matrix_keeps_every_registered_cell_and_never_averages(tmp_path):
    run = collected(tmp_path, [("G32-B1-T2048-g-0.1", None, "TARGET_MET"),
                               ("G64-B1-T2048-g-0.1", None, "TARGET_NOT_MET")])
    result = matrix.matrix(REGISTERED, REGISTERED_SHA, [run], "uuid-fixture")
    assert result["denominator"] == 54 and len(result["cells"]) == 54
    assert result["counts"]["NOT_RUN"] == 52 and result["counts"]["TARGET_MET"] == 1
    assert not result["phase1_complete"] and not result["all_met"]
    assert "mean" not in json.dumps({k: v for k, v in result.items() if k != "cells"})


@pytest.mark.parametrize("plant", ["hv64-as-hv48", "b2-label-only", "unregistered", "duplicate", "device"])
def test_matrix_relabel_duplicate_or_foreign_device_is_red(tmp_path, plant):
    key = "G64-B1-T8192-g-0.1" if plant == "hv64-as-hv48" else "G32-B2-T2048-g-1.0"
    measured = None
    if plant == "hv64-as-hv48":
        measured = dict(B=1, T=8192, Hk=16, Hv=48, K=128, V=128)
    elif plant == "b2-label-only":
        measured = dict(B=1, T=2048, Hk=16, Hv=32, K=128, V=128)
    runs = [collected(tmp_path, [(key, measured, "TARGET_MET")])]
    if plant == "unregistered":
        data = json.loads(runs[0].read_text())
        data["captures"]["G32-B8-T2048-g-1.0"] = data["captures"].pop(key)
        runs[0].write_text(json.dumps(data))
    elif plant == "duplicate":
        runs.append(collected(tmp_path, [(key, None, "TARGET_MET")], "again.json"))
    uuid = "other-device" if plant == "device" else "uuid-fixture"
    with pytest.raises(ValueError):
        matrix.matrix(REGISTERED, REGISTERED_SHA, runs, uuid)


def test_direct_pack_inventory_counts_all_helpers_not_free_core():
    kernels = collect.CORE + collect.DIRECT_TRANSPORT
    captured = collect.inventory("serving", collect.parse_raw(raw_csv(kernels)))
    assert captured["status"] == "ADMITTED"
    assert captured["transport_layout"] == "direct-pack"
    assert captured["transport_us"] == sum(range(5, 9))
    assert captured["total_us"] == sum(range(1, 9))
    for extra in ("pack_prefill", "prepare_gdn", "unexpected_setup_or_autotune"):
        wrong = collect.inventory("serving", collect.parse_raw(raw_csv(kernels + (extra,))))
        assert wrong["status"] == "UNADJUDICATED", "混合旧新/未知kernel不能过滤为通过"
    bad = collect.inventory("serving", collect.parse_raw(raw_csv(kernels + ("prepare_pack_prefill",))))
    assert bad["status"] == "UNADJUDICATED"
