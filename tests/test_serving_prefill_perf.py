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


def test_both_winners_overlap_and_scope():
    def captures(core, fla):
        return {"serving": [dict(total_us=200, transport_us=100, status="ADMITTED")] * 3,
                "retained": [dict(total_us=t, status="ADMITTED") for t in core],
                "fla": [dict(total_us=t, status="ADMITTED") for t in fla]}
    for core, fla, expected in (([1, 2, 3], [4, 5, 6], "RETAINED_CORE_WINS"),
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


def test_retained_host_abi_pointer_order_without_device_claim(monkeypatch):
    raw_empty = torch.empty
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch, "empty", lambda *shape, **kw: raw_empty(*shape, **(kw | {"device": "cpu"})))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=179))
    packed = tuple(raw_empty(shape, dtype=dtype) for shape, dtype in (
        ((1, 64, 16, 128), torch.bfloat16), ((1, 64, 16, 128), torch.bfloat16),
        ((1, 64, 32, 128), torch.bfloat16), ((1, 64, 32), torch.float32),
        ((1, 64, 32), torch.bfloat16)))
    initial = raw_empty(1, 32, 128, 128)
    launch, configure = Mock(return_value=0), Mock(return_value=0)
    library = SimpleNamespace(lib=SimpleNamespace(actlize_la_ppu10_forward_prepared=launch,
                                                 actlize_la_ppu10_configure=configure))
    call = profile.RetainedCall(library, packed, initial)
    call()
    pointers = [value.data_ptr() for value in (*packed, initial, call.output, call.final,
                 *(call.work[name] for name in ("inverse", "snapshots", "vnew", "gates")))]
    assert launch.call_args.args == (*pointers, 1, 64, 16, 32, True, 179)
    configure.assert_called_once()
    call()
    configure.assert_called_once()  # Configuration is not in the measured call.


def admission_receipt():
    return dict(status="PASS", numeric_repeats=8, serving_core_bits="PASS", device={"uuid": "fixture-only"},
                library_sha="binary", shape={"T": 2048}, gate=-.1, operands_sha="payload",
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
    subject = subject_receipt(admitted, "retained")
    collect.validate_receipts(admitted, subject, "retained")
    subject[plant] = "wrong"
    with pytest.raises(ValueError):
        collect.validate_receipts(admitted, subject, "retained")


@pytest.mark.parametrize("plant", [None, "wrong-receipt", "duplicate-math"])
def test_collector_control_flow_not_device_execution(tmp_path, monkeypatch, plant):
    origin = make_metadata_run(tmp_path, "admitted")
    tool = tmp_path / "acu"
    tool.write_bytes(b"not executable; test replaces child process transport")
    out = tmp_path / "output"
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.setattr(collect, "DEFAULT_ACU", tool)
    monkeypatch.setattr(collect.shutil, "which", lambda name: None)
    monkeypatch.setattr(collect.subprocess, "check_output", lambda command, **kwargs:
                        "" if "diff" in command else "a" * 40)
    commands = []

    def transport(command, log, env, **kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        log.write_text("synthetic transport; no device ran\n")
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
    assert result == (1 if plant == "wrong-receipt" else 3 if plant == "duplicate-math" else 0)
    assert (out / "results.tar.gz").is_file(), "failure must preserve the small handoff"
    if plant is None:
        subjects = [command for command in commands if "--phase" in command
                    and command[command.index("--phase") + 1] == "subject"]
        assert len(subjects) == 9
        for position in range(3):
            assert {command[command.index("--role") + 1] for command in subjects[position::3]} == set(collect.ROLES)
