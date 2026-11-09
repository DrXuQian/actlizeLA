"""Host-only adapter/denominator/inventory negatives; no PPU performance claim."""
import copy
import fcntl
import importlib
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "benchmarks"):
    sys.path.insert(0, str(path))
contract = importlib.import_module("gdn_threeway_contract")
profile = importlib.import_module("profile_gdn_threeway")
collector = importlib.import_module("collect_gdn_threeway_acu")
host_cxx = importlib.import_module("ppu_pla_host_cxx")


def fixture():
    # Small T for adapter tests; the actual measurement denominator is fixed.
    g = torch.Generator().manual_seed(43)
    q = torch.randn(1, 3, 16, 128, generator=g).bfloat16()
    k = torch.randn(1, 3, 16, 128, generator=g).bfloat16()
    v = torch.randn(1, 3, 48, 128, generator=g).bfloat16()
    log_g = torch.full((1, 3, 48), -.1)
    beta = torch.rand(1, 3, 48, generator=g).bfloat16()
    state = torch.randn(1, 48, 128, 128, generator=g)
    return (q, k, v, log_g, beta), state


def test_beta_promotion_and_state_are_not_reprocessed():
    packed, state = fixture()
    before = profile.digest((*packed, state))
    adapted, actual_state = profile.pla_operands(packed, state)
    assert actual_state is state
    assert all(a is b for a, b in zip(packed[:4], adapted[:4]))
    assert adapted[4].dtype == torch.float32
    assert torch.equal(adapted[4], packed[4].double()), "all BF16 beta values must promote exactly"
    assert profile.digest((*packed, state)) == before
    assert not torch.equal(state, state.transpose(-1, -2))
    # A one-FP32-ULP corruption hides if we cast back to BF16 before hashing.
    altered = adapted[4].clone()
    altered.view(torch.int32).flatten()[0] += 1
    assert torch.equal(altered.bfloat16(), adapted[4].bfloat16())
    assert profile.digest((altered,)) != profile.digest((adapted[4],))


@pytest.mark.parametrize("plant", ["beta-dtype", "g-dtype", "state-dtype", "state-shape"])
def test_input_contract_cannot_silently_cast_or_reshape(plant):
    packed, state = fixture()
    packed = list(packed)
    if plant == "beta-dtype":
        packed[4] = packed[4].float()
    elif plant == "g-dtype":
        packed[3] = packed[3].bfloat16()
    elif plant == "state-dtype":
        state = state.bfloat16()
    else:
        state = state[:, :-1]
    with pytest.raises(ValueError):
        profile.pla_operands(packed, state)


def test_pla_public_call_keeps_scale_state_and_native_default():
    packed, state = fixture()
    adapted, state = profile.pla_operands(packed, state)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    fn = Mock(return_value="sentinel")
    call = profile.pla_call(fn, adapted, state, offsets)
    assert fn.call_count == 0
    assert call() == "sentinel" and fn.call_count == 1
    args, kwargs = fn.call_args
    assert all(a is b for a, b in zip(args, adapted))
    assert kwargs["initial_state"] is state
    assert kwargs["scale"] == 128 ** -.5
    assert kwargs["normalize_qk"] is False
    assert kwargs["initial_state_indices"] is None
    assert kwargs["cu_seqlens"] is offsets and kwargs["raw_cu_seqlens"] == (0, 3)
    assert kwargs["final_state_dtype"] == torch.float32
    assert "auto_cp" not in kwargs and "force_tc01_cp_candidate" not in kwargs


@pytest.mark.parametrize("plant", ["none", "transpose-state", "change-beta", "wrong-shape", "zero-state"])
def test_saved_payload_hash_and_asymmetric_state_contract(plant):
    packed, state = fixture()
    previous = dict(gate=-.1, packed_input_sha=profile.digest((*packed, state)))
    cell = dict(id="host-test", B=1, T=3, Hk=16, Hv=48, K=128, V=128, gate=-.1)
    if plant == "none":
        assert profile.validate_payload(dict(packed=packed, initial=state), previous, cell) == previous["packed_input_sha"]
        return
    if plant == "transpose-state":
        state = state.transpose(-1, -2).contiguous()  # same shape; different meaning
    elif plant == "change-beta":
        packed[-1].view(torch.int16).flatten()[0] ^= 1
    elif plant == "wrong-shape":
        cell["T"] = 4
    else:
        state.zero_()
        previous["packed_input_sha"] = profile.digest((*packed, state))
    with pytest.raises(ValueError):
        profile.validate_payload(dict(packed=packed, initial=state), previous, cell)


def kernel_rows():
    return [dict(id=str(i), process="4", kernel=name + "(args)", us=float(i + 1))
            for i, name in enumerate(contract.PLA_KERNELS)]


def test_pla_three_kernel_inventory():
    measured = contract.inventory("pla", kernel_rows())
    assert measured["status"] == "ADMITTED" and measured["total_us"] == 6


@pytest.mark.parametrize("plant", ["missing", "duplicate", "unknown", "mixed-process", "other-arch"])
def test_pla_inventory_negatives_are_not_silently_dropped(plant):
    rows = kernel_rows()
    if plant == "missing":
        rows.pop()
    elif plant == "duplicate":
        rows.append(rows[0].copy())
    elif plant == "unknown":
        rows.append(dict(id="7", process="4", kernel="extra_cp_or_normalization_kernel", us=17.))
    elif plant == "mixed-process":
        rows[0]["process"] = "5"
    else:
        rows[1]["kernel"] = rows[1]["kernel"].replace("tc01", "tc02")
    measured = contract.inventory("pla", rows)
    assert measured["status"] == "UNADJUDICATED"
    assert measured["total_us"] == sum(row["us"] for row in rows)
    assert measured["kernels"] == rows


def samples():
    return {role: [dict(status="ADMITTED", total_us=t) for t in values]
            for role, values in dict(default=[10, 11, 12], pla=[5, 6, 7], fla=[20, 21, 22]).items()}


def test_all_three_pairwise_verdicts():
    summary = contract.summarize(samples())
    assert summary["comparisons"]["default_vs_pla"]["verdict"] == "PLA_WINS"
    assert summary["comparisons"]["default_vs_fla"]["verdict"] == "DEFAULT_WINS"
    assert summary["comparisons"]["pla_vs_fla"]["verdict"] == "PLA_WINS"
    overlap = samples()
    overlap["fla"][0]["total_us"] = 5
    assert contract.summarize(overlap)["comparisons"]["pla_vs_fla"]["verdict"] == "UNRESOLVED"
    overlap["pla"][0]["status"] = "UNADJUDICATED"
    assert all(p["verdict"] == "UNADJUDICATED" for p in contract.summarize(overlap)["comparisons"].values())


def test_missing_role_capture_and_cell_denominators_are_red():
    incomplete = samples()
    incomplete["pla"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        contract.summarize(incomplete)
    registered = contract.cells()
    assert len(registered) == 8 and {c["Hv"] for c in registered} == {32, 48}
    assert {c["T"] for c in registered} == {2048, 8192}
    assert {c["gate"] for c in registered} == {-.1, -1.}
    records = {c["id"]: dict(summary=contract.summarize(samples())) for c in registered}
    assert contract.complete(records) == "COLLECTED"
    records.pop(registered[-1]["id"])
    with pytest.raises(ValueError, match="denominator"):
        contract.complete(records)
    assert "NOT_RUN" in collector.table(records)


def runtime(tmp_path):
    root = tmp_path / "runtime"
    package = root / "pla/prefill/flashqla"
    package.mkdir(parents=True)
    (package / "ops.py").write_text("CHUNK_SIZE=32\n")
    (package / "_C.abi3.so").write_bytes(b"host-fixture-not-a-device-binary")
    return root


def test_runtime_is_bound_not_installed_fallback(tmp_path):
    root = runtime(tmp_path)
    manifest = dict(runtime=str(root), files=contract.runtime_files(root), chunk=32,
                    source_sha=contract.PLA_SHA, actlize_sha=contract.PLA_ACTLIZE_SHA)
    contract.validate_pla(manifest)
    image = root / "pla/prefill/flashqla/_C.abi3.so"
    image.write_bytes(b"different-image")
    with pytest.raises(ValueError, match="changed"):
        contract.validate_pla(manifest)
    (image.parent / "_C.other.so").write_bytes(b"ambiguous")
    with pytest.raises(ValueError, match="exactly one"):
        contract.runtime_files(root)


def test_runtime_symlink_is_not_an_immutable_build(tmp_path):
    root = runtime(tmp_path)
    source = root / "pla/prefill/flashqla/ops.py"
    external = tmp_path / "outside.py"
    external.write_text("CHUNK_SIZE=32\n")
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(ValueError, match="links"):
        contract.runtime_files(root)


@pytest.mark.parametrize("plant", ["python", "binary"])
def test_imported_module_cannot_escape_bound_runtime(tmp_path, monkeypatch, plant):
    root = runtime(tmp_path)
    package = root / "pla/prefill/flashqla"
    manifest = dict(runtime=str(root), files=contract.runtime_files(root), chunk=32,
                    source_sha=contract.PLA_SHA, actlize_sha=contract.PLA_ACTLIZE_SHA)
    ops = SimpleNamespace(__file__=str(package / "ops.py"), CHUNK_SIZE=32)
    image = SimpleNamespace(__file__=str(package / "_C.abi3.so"))
    external = tmp_path / "installed-fallback"
    external.write_bytes(b"not-this-build")
    (ops if plant == "python" else image).__file__ = str(external)
    loader = Mock(side_effect=[ops, image])
    monkeypatch.setattr(profile.importlib, "import_module", loader)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    packed, state = fixture()
    with pytest.raises(ValueError, match="unbound"):
        profile.PlaCall(manifest, packed, state)


def test_build_is_exact_upstream_namespace_in_private_cwd(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "bin/hgcc").write_bytes(b"host-compiler-placeholder")
    make_cuda_headers(sdk)
    run = Mock()
    monkeypatch.setattr(collector, "run", run)
    monkeypatch.setattr(collector, "check_pla_source", Mock())
    monkeypatch.setattr(collector, "record_pla", Mock(return_value={}))
    collector.build_pla(tmp_path, tmp_path, {"PPU_SDK": str(sdk)}, 4)
    command, _, env = run.call_args.args
    assert command[:4] == ["bash", "-c", 'cd -- "$1" && shift && exec "$@"', "pla-build"]
    assert command[4] == tmp_path / "pla-source"
    assert command[6:8] == ["setup.py", "build"]
    assert env["PLA_BUILD_NAMESPACE"] == "pla.prefill.flashqla"
    assert env["PPU_ACTLIZE_DIR"] == str(tmp_path / "pla-source/3rdparty/actlize")
    assert env["MAX_JOBS"] == "4" and env["HGCC_THREADS"] == "1"
    assert env["CXX"] == str(ROOT / "tools/ppu_pla_host_cxx.py")
    assert env["ACTLIZE_LA_PLA_HOST_CXX"] == host_cxx.compiler({})
    assert "CPLUS_INCLUDE_PATH" not in env and "CPATH" not in env
    assert run.call_count == 6  # five source operations, exactly one build
    assert collector.record_pla.call_args.args == (tmp_path / "pla-source", tmp_path / "pla-runtime")


def make_cuda_headers(sdk):
    include = sdk / "CUDA_SDK/include"
    include.mkdir(parents=True)
    for name in ("cuda_runtime_api.h", "cuda.h", "cublas_v2.h", "cusparse.h", "cusolverDn.h"):
        (include / name).write_text("#pragma once\n")
    return include


def test_host_sdk_layout_and_ambiguity(tmp_path):
    sdk = tmp_path / "sdk"
    include = make_cuda_headers(sdk)
    (include / "cccl").mkdir()
    target = sdk / "CUDA_SDK/targets/x86_64-linux/include"
    target.parent.mkdir(parents=True)
    target.symlink_to(include)
    assert host_cxx.include_dirs(sdk) == [include, include / "cccl"]
    target.unlink()
    target.mkdir()
    (target / "cuda_runtime_api.h").write_text("#pragma once\n")
    with pytest.raises(ValueError, match="one CUDA API"):
        host_cxx.include_dirs(sdk)


def test_missing_host_header_is_fail_not_system_cuda_fallback(tmp_path):
    sdk = tmp_path / "sdk"
    include = make_cuda_headers(sdk)
    (include / "cuda_runtime_api.h").unlink()
    with pytest.raises(ValueError, match="one CUDA API"):
        host_cxx.include_dirs(sdk)


def test_real_host_compiler_negative_then_adapter_positive(tmp_path):
    sdk = tmp_path / "sdk"
    include = make_cuda_headers(sdk)
    (include / "actlize_test_sdk_marker.h").write_text("#define ACTLIZE_TEST_SDK_MARKER 42\n")
    source = tmp_path / "host.cc"
    source.write_text('#include <actlize_test_sdk_marker.h>\n#include <cuda_runtime_api.h>\n'
                      'static_assert(ACTLIZE_TEST_SDK_MARKER == 42);\n')
    cxx = host_cxx.compiler({})
    flags = ["-std=c++20", "-fsyntax-only", str(source)]
    before = subprocess.run([cxx, *flags], capture_output=True, text=True)
    assert before.returncode != 0 and "actlize_test_sdk_marker.h" in before.stderr
    command = host_cxx.command({"PPU_SDK": str(sdk), "ACTLIZE_LA_PLA_HOST_CXX": cxx}, flags)
    after = subprocess.run(command, capture_output=True, text=True)
    assert after.returncode == 0, after.stderr


def test_ninja_bookkeeping_does_not_hide_new_compilable_sources(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "3rdparty/actlize").mkdir(parents=True)
    untracked = [".ninja_log", ".ninja_deps"]
    def git(path, *args):
        if args == ("rev-parse", "HEAD"):
            return contract.PLA_SHA if path == source else contract.PLA_ACTLIZE_SHA
        if args[0] == "ls-files":
            return "\n".join(untracked) if path == source else ""
        return ""
    monkeypatch.setattr(contract, "git", git)
    contract.check_pla_source(source)
    untracked.append("src/prefill/flashqla/kernels/injected.cu")
    with pytest.raises(ValueError, match="unmodified"):
        contract.check_pla_source(source)


def failed_build(tmp_path):
    previous = tmp_path / "failed-run"
    evidence = previous / "evidence"
    evidence.mkdir(parents=True)
    (previous / "pla-source").mkdir()
    (previous / "pla-build").mkdir()
    (previous / "pla-build/device.o").write_bytes(b"host-fixture-device-object")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    selected = {"origin": {"sha256": "unchanged-admitted-binary"}}
    (evidence / "summary.json").write_text(json.dumps(dict(
        status="INCOMPLETE", captures={}, input_run=selected, sdk=str(sdk))))
    (evidence / "pla-build.log").write_text("fatal error: cuda_runtime_api.h: No such file or directory\n")
    (evidence / "compiler.log").write_text("same measured HGCC\n")
    compiler_log = tmp_path / "current-compiler.log"
    compiler_log.write_bytes((evidence / "compiler.log").read_bytes())
    return previous, selected, sdk, compiler_log


@pytest.mark.parametrize("plant", ["none", "input", "sdk", "compiler", "capture", "completed", "other-error"])
def test_reuse_is_limited_to_the_exact_failed_build(tmp_path, monkeypatch, plant):
    previous, selected, sdk, compiler_log = failed_build(tmp_path)
    monkeypatch.setattr(collector, "check_pla_source", Mock())
    summary_path = previous / "evidence/summary.json"
    summary = json.loads(summary_path.read_text())
    if plant == "input":
        selected = {"origin": {"sha256": "another-binary"}}
    elif plant == "sdk":
        summary["sdk"] = str(tmp_path / "another-sdk")
    elif plant == "compiler":
        compiler_log.write_text("new compiler\n")
    elif plant == "capture":
        summary["captures"] = {"already-measured": {}}
    elif plant == "completed":
        summary["status"] = "COLLECTED"
    elif plant == "other-error":
        (previous / "evidence/pla-build.log").write_text("unknown error\n")
    summary_path.write_text(json.dumps(summary))
    if plant == "none":
        reuse = collector.reuse_failed_build(previous, selected, sdk, compiler_log, workspace=tmp_path)
        assert reuse["source"] == str(previous / "pla-source")
        assert len(reuse["objects_before"]) == 1
    else:
        with pytest.raises(ValueError):
            collector.reuse_failed_build(previous, selected, sdk, compiler_log, workspace=tmp_path)


def test_second_builder_cannot_write_active_object_tree(tmp_path, monkeypatch):
    previous, selected, sdk, compiler_log = failed_build(tmp_path)
    make_cuda_headers(sdk)
    monkeypatch.setattr(collector, "check_pla_source", Mock())
    reuse = collector.reuse_failed_build(previous, selected, sdk, compiler_log, workspace=tmp_path)
    run = Mock()
    monkeypatch.setattr(collector, "run", run)
    with (previous / "pla-build/.threeway-build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            collector.build_pla(tmp_path, tmp_path, {"PPU_SDK": str(sdk)}, 4, reuse)
    run.assert_not_called()


def receipt():
    base = dict(device={"uuid": "measured"}, shape={"B": 1}, gate=-.1,
                library_sha="image", operands_sha="operands", input_admission_sha="prior",
                canonical_input_sha="canonical", pla={"source_sha": contract.PLA_SHA}, chunks=contract.CHUNKS)
    arms = {r: dict(input_sha=f"input-{r}", output_sha=f"output-{r}", route={"role": r}) for r in contract.ROLES}
    admission = dict(base, status="PASS", numeric_repeats=8, arms=arms)
    subject = dict(base, **arms["pla"], status="PASS", phase="subject", role="pla", public_calls=1,
                   warmup=0, verification="CPU")
    return admission, subject


@pytest.mark.parametrize("plant", ["public_calls", "warmup", "device", "operands_sha", "input_sha", "output_sha", "route", "chunks"])
def test_changed_subject_cannot_keep_a_green_receipt(plant):
    admission, subject = receipt()
    contract.validate_receipt(admission, subject, "pla")
    subject = copy.deepcopy(subject)
    subject[plant] = "wrong"
    with pytest.raises(ValueError):
        contract.validate_receipt(admission, subject, "pla")


@pytest.mark.parametrize("name", ["FF_GEMM_REORDER", "PLA_SANITY_CHECK", "PLA_FLASHQLA_CP_TC01_TFU_MODEL", "GDN_QSA_WY_EXTENSION"])
def test_inherited_experiment_overrides_are_red(name):
    with pytest.raises(ValueError):
        collector.reference_environment({name: "1"})


def test_runner_is_box_only_isolated_and_uses_site_profiler():
    shell = (ROOT / "tools/run_ppu_gdn_threeway_box.sh").read_text()
    code = (ROOT / "tools/collect_gdn_threeway_acu.py").read_text()
    assert "/sim/eec/shared/junfu.qx/asight/bin/acu" in shell
    assert "mktemp" not in shell and "/tmp/" not in shell + code
    assert '"setup.py", "build"' in code and '"pip"' not in code
    assert "read_inputs(args.input_run)" in code
    assert "check_profiler(" in code and "--set\", \"full" in code
    assert "PPUProfiler" not in code
    assert "results.tar.gz" in code
