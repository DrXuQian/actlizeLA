"""Round-one algebra, lifetime and evidence negatives; no device speed claim."""
import copy
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "benchmarks"):
    sys.path.insert(0, str(path))
contract = importlib.import_module("gdn_kda_contract")
profile = importlib.import_module("profile_gdn_parallel_wu")


def sequential(q, k, v, g, beta, h):
    h = h.clone()
    outputs = []
    for i in range(len(q)):
        h = g[i].exp() * h
        delta = beta[i] * (v[i] - k[i] @ h)
        h = h + torch.outer(k[i], delta)
        outputs.append(q[i] @ h)
    return torch.stack(outputs), h


def materialized(q, k, v, g, beta, h, plant=None):
    # Independent float64 math, not a simulation of the CUDA fragment layouts.
    prefix = g.cumsum(0)
    weights = (prefix[:, None] - prefix[None, :]).exp()
    lower = torch.tril((k @ k.T) * weights * beta[:, None], diagonal=-1)
    eye = torch.eye(len(q), dtype=torch.float64)
    p = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
    w = p @ (beta[:, None] * prefix.exp()[:, None] * k)
    u = p @ (beta[:, None] * v)
    if plant == "transposed-state":
        h = h.T
    residual = u - w @ h if plant != "wrong-sign" else u + w @ h
    output = prefix.exp()[:, None] * (q @ h) + torch.tril((q @ k.T) * weights) @ residual
    final = prefix[-1].exp() * h + k.T @ ((prefix[-1] - prefix).exp()[:, None] * residual)
    return output, final


@pytest.mark.parametrize("gate", [0., -.1, -1.])
@pytest.mark.parametrize("length", [64, 128])
def test_wu_real_algebra_matches_independent_recurrence(gate, length):
    generator = torch.Generator().manual_seed(47)
    q = torch.randn(length, 16, generator=generator, dtype=torch.float64) / 4
    k = torch.randn(length, 16, generator=generator, dtype=torch.float64)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(length, 16, generator=generator, dtype=torch.float64)
    g = torch.full((length,), gate, dtype=torch.float64)
    beta = torch.rand(length, generator=generator, dtype=torch.float64)
    h = torch.randn(16, 16, generator=generator, dtype=torch.float64) * .03
    want = sequential(q, k, v, g, beta, h)
    current, outputs = h, []
    for start in range(0, length, 64):
        output, current = materialized(q[start:start+64], k[start:start+64], v[start:start+64],
                                       g[start:start+64], beta[start:start+64], current)
        outputs.append(output)
    for got, expected in zip((torch.cat(outputs), current), want):
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-11)
    for plant in ("transposed-state", "wrong-sign"):
        bad = materialized(q[:64], k[:64], v[:64], g[:64], beta[:64], h, plant)
        reference = sequential(q[:64], k[:64], v[:64], g[:64], beta[:64], h)
        assert max((a-b).abs().max().item() for a,b in zip(bad, reference)) > 1e-6


def test_workspace_padded_pitch_and_per_group_retirement(tmp_path):
    exe = tmp_path / "workspace"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "include"),
                    ROOT / "tests/ppu_parallel_wu_workspace.cpp", "-o", exe], check=True)
    subprocess.run([exe], check=True)


@pytest.mark.parametrize("gate", [-.1, -1.])
def test_bf16_boundary_model_against_token_reference(gate):
    # A CPU precision screen, NOT native MMA/TF32 instruction emulation. The
    # device gate still uses the unchanged independent 2% output AND state rule.
    from test_ppu_gdn_backend import fixture, assert_pair
    from test_ppu_wy_backend import oracle
    inputs = fixture(1, 128, 1, 2, gate)
    initial = torch.randn(1, 2, 128, 128, generator=torch.Generator().manual_seed(91)) * .02
    want = oracle(inputs, initial)
    q, k, v, g, beta = inputs
    outputs, finals = [], []
    bf = lambda x: x.bfloat16().float()
    for head in range(2):
        state = initial[0, head].clone()
        blocks = []
        for start in (0, 64):
            sl = slice(start, start + 64)
            qq, kk, vv = q[0, sl, 0].float(), k[0, sl, 0].float(), v[0, sl, head].float()
            bb, prefix = beta[0, sl, head].float(), g[0, sl, head].float().cumsum(0)
            weights = (prefix[:, None] - prefix[None, :]).exp()
            lower = torch.tril((kk @ kk.T) * bb[:, None] * weights, diagonal=-1)
            eye = torch.eye(64)
            p = bf(torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True))
            w = bf(p @ bf((kk * bb[:, None]) * prefix.exp()[:, None]))
            u = bf(p @ bf(vv * bb[:, None]))
            snapshot = bf(state)
            delta = u - w @ snapshot
            attn = bf(torch.tril((qq @ kk.T) * weights))
            blocks.append(((prefix.exp()[:, None] * (qq @ snapshot) + attn @ bf(delta)) * (128 ** -.5)).bfloat16())
            state = prefix[-1].exp() * state + kk.T @ bf(delta * (prefix[-1] - prefix).exp()[:, None])
        outputs.append(torch.cat(blocks))
        finals.append(state)
    got = torch.stack(outputs, dim=1).unsqueeze(0), torch.stack(finals).unsqueeze(0)
    assert_pair(got, want)


def rows():
    return [dict(kernel=k, process="1", us=float(i+1)) for i,k in enumerate(contract.CANDIDATE_KERNELS)]


def test_materialization_is_not_removed_from_timing():
    assert contract.inventory("parallel-wu", rows())["total_us"] == 15
    data = rows()
    data.pop(2)
    assert contract.inventory("parallel-wu", data)["status"] == "UNADJUDICATED"
    data = rows() + [dict(kernel="extra_helper", process="1", us=20.)]
    measured = contract.inventory("parallel-wu", data)
    assert measured["status"] == "UNADJUDICATED" and measured["total_us"] == 35


def test_winning_state_cannot_hide_losing_full_call():
    samples = {role: [dict(status="ADMITTED", total_us=t) for t in values] for role,values in
               {"default": [10., 11., 12.], "parallel-wu": [15., 16., 17.], "fla": [8., 8.5, 9.]}.items()}
    result = contract.summarize(samples)
    assert result["comparisons"]["default_vs_parallel-wu"]["verdict"] == "default_WINS"
    assert result["comparisons"]["parallel-wu_vs_fla"]["verdict"] == "fla_WINS"
    samples["parallel-wu"].pop()
    with pytest.raises(ValueError):
        contract.summarize(samples)
    with pytest.raises(ValueError):
        contract.complete({c["id"]: {} for c in contract.cells()[:-1]})


def test_subject_is_bound_to_candidate_and_numeric_gate():
    admission = dict(status="PASS", numeric_repeats=8, unchanged_public_default_bits="PASS",
                     device={"uuid":"fixture"}, shape={}, gate=-.1, libraries={"candidate":"bound"},
                     operands_sha="input", input_admission_sha="prior", canonical_input_sha="canonical",
                     arms={role: dict(input_sha="i", output_sha="o", route={"role":role}) for role in contract.ROLES})
    subject = dict(admission, phase="subject", role="parallel-wu", public_calls=1, warmup=0, verification="CPU",
                   **admission["arms"]["parallel-wu"])
    contract.validate_receipt(admission, subject, "parallel-wu")
    for field, wrong in (("libraries", {"candidate":"stale"}), ("output_sha","bad"), ("public_calls",2), ("route",{})):
        changed = copy.deepcopy(subject)
        changed[field] = wrong
        with pytest.raises(ValueError):
            contract.validate_receipt(admission, changed, "parallel-wu")
    admission["unchanged_public_default_bits"] = "FAIL"
    with pytest.raises(ValueError):
        contract.validate_receipt(admission, subject, "parallel-wu")


def test_new_entry_is_opt_in_and_original_sources_are_not_modified():
    cmake = (ROOT / "csrc/serving/CMakeLists.txt").read_text()
    assert 'option(ALA_PPU10_PARALLEL_WU_EXPERIMENT "Build an opt-in KDA W/U counterfactual; never replace the public default" OFF)' in cmake
    for name in ("gdn_wy_residual_full_chunk_ppu.cu", "gdn_wy_split_prepare_ppu.cu", "gdn_wy_solve_static_ppu.cu",
                 "gdn_wy_residual_warps8_hvlayout_ppu.cu"):
        path = "csrc/gdn_chunk/" + name
        assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", "672400d:" + path], cwd=ROOT)


@pytest.mark.parametrize("target", ["ppu15", "cuda_sm120"])
def test_experiment_cannot_enter_another_architecture_graph(tmp_path, target):
    result = subprocess.run(["cmake", "-S", ROOT / "csrc/serving", "-B", tmp_path / "build",
                             "-DALA_TARGET=" + target, "-DALA_PPU10_PARALLEL_WU_EXPERIMENT=ON"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode != 0 and "Parallel W/U experiment requires explicit ppu10" in result.stdout


def test_no_device_is_not_a_numerical_pass(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["profile", "--library", "missing.so", "--candidate", "candidate.so",
                                      "--inputs", str(tmp_path), "--directory", str(tmp_path), "--receipt", str(receipt),
                                      "--cell", contract.cells()[0]["id"], "--phase", "admit"])
    assert profile.main() == 3
    assert '"status": "SKIP"' in receipt.read_text()
