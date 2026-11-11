"""Host policy/ownership gates; no host fixture counts as device admission."""
from collections import Counter
import ctypes as C
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from actlize_la.serving.gdn import _prefill_transport_route

ROOT = Path(__file__).resolve().parents[1]


def owners(extent, blocks, lanes, *, omit_z=False, short_stride=False):
    return Counter(i for z in range(blocks) for lane in range(lanes)
                   for i in range((0 if omit_z else z) * lanes + lane, extent,
                                  lanes if short_stride else blocks * lanes))


@pytest.mark.parametrize("width", [1, 2, 4, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 8192])
def test_token_packed_output_and_state_ownership(width):
    blocks = (width + 31) // 32
    for extent, lanes in ((width, 4), (width * 128, 128),
                          (width * 128, 256), (16384, 256)):
        actual = owners(extent, blocks, lanes)
        assert len(actual) == extent and set(actual.values()) == {1}
    for length in (0, 1, width // 2, width - 1):
        assert owners(length * 128, blocks, 256) == Counter(range(length * 128))


def compile_policy(tmp_path):
    binary = tmp_path / "policy"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(tmp_path),
                    "-I", str(ROOT / "csrc/serving"),
                    str(ROOT / "tests/prefill_schedule_ownership.cpp"),
                    "-o", str(binary)], check=True)
    return subprocess.run([str(binary)]).returncode


def test_real_policy_compiles_and_bounds_grid(tmp_path):
    assert compile_policy(tmp_path) == 0


@pytest.mark.parametrize("old,new", [
    ("return partition * lanes + lane;", "return lane;"),
    ("return partitions * lanes;", "return lanes;"),
    ("requests == 1", "requests <= 2"),
    ("hv == 48 ||", ""),
])
def test_real_production_policy_mutations_are_red(tmp_path, old, new):
    header = (ROOT / "csrc/serving/prefill_schedule.h").read_text()
    assert header.count(old) == 1
    (tmp_path / "prefill_schedule.h").write_text(header.replace(old, new))
    assert compile_policy(tmp_path) != 0


@pytest.mark.parametrize("target", ["ppu10", "ppu15", "cuda_sm120"])
def test_old_libraries_remain_staged(target):
    library = SimpleNamespace(target=target, lib=SimpleNamespace())
    assert _prefill_transport_route(library, 1, 2048, 16, 32) == {
        "entry": "ala_prefill_run", "direct_pack": False}


@pytest.mark.parametrize("selected", [0, 1])
def test_real_native_query_controls_report_without_launch(selected):
    query = Mock(return_value=selected)
    library = SimpleNamespace(target="ppu10", lib=SimpleNamespace(ala_prefill_direct_pack_route=query))
    result = _prefill_transport_route(library, 1, 8192, 16, 48)
    query.assert_called_once_with(1, 8192, 16, 48)
    assert result == {"entry": "ala_prefill_run", "direct_pack": bool(selected)}


@pytest.mark.parametrize("target,selected", [("ppu10", -1), ("ppu10", 2), ("ppu15", 1), ("cuda_sm120", 1)])
def test_unknown_or_wrong_target_route_is_red(target, selected):
    library = SimpleNamespace(target=target, lib=SimpleNamespace(ala_prefill_direct_pack_route=Mock(return_value=selected)))
    with pytest.raises(RuntimeError, match="unknown native"):
        _prefill_transport_route(library, 1, 2048, 16, 32)


def test_device_comparison_requires_full_denominator_and_detects_one_bit():
    from prefill_transport_device import compare, PACKED_FIELDS
    expected = {name: torch.ones(2, dtype=torch.bfloat16) for name in PACKED_FIELDS}
    expected.update(output=torch.ones(2), state=torch.eye(3))
    compare(expected, expected)
    planted = {**expected, "q": expected["q"].clone()}
    planted["q"].view(torch.int16)[0] ^= 1
    with pytest.raises(AssertionError, match="q"):
        compare(planted, expected)
    with pytest.raises(AssertionError, match="denominator"):
        compare({n: v for n, v in expected.items() if n != "errors"}, expected)


def test_device_gate_has_exact_registered_and_fallback_inventory():
    from prefill_transport_device import cases
    rows = cases()
    assert len(rows) == len(set(rows)) == 9
    assert sum(selected for _, _, _, selected in rows) == 6
    assert {(r, t, hv) for r, t, hv, selected in rows if selected} == {
        (1, t, hv) for t in (2048, 8192) for hv in (32, 48, 64)}
    assert {(r, t, hv) for r, t, hv, selected in rows if not selected} == {
        (2, 2048, 32), (1, 512, 48), (1, 2049, 64)}


def test_loaded_native_transport_queries_do_not_initialize_a_device():
    from prefill_transport_device import cases

    path = os.environ.get("ALA_SERVING_TEST_LIBRARY")
    if not path:
        pytest.skip("no compiled native library supplied; not device evidence")
    lib = C.CDLL(path)
    lib.ala_prefill_token_parallel.argtypes = []
    lib.ala_prefill_direct_pack_route.argtypes = [C.c_int] * 4
    ppu10 = lib.ala_serving_target() == 100
    assert lib.ala_prefill_token_parallel() == int(ppu10)
    for requests, width, hv, selected in cases():
        assert lib.ala_prefill_direct_pack_route(requests, width, 16, hv) == int(ppu10 and selected)
