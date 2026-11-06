"""Token ownership proof and negatives; numerical/device proof is separate."""
from collections import Counter
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def owners(extent, blocks, threads, *, omit_z=False, short_stride=False):
    return Counter(i for z in range(blocks) for lane in range(threads)
                   for i in range((0 if omit_z else z) * threads + lane, extent,
                                  threads if short_stride else blocks * threads))


@pytest.mark.parametrize("width", [1, 2, 4, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 8192])
def test_every_token_and_output_state_element_has_exactly_one_owner(width):
    blocks = (width + 31) // 32
    # Same mathematical block-stride partition used in the three device loops;
    # denominator is independently the public logical tensor's full extent.
    for extent, lanes in ((width, 4), (width * 128, 128),
                          (width * 128, 256), (16384, 256)):
        actual = owners(extent, blocks, lanes)
        assert len(actual) == extent and set(actual.values()) == {1}
    # Changing device metadata never changes launch capacity. Every ragged
    # output is covered without touching the remainder; pack still pads width.
    for length in (0, 1, width // 2, width - 1):
        actual = owners(length * 128, blocks, 256)
        assert actual == Counter(range(length * 128))


def test_missing_offset_and_short_stride_are_red():
    expected = Counter(range(65 * 128))
    for plant in (dict(omit_z=True), dict(short_stride=True)):
        assert owners(65 * 128, 3, 128, **plant) != expected
    missing_block = owners(65 * 128, 3, 128)
    del missing_block[8192]
    assert missing_block != expected


def test_real_launch_policy_compiles_on_host_and_bounds_grid(tmp_path):
    source = ROOT / "tests/prefill_schedule_ownership.cpp"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(ROOT / "csrc/serving"), str(source),
                    "-o", str(tmp_path / "policy")], check=True)
    subprocess.run([str(tmp_path / "policy")], check=True)


@pytest.mark.parametrize("old,new", [("return partition * lanes + lane;", "return lane;"),
                                     ("return partitions * lanes;", "return lanes;")])
def test_actual_production_index_mutations_are_red(tmp_path, old, new):
    header = (ROOT / "csrc/serving/prefill_schedule.h").read_text()
    assert header.count(old) == 1
    (tmp_path / "prefill_schedule.h").write_text(header.replace(old, new))
    subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(tmp_path),
                    str(ROOT / "tests/prefill_schedule_ownership.cpp"),
                    "-o", str(tmp_path / "negative")], check=True)
    assert subprocess.run([str(tmp_path / "negative")]).returncode != 0


def test_scope_and_arithmetic_boundaries_are_explicit():
    source = (ROOT / "csrc/serving/gdn.cu").read_text()
    assert "if (p.mode == ChunkPrefill)" in source
    assert "prepare_gdn<false>" in source
    assert "float factor = rsqrtf(warp_sum(square) + 1e-6f)" in source
    assert "if (h < p.hk && p.mode != ChunkPrefill)" in source
    cmake = (ROOT / "csrc/serving/CMakeLists.txt").read_text()
    assert 'device admission required)" OFF)' in cmake
    assert 'NOT ALA_TARGET STREQUAL "ppu10"' in cmake


def test_wrong_target_cmake_rejected_before_sdk_configuration(tmp_path):
    result = subprocess.run(["cmake", "-S", str(ROOT / "csrc/serving"),
                             "-B", str(tmp_path / "wrong-target"),
                             "-DALA_TARGET=ppu15", "-DALA_PREFILL_TOKEN_PARALLEL=ON"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode != 0
    assert "Token-parallel prefill experiment is scoped to ppu10" in result.stdout


def test_device_comparator_one_bit_and_missing_field_are_red():
    import torch
    from prefill_parallel_device import compare
    q = torch.tensor([1., 2., 3.], dtype=torch.bfloat16)
    expected = dict(q=q, state=torch.eye(3))
    compare(expected, expected)
    planted = dict(expected, q=q.clone())
    planted["q"].view(torch.int16)[0] ^= 1
    with pytest.raises(AssertionError, match="bit mismatch: q"):
        compare(planted, expected)
    with pytest.raises(AssertionError, match="denominator"):
        compare(dict(q=q), expected)
