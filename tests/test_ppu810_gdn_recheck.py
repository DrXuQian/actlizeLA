"""Host-only negatives for the ZW810 recheck; no device PASS inferred."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ppu810_recheck", ROOT / "tools/check_ppu810_gdn_recheck.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize("name", ["PPU-ZW810", "PPU-ZW810E"])
def test_810_identity(name):
    check.validate_810({"name": name})


@pytest.mark.parametrize("name", ["PPU-ZW610", "PPU-ZW890", "PPU-ZW8100", "NVIDIA RTX 5090"])
def test_other_device_is_not_810(name):
    with pytest.raises(ValueError):
        check.validate_810({"name": name})


@pytest.mark.parametrize("length,hv,wanted", [(2048, 32, 1), (2048, 48, 0), (8192, 32, 0), (8192, 48, 0)])
def test_only_admitted_shape_changes_route(length, hv, wanted):
    assert check.expected_route(length, hv) == wanted
    check.check_route({"id": wanted, "name": "solve8" if wanted else "retained"}, length, hv)
    # Wrong binary or wrongly widened selector must remain a hard error.
    with pytest.raises(ValueError):
        check.check_route({"id": 1 - wanted, "name": "retained" if wanted else "solve8"}, length, hv)


def test_invalid_route_name_and_unknown_shape():
    with pytest.raises(ValueError):
        check.check_route({"id": 1, "name": "retained"}, 2048, 32)
    with pytest.raises(ValueError):
        check.expected_route(2048, 64)


def test_site_acu_and_complete_eight_cell_scope():
    runner = (ROOT / "tools/run_ppu810_gdn_recheck_box.sh").read_text()
    builder = (ROOT / "tools/run_ppu10_gdn_tuning_box.sh").read_text()
    for text in (runner, builder):
        assert "/sim/eec/shared/junfu.qx/asight/bin/acu" in text
        assert '$SDK/asight/bin/acu' not in text
    assert '--batches 1 --lengths 2048 8192 --value-heads 32 48 --gates -0.1 -1.0' in runner
    assert '--manifest ' not in runner and '--no-confirm' not in runner
    assert '--repeats 3' in runner and 'acu/results.tar.gz' in runner
    assert 'TMPDIR="$OUT/scratch"' in runner and '$OUT/triton-cache' in runner
