"""Offline phase accounting tests; no synthetic GPU timings are evidence."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kda_budget", ROOT / "dev/ppu/zw810_fla_kda/analyze_baseline.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def rows(names):
    return [dict(kernel=name, us=float(i + 1)) for i, name in enumerate(names)]


def test_wu_is_part_of_state_equivalent_and_fills_are_not_dropped():
    data = rows([*budget.retained.FLA, "vectorized_elementwise_kernel<FillFunctor<float>>"])
    parts = budget.phase_times("fla", data)
    assert parts["state_equivalent"] == 3 + 4
    assert parts["other_included"] == 6
    assert sum(parts.values()) == 21
    assert budget.phase_times("default", rows(budget.retained.CORE))["state_equivalent"] == 3


def test_unknown_work_is_not_silently_dropped():
    with pytest.raises(ValueError, match="unregistered"):
        budget.phase_times("fla", rows(["unexpected_extra_kernel"]))


@pytest.mark.parametrize("plant", ["traversal", "link", "duplicate"])
def test_archive_is_data_only_and_fail_closed(plant):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        name = "../escape" if plant == "traversal" else "data"
        item = tarfile.TarInfo(name)
        if plant == "link":
            item.type, item.linkname = tarfile.SYMTYPE, "/root/private"
        archive.addfile(item)
        if plant == "duplicate":
            archive.addfile(item)
    with pytest.raises(ValueError):
        budget.files(buffer.getvalue())


def test_missing_cell_stays_red(monkeypatch):
    summary = dict(status="COLLECTED", captures={c["id"]: {} for c in budget.threeway.cells()[:-1]})
    monkeypatch.setattr(budget, "files", lambda _: {"summary.json": json.dumps(summary).encode()})
    with pytest.raises(ValueError, match="eight-cell denominator"):
        budget.analyze(b"synthetic-missing-cell-not-device-evidence")


@pytest.mark.parametrize("plant", ["missing-role", "missing-repeat", "wrong-shape"])
def test_admitted_label_cannot_hide_incomplete_registered_inputs(monkeypatch, plant):
    cells = budget.threeway.cells()
    first = cells[0]
    captures = {role: [{}] * 3 for role in budget.retained.ROLES}
    admission = dict(shape=budget.threeway.geometry(first), gate=first["gate"])
    if plant == "missing-role":
        del captures["fla"]
    elif plant == "missing-repeat":
        captures["fla"] = captures["fla"][:-1]
    else:
        admission["shape"] = {}
    summary = dict(status="COLLECTED", captures={c["id"]: {} for c in cells})
    summary["captures"][first["id"]] = dict(captures=captures)
    data = {"summary.json": json.dumps(summary).encode(),
            first["id"] + "/admission.json": json.dumps(admission).encode()}
    monkeypatch.setattr(budget, "files", lambda _: data)
    with pytest.raises(ValueError, match="denominator changed"):
        budget.analyze(b"synthetic-denominator-negative-not-device-evidence")
