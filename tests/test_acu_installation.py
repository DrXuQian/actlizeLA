"""Run an executable parser stand-in, not just an argv mock; no PPU claims."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
acu = importlib.import_module("acu_installation")

PARSER = r'''
import os
from pathlib import Path
import sys
home = Path(__file__).resolve().parent.parent
args = sys.argv[1:]
bound = (os.environ.get("ASIGHT_HOME") == str(home)
         and "--section-folder" in args
         and args[args.index("--section-folder") + 1] == str(home / "sections"))
schema = (home / "sections/fixture.section").read_text().strip()
if not bound or schema.startswith("incompatible"):
    print('[libprotobuf ERROR] Error parsing text-format HG.Profiler.ProfilerSection: '
          'Message type "HG.Profiler.ProfilerSectionBodyItem" has no field named "MemoryMetricTable".')
    sys.exit(0 if schema == "incompatible-rc0" else 101)
print("synthetic ACU: unchanged full sections parsed; no device")
'''


def fake_acu(tmp_path, schema="compatible"):
    home = tmp_path / "site-asight"
    (home / "bin").mkdir(parents=True)
    (home / "sections").mkdir()
    section = home / "sections/fixture.section"
    section.write_text(schema)
    tool = home / "bin/acu"
    tool.write_text(f"#!{sys.executable}\n" + PARSER)
    tool.chmod(0o755)
    return tool, section


def test_foreign_installation_is_red_even_at_version_then_bound_pair_passes(tmp_path, monkeypatch):
    tool, section = fake_acu(tmp_path)
    monkeypatch.setenv("ASIGHT_HOME", "/foreign-sdk/asight")
    # Old builder path, same executable and sections: this must really be red.
    old = subprocess.run([str(tool), "--version"], capture_output=True, text=True)
    assert old.returncode == 101 and "MemoryMetricTable" in old.stdout
    before = section.read_bytes()
    out = tmp_path / "check"
    monkeypatch.setattr(sys, "argv", ["acu-check", "--acu", str(tool), "--out", str(out)])
    assert acu.main() == 0
    receipt = json.loads((out / "profiler.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["inherited_asight_home"] == "/foreign-sdk/asight"
    assert section.read_bytes() == before, "fix may not delete incompatible fields"
    for name in ("acu-version.log.command", "acu-section-check.log.command"):
        assert "--section-folder" in (out / name).read_text()


@pytest.mark.parametrize("schema", ["incompatible", "incompatible-rc0"])
def test_incompatible_companions_remain_failure_including_false_success_rc(tmp_path, monkeypatch, schema):
    tool, section = fake_acu(tmp_path, schema)
    out = tmp_path / "check"
    monkeypatch.setattr(sys, "argv", ["acu-check", "--acu", str(tool), "--out", str(out)])
    assert acu.main() == 1
    assert json.loads((out / "profiler.json").read_text())["status"] == "INCOMPLETE"
    assert "MemoryMetricTable" in (out / "acu-version.log").read_text()
    assert not (out / "acu-section-check.log").exists()
    assert section.read_text() == schema


def test_actual_builder_fails_before_build_or_device_with_incompatible_profiler(tmp_path):
    tool, _ = fake_acu(tmp_path, "incompatible")
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    hgcc = sdk / "bin/hgcc"
    hgcc.write_text("#!/bin/sh\necho synthetic-hgcc-version\n")
    hgcc.chmod(0o755)
    out = tmp_path / "build-run"
    result = subprocess.run(
        ["bash", str(ROOT / "tools/run_ppu10_gdn_tuning_box.sh")],
        env=os.environ | dict(PPU_SDK=str(sdk), ACU=str(tool), OUT=str(out),
                              ASIGHT_HOME="/foreign-sdk/asight", PYTHON=sys.executable,
                              VALIDATE_ONLY="1"), capture_output=True, text=True)
    assert result.returncode != 0
    assert "MemoryMetricTable" in result.stdout
    receipt = json.loads((out / "profiler/profiler.json").read_text())
    assert receipt["profiler"]["home"] == str(tool.parent.parent)
    assert receipt["status"] == "INCOMPLETE"
    for path in ("build", "build.log", "results.json", "device-tests.log"):
        assert not (out / path).exists(), path


def test_builder_has_no_unbound_version_probe():
    builder = (ROOT / "tools/run_ppu10_gdn_tuning_box.sh").read_text()
    assert '"$ACU" --version' not in builder
    assert builder.index("tools/acu_installation.py") < builder.index("cmake -S")


def test_bare_acu_resolves_path_and_real_symlink_home_not_cwd_or_foreign_sdk(tmp_path, monkeypatch):
    tool, _ = fake_acu(tmp_path)
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    (path_bin / "acu").symlink_to(tool)
    (tmp_path / "acu").write_text("must not execute cwd/acu")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(path_bin))
    monkeypatch.setenv("ASIGHT_HOME", "/foreign-sdk/asight")
    layout, env, command = acu.bind_profiler("acu", os.environ)
    assert layout["executable"] == str(tool)
    assert env["ASIGHT_HOME"] == str(tool.parent.parent)
    assert command[0] == str(tool) and command[2] == str(tool.parent.parent / "sections")


def test_bare_acu_absence_is_failure_not_cwd_fallback(tmp_path, monkeypatch):
    (tmp_path / "acu").write_text("not on PATH")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        acu.profiler_layout("acu")
