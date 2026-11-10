"""Exercise the actual runner shell up to discovery; never load Torch or a GPU."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LINE = 'export LD_LIBRARY_PATH="$SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
OLD_RUNTIME_LINE = 'export LD_LIBRARY_PATH="$SDK/lib:$SDK/CUDA_SDK/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'


def run_to_discovery(work, source, inherited):
    tools = work / "source/tools"
    tools.mkdir(parents=True)
    runner = tools / "run_ppu15_gdn_box.sh"
    runner.write_text(source)
    binaries = work / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "assert sys.argv[3:] in (['rev-parse', '--short', 'HEAD'], ['rev-parse', 'HEAD'])\n"
        "print('0123456789abcdef')\n"
    )
    git.chmod(0o755)
    python = binaries / "python-recorder"
    python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "assert sys.argv[1:] == ['-m', 'actlize_la.serving.device', '--target', 'ppu15']\n"
        "print(json.dumps({'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']}))\n"
        "sys.exit(3)  # Host fixture: discovery stops before any build/device run.\n"
    )
    python.chmod(0o755)
    sdk, output = work / "sdk", work / "output"
    env = dict(os.environ, PPU_SDK=str(sdk), PYTHON=str(python), OUT=str(output),
               PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               LD_LIBRARY_PATH=inherited, NO_PERF="1")
    result = subprocess.run(["bash", str(runner)], env=env, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "NOT_ADMITTED rc=3" in result.stderr
    assert not (output / "build").exists()
    with tarfile.open(output / "results.tar.gz") as archive:
        assert {"sha.txt", "runtime.txt", "identity.log"} <= set(archive.getnames())
    observed = json.loads((output / "identity.log").read_text())['LD_LIBRARY_PATH']
    expected = str(sdk / "lib") + (":" + inherited if inherited else "")
    return observed, expected


@pytest.mark.parametrize("inherited", ["", "/container/torch/lib:/container/ppu-cuda/lib64"])
def test_runner_keeps_container_facade_and_old_injection_is_detected(tmp_path, inherited):
    source = (ROOT / "tools/run_ppu15_gdn_box.sh").read_text()
    assert source.count(RUNTIME_LINE) == 1
    observed, expected = run_to_discovery(tmp_path / "current", source, inherited)
    assert observed == expected
    # Same shell harness and environment; only restore the defective export.
    old = source.replace(RUNTIME_LINE, OLD_RUNTIME_LINE)
    observed, expected = run_to_discovery(tmp_path / "negative", old, inherited)
    with pytest.raises(AssertionError):
        assert observed == expected
