"""Host integrity and no-false-green accounting; real DSO gates run separately."""

import ctypes
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from actlize_la import _library
from actlize_la._serving_contract import (
    EXPORTS,
    HEADERS,
    REQUIRED_BINDINGS,
    STRUCT_NAMES,
)
from actlize_la._version import __version__


def seal(root, manifest):
    path = root / "_native/manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False))
    (root / "_native/wheel.json").write_text(
        json.dumps(
            dict(
                schema=1, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        )
    )


@pytest.fixture
def metadata_bundle(tmp_path):
    # Deliberately not executable: isolates the file-only integrity checker.
    resources = list(REQUIRED_BINDINGS) + [
        f"_native/ppu10/{n}" for n in (*HEADERS, "lib.so", "graph")
    ]
    for name in resources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"metadata-only fixture, NOT a native backend")
    manifest = dict(
        schema=1,
        version=__version__,
        source_revision="a" * 40,
        source_dirty=False,
        source_identity={"note": "非 ASCII 元数据"},
        default_target="ppu10",
        targets={
            "ppu10": dict(
                library="_native/ppu10/lib.so",
                native_graph_example="_native/ppu10/graph",
                compatibility=dict(compiler_version="test-only"),
                abi=dict(
                    version=1,
                    target=100,
                    exports=list(EXPORTS),
                    struct_sizes={s: 1 for s in STRUCT_NAMES},
                ),
            )
        },
        resources={
            n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
            for n in resources
        },
    )
    seal(tmp_path, manifest)
    return tmp_path, manifest


def test_file_only_manifest_and_non_ascii(metadata_bundle, monkeypatch):
    root, manifest = metadata_bundle

    def forbidden(*args, **kwargs):
        raise AssertionError("file discovery must not call native or subprocess")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    info = _library._inspect(root, None)
    assert info["target"] == "ppu10"
    assert info["source_identity"]["note"] == "非 ASCII 元数据"
    assert (
        info["manifest_sha256"]
        == hashlib.sha256((root / "_native/manifest.json").read_bytes()).hexdigest()
    )
    json.dumps(info)


@pytest.mark.parametrize(
    "plant",
    [
        "manifest",
        "binding",
        "missing",
        "extra",
        "omitted-header",
        "omitted-binding",
        "target",
        "abi",
        "missing-default",
        "path-escape",
        "symlink",
        "missing-size",
    ],
)
def test_fail_closed(metadata_bundle, plant):
    root, m = metadata_bundle
    if plant == "manifest":
        (root / "_native/manifest.json").write_text("{}")
    elif plant == "binding":
        (root / "serving/gdn.py").write_text("wrong binding")
    elif plant == "missing":
        (root / "_native/ppu10/lib.so").rename(root / "parked.so")
    elif plant == "extra":
        (root / "shadow.py").write_text("pass")
    elif plant == "symlink":
        (root / "_native/ppu10/lib.so").rename(root / "parked.so")
        (root / "_native/ppu10/lib.so").symlink_to(root / "parked.so")
        m["resources"]["parked.so"] = m["resources"]["_native/ppu10/lib.so"]
        seal(root, m)
    else:
        if plant == "omitted-header":
            del m["resources"]["_native/ppu10/aux_v1.h"]
        elif plant == "omitted-binding":
            del m["resources"]["serving/gdn.py"]
        elif plant == "target":
            m["targets"]["ppu10"]["abi"]["target"] = 150
        elif plant == "abi":
            m["targets"]["ppu10"]["abi"]["version"] = 0
        elif plant == "missing-default":
            del m["default_target"]
        elif plant == "path-escape":
            m["resources"]["../escape"] = "0" * 64
        elif plant == "missing-size":
            del m["targets"]["ppu10"]["abi"]["struct_sizes"]["_Qsa"]
        seal(root, m)
    with pytest.raises(_library.LibraryError):
        _library._inspect(root, None)


def test_discovery_has_no_source_fallback(tmp_path):
    with pytest.raises(_library.LibraryError, match="complete precompiled"):
        _library._inspect(tmp_path, "ppu10")


def test_real_subprocess_top_import_does_not_load_torch():
    root = Path(__file__).resolve().parents[1]
    script = "import sys; import actlize_la; from actlize_la import get_library_path, get_library_info; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], cwd=root, check=True)


@pytest.mark.parametrize(
    "plant,flags,error,match",
    [
        ("old", [], ValueError, "ABI/target mismatch"),
        ("size", ["-DTEST_ABI_VERSION=1"], RuntimeError, "ABI size mismatch"),
        (
            "export",
            ["-DTEST_ABI_VERSION=1", "-DTEST_MISSING_EXPORT=1"],
            AttributeError,
            "ala_ple_hash_v1_run",
        ),
    ],
)
def test_real_native_admission_negatives(tmp_path, plant, flags, error, match):
    from actlize_la.serving import NativeLibrary

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable: actual ABI negative NOT_RUN")
    source = Path(__file__).with_name("serving_invalid_abi.c")
    library = tmp_path / (plant + ".so")
    subprocess.run(
        [compiler, "-shared", "-fPIC", *flags, str(source), "-o", str(library)],
        check=True,
    )
    with pytest.raises(error, match=match):
        NativeLibrary(library, target="ppu10")


@pytest.mark.parametrize(
    "count,skip,fail,rc,expected",
    [
        (3, 0, 0, 0, "PASS"),
        (3, 1, 0, 0, "NOT_RUN"),
        (0, 0, 0, 0, "NOT_RUN"),
        (1, 0, 0, 0, "NOT_RUN"),
        (3, 0, 1, 1, "FAIL"),
        (3, 0, 0, 2, "FAIL"),
    ],
)
def test_pytest_zero_skip_and_missing_component_never_pass(
    tmp_path, count, skip, fail, rc, expected
):
    source = Path(__file__).resolve().parents[1] / "tools/qualify_serving_wheel.py"
    spec = importlib.util.spec_from_file_location("wheel_qualifier", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = mod.FLARE_TESTS
    rows = []
    for i in range(count):
        child = (
            '<skipped message="device missing"/>'
            if i < skip
            else ("<failure/>" if i < skip + fail else "")
        )
        rows.append(
            f'<testcase classname="tests.components.{names[i]}" name="case">{child}</testcase>'
        )
    report = tmp_path / "junit.xml"
    report.write_text(
        "<testsuites><testsuite>" + "".join(rows) + "</testsuite></testsuites>"
    )
    got = mod.junit_verdict(report, rc, names)
    assert got["status"] == expected
    assert got["passed"] == count - skip - fail
