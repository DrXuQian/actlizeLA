"""Release metadata/negative controls; never claims native device admission."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest

TOOLS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/verify_serving_release.py"))


@pytest.fixture
def assets(tmp_path):
    data = dict(schema=1, tag="v0.2.2", version="0.2.2", target="ppu10",
                source_revision="a" * 40, validation="HOST_QUALIFIED_DEVICE_NOT_RUN",
                wheel="test.whl", assets={})
    (tmp_path / "test.whl").write_bytes(b"fixture only, not a runnable wheel")
    qualification = dict(status=data["validation"], source_revision=data["source_revision"],
                         wheel_sha256=TOOLS["sha"](tmp_path / "test.whl"))
    (tmp_path / "qualification.json").write_text(json.dumps(qualification))
    for name in ("qualification-kit.tar.gz", "host-evidence.tar.gz"):
        (tmp_path / name).write_bytes(b"fixture")
    for name in ("test.whl", "qualification.json", "qualification-kit.tar.gz", "host-evidence.tar.gz"):
        data["assets"][name] = TOOLS["sha"](tmp_path / name)
    (tmp_path / "release-manifest.json").write_text(json.dumps(data))
    return tmp_path, data


def test_bound_release_preserves_not_run(assets):
    root, data = assets
    assert TOOLS["checked_assets"](root, "v0.2.2", "a" * 40) == data


@pytest.mark.parametrize("plant", ["source", "tag", "target", "device-claim", "missing", "payload", "qualification", "filename", "symlink"])
def test_release_rejects_wrong_identity_and_false_admission(assets, plant):
    root, data = assets
    if plant == "source":
        data["source_revision"] = "b" * 40
    elif plant == "tag":
        data["tag"] = "v0.2.1"
    elif plant == "target":
        data["target"] = "ppu15"
    elif plant == "device-claim":
        data["validation"] = "DEVICE_PASS"
    elif plant == "missing":
        del data["assets"]["host-evidence.tar.gz"]
    elif plant == "payload":
        (root / "test.whl").write_bytes(b"corruption")
    elif plant == "qualification":
        q = json.loads((root / "qualification.json").read_text())
        q["source_revision"] = "b" * 40
        (root / "qualification.json").write_text(json.dumps(q))
        data["assets"]["qualification.json"] = TOOLS["sha"](root / "qualification.json")
    elif plant == "filename":
        data["wheel"] = "../test.whl"
        data["assets"][data["wheel"]] = data["assets"].pop("test.whl")
    else:
        (root / "test.whl").rename(root / "original.whl")
        (root / "test.whl").symlink_to(root / "original.whl")
    (root / "release-manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        TOOLS["checked_assets"](root, "v0.2.2", "a" * 40)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b"])
def test_unsafe_wheel_member_is_rejected(tmp_path, name):
    wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(name, b"bad")
    with pytest.raises(ValueError):
        TOOLS["unpack_wheel"](wheel, tmp_path / "inspect")
