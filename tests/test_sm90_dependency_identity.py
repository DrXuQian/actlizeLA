"""Actual filesystem/Git boundary tests; no CUDA, Torch, GPU or network."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sm90_build", ROOT / "tools/build_gdn_sm90.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class DependencyIdentity(unittest.TestCase):
    def setUp(self):
        self.root = Path("/workspace") / f"gdn-header-identity-{uuid4().hex}"
        self.dep = self.root / "third_party/cutlass"
        (self.dep / "include/cute").mkdir(parents=True)
        self.header = self.dep / "include/cute/tensor.hpp"
        self.header.write_text("// fixture v1\n")

    def test_archive_without_git_is_hashed_not_mislabeled(self):
        identity = build.dependency_identity(self.dep)
        self.assertIsNone(identity["dependency_revision"])
        self.assertIsNone(identity["dependency_containing_repository"])
        self.assertEqual(len(identity["dependency_tree_sha256"]), 64)

    def test_enclosing_git_revision_is_not_dependency_revision(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "add", "--", "third_party/cutlass/include/cute/tensor.hpp"], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Local test",
                        "-c", "user.email=local@example.invalid", "commit", "-qm", "fixture"], check=True)
        identity = build.dependency_identity(self.dep)
        self.assertIsNone(identity["dependency_revision"])
        self.assertEqual(identity["dependency_containing_repository"]["subtree"], "third_party/cutlass")

    def test_untracked_or_modified_header_invalidates_reuse(self):
        first = build.dependency_identity(self.dep)
        obj = self.root / "object.o"
        obj.write_bytes(b"synthetic object")
        previous = first | {"device_object_sha256": build.sha(obj)}
        build.admit_reuse(previous, first, obj)
        self.header.write_text("// fixture v2\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            build.admit_reuse(previous, build.dependency_identity(self.dep), obj)
        self.header.write_text("// fixture v1\n")
        (self.dep / "include/cute/new.hpp").write_text("// previously untracked\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            build.admit_reuse(previous, build.dependency_identity(self.dep), obj)

    def test_old_partial_manifest_is_not_a_reuse_certificate(self):
        identity = build.dependency_identity(self.dep)
        previous = dict(identity)
        previous.pop("dependency_tree_sha256")
        with self.assertRaisesRegex(ValueError, "complete header hash"):
            build.admit_reuse(previous, identity, self.root / "unused.o")


if __name__ == "__main__":
    unittest.main()
