"""Packaging preserves six captures, their original SHA and missing-cell failures."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class Packaging(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path("/workspace") / ("actlizeLA-stages-pack-test-" + uuid4().hex)
        self.run_dir.mkdir()
        self.files = ["sha.txt", "binaries.sha256", "comparison.json", "residual-correctness.log"]
        for name in self.files:
            (self.run_dir / name).write_text("fixture\n")
        (self.run_dir / "sha.txt").write_text("d7c5c663bf87902251e29f6bc59aefd18d7488c3\n")
        for stage in ("solve", "output", "both"):
            for gate in ("-1.0", "-0.1"):
                name = f"acu-{stage}-g{gate}"
                (self.run_dir / name).mkdir()
                member = f"{name}/{name}.tar.gz"
                with tarfile.open(self.run_dir / member, "w:gz") as archive:
                    info = tarfile.TarInfo(name + "/evidence.txt")
                    content = f"{stage}/{gate}\n".encode()
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                self.files.append(member)

    def pack(self):
        return subprocess.run(["bash", str(ROOT / "tools/pack_ppu10_full_chunk_stages.sh"), str(self.run_dir)],
                              capture_output=True, text=True)

    def test_one_tar_contains_every_capture_and_preserves_sha_and_bytes(self):
        run = self.pack()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("captures=6 measurement_sha=d7c5c66", run.stdout)
        with tarfile.open(self.run_dir / "full-chunk-stages.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), {*self.files, "full-chunk-stages.SHA256SUMS"})
            manifest = archive.extractfile("full-chunk-stages.SHA256SUMS").read().decode().splitlines()
            self.assertEqual(len(manifest), 10)
            for line in manifest:
                digest, name = line.split("  ", 1)
                content = archive.extractfile(name).read()
                self.assertEqual(content, (self.run_dir / name).read_bytes())
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_missing_capture_is_red_and_does_not_make_partial_upload(self):
        missing = self.run_dir / self.files[-1]
        missing.rename(missing.with_suffix(".held"))
        run = self.pack()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("missing/empty/nonregular evidence", run.stderr)
        self.assertFalse((self.run_dir / "full-chunk-stages.tar.gz").exists())

    def test_existing_upload_is_not_overwritten(self):
        self.assertEqual(self.pack().returncode, 0)
        path = self.run_dir / "full-chunk-stages.tar.gz"
        before = path.read_bytes()
        run = self.pack()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("preserved without overwrite", run.stderr)
        self.assertEqual(path.read_bytes(), before)

    def test_bad_measurement_sha_is_not_replaced_with_current_checkout(self):
        (self.run_dir / "sha.txt").write_text("unknown\n")
        run = self.pack()
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("invalid measurement", run.stderr)
        self.assertFalse((self.run_dir / "full-chunk-stages.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
