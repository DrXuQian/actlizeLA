"""Existing evidence only; no Torch, profiler or GPU needed for these tests."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pack_ppu10_geometry_light as light
from collect_ppu_gdn_acu import capture_arms
from sweep_ppu10_geometry import CONTROL, DELIVERIES, cells


class GeometryLightPack(unittest.TestCase):
    def setUp(self):
        self.run = Path("/workspace") / ("actlizeLA-geometry-light-test-" + uuid4().hex)
        self.run.mkdir()
        self.shapes = json.loads((ROOT / "dev/ppu/geometry_shapes.json").read_text())
        root_files = {
            "sha.txt": "d" * 40 + "\n", "geometry-shapes.json": json.dumps(self.shapes),
            "geometry-registration.md": "Synthetic preregistration, not device evidence", "binaries.sha256": "synthetic",
            "residual-correctness.log": "synthetic", "geometry-edge.log": "synthetic", "geometry-native.log": "synthetic",
            "comparison.json": json.dumps(dict(cases=[dict(g=g, shape=dict(**s, K=128, V=128),
                arms={"wy-" + d: {} for d in (*DELIVERIES, CONTROL)}) for _, s, g in cells(self.shapes)])),
            "source.diff": "",
        }
        for name, content in root_files.items():
            (self.run / name).write_text(content)
        self.write_manifest(self.run, root_files, "geometry.SHA256SUMS")
        for name, shape, gate in cells(self.shapes):
            bundle = self.run / name / "bundle"
            bundle.mkdir(parents=True)
            arms = capture_arms(bundle, "wy", DELIVERIES[0], CONTROL, DELIVERIES[1:])
            status = dict(status="PASS", errors=[], capture_order=[a.label for a in arms],
                capture_arms={a.label: dict(role=a.role, wy_delivery=a.delivery,
                    directory=str(a.directory.relative_to(bundle))) for a in arms},
                comparison_origin=dict(source_sha="d" * 40), probes={})
            for arm in arms:
                arm.directory.mkdir(exist_ok=True)
                receipt = dict(status="PASS", phase="subject", public_api_calls=1, gate=gate,
                    role=arm.role, wy_delivery=arm.delivery, shape=dict(**shape, K=128, V=128),
                    device=dict(uuid="synthetic-device", cu=72), torch="synthetic-runtime",
                    extension_sha256="synthetic-binding", library_sha256="synthetic-library")
                for suffix in (".json", "-preflight.json"):
                    (arm.directory / (arm.role + suffix)).write_text(json.dumps(receipt))
                for page in ("details", "raw"):
                    (arm.directory / f"{arm.role}-{page}.txt").write_text(
                        f"Synthetic all-kernel export, not device evidence: {name}/{arm.label}/{page}\n")
                    status["probes"][f"{arm.label}-{page}"] = dict(status="COLLECTED")
                (arm.directory / f"{arm.role}-g{gate}.report.acurep").write_bytes(b"RAW-REPORT-OMITTED" * 100)
                (arm.directory / f"{arm.role}-acu.log.command").write_text("synthetic acu command\n")
            (bundle / "STATUS.json").write_text(json.dumps(status))
            (bundle / "isa.txt").write_bytes(b"DISASSEMBLY-OMITTED" * 100)
            for directory, filename in (("sources", "unused.cpp"), ("binaries", "unused.so")):
                (bundle / directory).mkdir()
                (bundle / directory / filename).write_bytes(b"HEAVY-OMITTED" * 100)
            self.write_manifest(bundle, [str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()])

    @staticmethod
    def write_manifest(root, files, filename="SHA256SUMS"):
        (root / filename).write_text("".join(
            f"{hashlib.sha256((root / f).read_bytes()).hexdigest()}  {f}\n" for f in files))

    def first_bundle(self):
        name, _, _ = next(cells(self.shapes))
        return self.run / name / "bundle"

    def test_keeps_all60_exports_but_no_reports_binary_source_or_isa(self):
        with patch("subprocess.run", side_effect=AssertionError("packer executed a command")), redirect_stdout(io.StringIO()):
            archive = light.pack(self.run)
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            self.assertEqual(sum(n.endswith("-raw.txt") for n in names), 60)
            self.assertEqual(sum(n.endswith("-details.txt") for n in names), 60)
            self.assertFalse(any(n.endswith((".acurep", ".so", "/isa.txt")) or "/sources/" in n for n in names))
            index = json.load(tar.extractfile("LIGHT_INDEX.json"))
            self.assertEqual(index["arms"], 60)
            self.assertEqual(len(index["reports"]), 60)
            for line in tar.extractfile("LIGHT_SHA256SUMS").read().decode().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(tar.extractfile(name).read()).hexdigest(), digest)
        self.assertTrue(list(self.run.rglob("*.acurep")))
        self.assertFalse((self.run / "geometry.tar.gz").exists(), "must not rebuild large archive")

    def test_changed_counter_is_red_and_creates_no_output(self):
        (self.first_bundle() / "wy-raw.txt").write_text("changed after capture")
        with self.assertRaisesRegex(ValueError, "checksum"):
            light.pack(self.run)
        self.assertFalse((self.run / "geometry-light.tar.gz").exists())

    def test_missing_last_cell_or_export_is_red(self):
        name, _, _ = list(cells(self.shapes))[-1]
        path = self.run / name / "bundle/fla-raw.txt"
        path.rename(path.with_suffix(".held"))
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            light.inventory(self.run)

    def test_status_cannot_omit_an_arm_even_with_valid_checksum(self):
        bundle = self.first_bundle()
        status = json.loads((bundle / "STATUS.json").read_text())
        del status["capture_arms"]["wy-residual-geometry-v16-w4"]
        (bundle / "STATUS.json").write_text(json.dumps(status))
        hashes = light.manifest(bundle, "SHA256SUMS")
        self.write_manifest(bundle, hashes)
        with self.assertRaisesRegex(ValueError, "capture arms"):
            light.inventory(self.run)

    def test_failed_export_is_not_counted_as_counters(self):
        bundle = self.first_bundle()
        status = json.loads((bundle / "STATUS.json").read_text())
        status["probes"]["fla-raw"]["status"] = "UNAVAILABLE"
        (bundle / "STATUS.json").write_text(json.dumps(status))
        self.write_manifest(bundle, light.manifest(bundle, "SHA256SUMS"))
        with self.assertRaisesRegex(ValueError, "successful existing export"):
            light.inventory(self.run)

    def test_symlink_cannot_export_other_data(self):
        path = self.first_bundle() / "wy-raw.txt"
        backup = path.with_suffix(".held")
        path.rename(backup)
        path.symlink_to(backup)
        with self.assertRaisesRegex(ValueError, "symlink"):
            light.inventory(self.run)

    def test_existing_output_preserved(self):
        archive = self.run / "geometry-light.tar.gz"
        archive.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "preserve"):
            light.pack(self.run)
        self.assertEqual(archive.read_bytes(), b"preserve")

    def test_empty_original_report_is_red(self):
        report = next(self.first_bundle().glob("wy-*.acurep"))
        report.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty original report"):
            light.inventory(self.run)

    def test_automatic_selection_does_not_guess_between_runs(self):
        for i in range(2):
            path = self.run / f"actlizeLA-ppu10-geometry-run{i}"
            path.mkdir()
            (path / "geometry.SHA256SUMS").write_text("synthetic marker")
        with self.assertRaisesRegex(ValueError, "candidates"):
            light.choose_run(None, self.run)
        self.assertEqual(light.choose_run(self.run / "geometry.tar.gz"), self.run.resolve())


if __name__ == "__main__":
    unittest.main()
