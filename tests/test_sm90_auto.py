"""Independent measured-inventory, routing and fail-closed bundle checks."""
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from actlize_la import gdn_forward
from actlize_la.sm90_auto import AutoSm90Forward, _default_forward, load_bundle, sha
from actlize_la.sm90_policy import DeviceProfile, policy, policy_digest, select, tensor_backend

ROOT = Path(__file__).resolve().parents[1]
H800 = DeviceProfile("NVIDIA H800 PCIe", 114, (9, 0))
# Frozen pre-measurement workload authority, not generated from the selector.
# Source: sm90-multishape-20260926/tools/sm90_workloads.py.
WORKLOADS = {
    "seq512": (1, 512, 16, 32), "seq1024": (1, 1024, 16, 32),
    "seq2048": (1, 2048, 16, 32), "seq4096": (1, 4096, 16, 32),
    "seq8192": (1, 8192, 16, 32), "tail2051": (1, 2051, 16, 32),
    "batch2": (2, 2048, 16, 32), "batch4": (4, 2048, 16, 32),
    "heads64-gva4": (1, 2048, 16, 64), "heads64-gva2": (1, 2048, 32, 64),
    "heads32-gva1": (1, 2048, 32, 32), "heads16": (1, 2048, 8, 16),
    "vary-fp32": (1, 2048, 16, 32), "initial-vary": (1, 2048, 16, 32),
}
VARIANTS = {"value64-original": "value64",
            "value64-local-inverse-paired": "value64-local-inverse",
            "value128-packed-standard": "value128-paired"}


def tensors(shape):
    b, t, hk, hv = shape
    device = SimpleNamespace(type="cuda", index=0)
    q = SimpleNamespace(ndim=4, shape=(b, t, hk, 128), device=device)
    v = SimpleNamespace(ndim=4, shape=(b, t, hv, 128), device=device)
    return q, object(), v, object(), object()


def variants():
    return {name: Mock(configuration=name, return_value=(name, "state"))
            for name in policy()["configurations"]}


class Automatic(unittest.TestCase):
    def tearDown(self):
        _default_forward.cache_clear()

    def audit_inventory(self):
        recorded = json.loads((ROOT / "dev/backends/sm90_h800_closure_20260927.json").read_text())
        self.assertEqual(len(recorded["rows"]), 56)
        self.assertEqual(len(WORKLOADS), 14)
        self.assertEqual({r["workload"] for r in recorded["rows"]}, set(WORKLOADS))
        cells = [(r["workload"], r["gate"], r["family"]) for r in recorded["rows"]]
        self.assertEqual(len(set(cells)), 56)
        self.assertEqual(set(cells), {(name, gate, family) for name in WORKLOADS
                                     for gate in (-0.1, -1.0)
                                     for family in ("flashinfer", "flashqla")})
        selected_rows = policy()["rows"]
        self.assertEqual(len(selected_rows), 12)
        names = [n for row in selected_rows for n in row["workloads"]]
        self.assertEqual(len(names), 14)
        self.assertEqual(set(names), set(WORKLOADS))
        for row in selected_rows:
            for name in row["workloads"]:
                self.assertEqual(tuple(row["shape"]), WORKLOADS[name])
        count = 0
        for row in recorded["rows"]:
            if "variant" not in row:
                self.assertEqual(row["status"], "REFERENCE_NUMERIC_FAIL")
                continue
            choice = select((*WORKLOADS[row["workload"]], 128, 128), H800)
            self.assertEqual(choice.configuration, VARIANTS[row["variant"]])
            self.assertEqual(choice.basis, "h800-measured-shape")
            count += 1
        self.assertEqual(count, 55)

    def test_complete_measured_inventory_and_both_gates_have_one_selection(self):
        self.audit_inventory()

    def test_missing_denominator_and_wrong_winner_are_negative_controls(self):
        missing = copy.deepcopy(policy())
        missing["rows"].pop(0)  # default V64 would otherwise hide the missing row
        wrong = copy.deepcopy(policy())
        next(r for r in wrong["rows"] if r["workloads"] == ["seq8192"])["configuration"] = "value64"
        for plant in (missing, wrong):
            with patch("actlize_la.sm90_policy.policy", return_value=plant), patch(
                __name__ + ".policy", return_value=plant
            ), self.assertRaises(AssertionError):
                self.audit_inventory()

    def test_unknown_shape_or_different_sm90_device_is_not_a_measured_winner(self):
        for shape in ((2, 8192, 16, 32, 128, 128), (1, 8191, 16, 32, 128, 128)):
            choice = select(shape, H800)
            self.assertEqual((choice.configuration, choice.basis), ("value64", "unmeasured-shape-default"))
        for device in (DeviceProfile("NVIDIA H100", 114, (9, 0)),
                       DeviceProfile("NVIDIA H800", 132, (9, 0))):
            choice = select((2, 2048, 16, 32, 128, 128), device)
            self.assertEqual((choice.configuration, choice.basis), ("value64", "unmeasured-device-default"))

    def test_invalid_shapes_and_non_hopper_cannot_borrow_the_policy(self):
        for shape in ((0, 2048, 16, 32, 128, 128), (1, 2048, 16, 31, 128, 128),
                      (1, 2048, 16, 32, 64, 128)):
            with self.assertRaises(ValueError):
                select(shape, H800)
        for profile in (DeviceProfile("PPU", 72, (9, 0)), DeviceProfile("A800", 108, (8, 0))):
            with self.assertRaises(ValueError):
                select((1, 2048, 16, 32, 128, 128), profile)

    def test_unified_entry_selects_each_implementation_without_input_or_state_changes(self):
        ops = variants()
        automatic = AutoSm90Forward(ops)
        for shape, wanted in (((1, 2048, 16, 32), "value64"),
                              ((1, 8192, 16, 32), "value64-local-inverse"),
                              ((2, 2048, 16, 32), "value128-paired")):
            args = tensors(shape)
            state = object()
            with patch("actlize_la.sm90_policy.device_profile", return_value=H800), patch(
                "actlize_la.sm90_auto.default_forward", return_value=automatic
            ), patch("subprocess.run", side_effect=AssertionError("no runtime compiler")):
                self.assertEqual(gdn_forward(*args, initial_state=state, output_final_state=False),
                                 (wanted, "state"))
            ops[wanted].assert_called_once_with(*args, initial_state=state, output_final_state=False)

    def test_missing_candidate_is_not_a_fallback_or_an_original_route(self):
        for missing in policy()["configurations"]:
            ops = variants()
            del ops[missing]
            with self.assertRaisesRegex(ValueError, "every"):
                AutoSm90Forward(ops)
        args = tensors((1, 2048, 16, 32))
        with patch("actlize_la.sm90_policy.device_profile", return_value=H800), patch(
            "actlize_la.sm90_auto.default_forward", side_effect=RuntimeError("missing bundle")
        ), patch("actlize_la.gdn_interface.import_module") as old:
            with self.assertRaisesRegex(RuntimeError, "missing bundle"):
                gdn_forward(*args)
            old.assert_not_called()

    def test_auto_does_not_guess_ppu_generation_or_take_legacy_configuration(self):
        args = tensors((1, 2048, 16, 32))
        with patch("actlize_la.sm90_policy.device_profile", return_value=DeviceProfile("PPU-ZW810", 72, (9, 0))):
            with self.assertRaisesRegex(ValueError, "PPU requires"):
                gdn_forward(*args)
        with self.assertRaisesRegex(ValueError, "auto selects"):
            gdn_forward(*args, configuration="value64")

    def test_default_bundle_loaded_once_not_read_or_hashed_each_call(self):
        auto = Mock(return_value=("O", "H"))
        with patch.dict(os.environ, {"ACTLIZE_LA_SM90_BUNDLE": "/workspace/a-bundle"}), patch(
            "actlize_la.sm90_auto.load_bundle", return_value=auto
        ) as loader, patch("actlize_la.sm90_policy.device_profile", return_value=H800):
            args = tensors((1, 2048, 16, 32))
            for _ in range(8):
                self.assertEqual(gdn_forward(*args), ("O", "H"))
            loader.assert_called_once()

    def test_operator_override_must_not_be_empty_or_relative(self):
        for value in ("", " ", "build/sm90"):
            with self.assertRaises(ValueError):
                _default_forward(value)

    def test_registered_installation_resolves_without_environment_and_rejects_mutation(self):
        directory = Path("/workspace") / f"actlizeLA-auto-registration-test-{uuid4().hex}"
        directory.mkdir()
        bundle = directory / "bundle.json"
        bundle.write_text("registered bundle\n")
        record = directory / "installation.json"
        record.write_text(json.dumps(dict(schema=1, directory=str(directory), bundle_sha256=sha(bundle))))
        with patch("actlize_la.sm90_auto.REGISTRATION_FILE", record), patch(
            "actlize_la.sm90_auto.load_bundle", return_value="loaded"
        ) as loader:
            self.assertEqual(_default_forward(None), "loaded")
            loader.assert_called_once_with(directory)
            _default_forward.cache_clear()
            bundle.write_text("changed bundle\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                _default_forward(None)
            loader.assert_called_once()


class Bundles(unittest.TestCase):
    def fixture(self):
        directory = Path("/workspace") / f"actlizeLA-auto-test-{uuid4().hex}"
        directory.mkdir()
        entries = {}
        for name in policy()["configurations"]:
            member = directory / name
            member.mkdir()
            receipt = dict(compiler_sha256="compiler", dependency_tree_sha256="headers",
                           source_sha256={"launch.cu": "source"}, torch="2.9+cu128", configuration=name)
            (member / "build.json").write_text(json.dumps(receipt))
            entries[name] = dict(directory=name, receipt_sha256=sha(member / "build.json"))
        manifest = dict(schema=1, target="cuda_sm90", policy_sha256=policy_digest(), configurations=entries)
        (directory / "bundle.json").write_text(json.dumps(manifest))
        return directory, manifest

    def test_actual_manifest_with_injected_native_loader(self):
        directory, _ = self.fixture()
        ops = variants()
        with patch("actlize_la.sm90.load_sm90", side_effect=lambda p: ops[p.name]) as loader:
            result = load_bundle(directory)
            self.assertEqual(set(result._variants), set(ops))
            self.assertEqual(loader.call_count, 3)

    def test_missing_mixed_or_rebound_members_fail(self):
        for plant in ("missing", "policy", "receipt", "directory", "source", "label"):
            directory, manifest = self.fixture()
            first = policy()["configurations"][0]
            if plant == "missing":
                del manifest["configurations"][first]
            elif plant == "policy":
                manifest["policy_sha256"] = "wrong"
            elif plant == "receipt":
                manifest["configurations"][first]["receipt_sha256"] = "wrong"
            elif plant == "directory":
                manifest["configurations"][first]["directory"] = "../escape"
            elif plant == "source":
                receipt = directory / first / "build.json"
                data = json.loads(receipt.read_text())
                data["source_sha256"] = {"launch.cu": "different"}
                receipt.write_text(json.dumps(data))
                manifest["configurations"][first]["receipt_sha256"] = sha(receipt)
            (directory / "bundle.json").write_text(json.dumps(manifest))
            ops = variants()
            if plant == "label":
                ops[first].configuration = "control"
            with patch("actlize_la.sm90.load_sm90", side_effect=lambda p: ops[p.name]), self.assertRaises(ValueError):
                load_bundle(directory)


class Builder(unittest.TestCase):
    def test_builds_every_candidate_and_does_not_publish_a_partial_bundle(self):
        builder = runpy.run_path(str(ROOT / "tools/build_sm90_bundle.py"))
        main = builder["main"]
        # One failed compile must stop before assembly/registration, never
        # announce a working auto install with an absent configuration.
        for fail in (False, True):
            directory = Path("/workspace") / f"actlizeLA-auto-builder-test-{uuid4().hex}"
            effect = subprocess.CalledProcessError(1, "compiler") if fail else None
            with patch.object(sys, "argv", ["builder", "--out", str(directory)]), patch(
                "subprocess.run", side_effect=effect
            ) as compiler, patch.dict(main.__globals__, {"assemble": Mock()}) as scope:
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError):
                        main()
                    scope["assemble"].assert_not_called()
                else:
                    main()
                    commands = [c.args[0] for c in compiler.call_args_list]
                    self.assertEqual([c[c.index("--configuration") + 1] for c in commands],
                                     policy()["configurations"])
                    scope["assemble"].assert_called_once_with(directory.resolve())


if __name__ == "__main__":
    unittest.main()
