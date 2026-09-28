"""Configured metadata is not device evidence; exercise real API/runner branches."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict
import hashlib
import inspect
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from actlize_la import gdn_forward, get_device_profile
from actlize_la import device as metadata
from actlize_la.sm90_auto import AutoSm90Forward
from actlize_la.sm90_policy import DeviceProfile, policy, select

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def forbid_queries():
    with ExitStack() as stack:
        for name in ("actlize_la.device.device_profile", "actlize_la.sm90_policy.device_profile"):
            stack.enter_context(patch(name, side_effect=AssertionError("device query is forbidden")))
        yield


class Metadata(unittest.TestCase):
    def tearDown(self):
        metadata.device_profile.cache_clear()

    def test_configured_metadata_returns_without_importing_torch(self):
        code = """
import sys
sys.modules['torch'] = None  # Any attempted Torch import is a hard failure.
from actlize_la import get_device_profile
p = get_device_profile(mode='perfmodel', backend='ppu17', sm_count=20)
assert (p.name, p.sm_count, p.capability, p.source) == ('ppu17', 20, None, 'configured')
assert sys.modules['torch'] is None
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_default_model_size_or_hardware_fallback(self):
        with forbid_queries():
            for count in (None, 0, -1, True, 20.0, "20"):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                    get_device_profile(mode="perfmodel", backend="ppu17", sm_count=count)
            for backend in (None, "ppu", "unknown", "ppu15"):
                with self.subTest(backend=backend), self.assertRaises((ValueError, RuntimeError)):
                    get_device_profile(mode="perfmodel", backend=backend, sm_count=20)

    def test_mode_typos_or_model_inputs_in_device_mode_are_errors(self):
        with forbid_queries():
            for mode in ("perf_model", "", None):
                with self.assertRaisesRegex(ValueError, "mode must"):
                    get_device_profile(mode=mode, backend="ppu17", sm_count=20)
            with self.assertRaisesRegex(ValueError, "requires mode"):
                get_device_profile(0, backend="ppu17", sm_count=20)

    def test_counts_remain_per_call_and_never_enter_hardware_cache(self):
        metadata.device_profile.cache_clear()
        props = SimpleNamespace(name="physical-device", multi_processor_count=72, major=9, minor=0)
        query = Mock(return_value=props)
        fake = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=query))
        with patch.dict(sys.modules, {"torch": fake}):
            for count in (20, 40, 20):
                profile = get_device_profile(3, mode="perfmodel", backend="ppu17", sm_count=count)
                self.assertEqual((profile.sm_count, profile.source), (count, "configured"))
                query.assert_not_called()
            for _ in range(2):
                actual = get_device_profile(3, backend="ppu17")
                self.assertEqual((actual.sm_count, actual.source), (72, "measured"))
            query.assert_called_once_with(3)

    def test_configured_h800_identity_cannot_claim_measured_winner(self):
        shape = (1, 8192, 16, 32, 128, 128)  # Actual H800 row selects local-inverse.
        configured = DeviceProfile("NVIDIA H800 PCIe", 114, (9, 0), "configured")
        actual = DeviceProfile("NVIDIA H800 PCIe", 114, (9, 0))
        self.assertEqual(select(shape, actual).configuration, "value64-local-inverse")
        choice = select(shape, configured)
        self.assertEqual((choice.configuration, choice.basis),
                         ("value64", "perfmodel-unmeasured-default"))
        with self.assertRaisesRegex(ValueError, "profile source"):
            select(shape, DeviceProfile("NVIDIA H800 PCIe", 114, (9, 0), "unknown"))

    def test_plant_ignored_mode_is_detected_by_no_query_contract(self):
        # Execute a mutation of the actual resolver, not a parallel fake model.
        source = inspect.getsource(metadata.get_device_profile)
        self.assertEqual(source.count('if mode == "perfmodel":'), 1)
        namespace = dict(vars(metadata))
        namespace["device_profile"] = Mock(side_effect=AssertionError("device query is forbidden"))
        exec(source.replace('if mode == "perfmodel":', 'if False:'), namespace)
        with self.assertRaisesRegex(AssertionError, "device query is forbidden"):
            namespace["get_device_profile"](mode="perfmodel", backend="ppu17", sm_count=20)
        namespace["device_profile"].assert_called_once()


class Forward(unittest.TestCase):
    def test_unified_sm90_auto_and_loaded_bundle_use_configured_profile(self):
        device = SimpleNamespace(type="cuda", index=3)
        q = SimpleNamespace(ndim=4, shape=(1, 8192, 16, 128), device=device)
        v = SimpleNamespace(ndim=4, shape=(1, 8192, 32, 128), device=device)
        args = (q, object(), v, object(), object())
        state = object()
        ops = {name: Mock(configuration=name, return_value=(name, "state"))
               for name in policy()["configurations"]}
        automatic = AutoSm90Forward(ops)
        with forbid_queries(), patch("actlize_la.sm90_auto.default_forward", return_value=automatic):
            self.assertEqual(gdn_forward(*args, initial_state=state, backend="cuda_sm90",
                                         mode="perfmodel", sm_count=20), ("value64", "state"))
            choice = automatic.select(q, v, mode="perfmodel", sm_count=20)
            self.assertEqual(choice.basis, "perfmodel-unmeasured-default")
        ops["value64"].assert_called_once_with(*args, initial_state=state, output_final_state=True)
        ops["value64-local-inverse"].assert_not_called()
        ops["value128-paired"].assert_not_called()

    def test_explicit_ppu_modes_keep_existing_algorithm_and_state(self):
        args = tuple(object() for _ in range(5))
        state = object()
        for target, algorithm, entry in (("ppu10", "residual", "gdn_chunk_residual"),
                                          ("ppu17", "fused_sm90", "gdn_chunk_sm90")):
            for options in ({}, {"mode": "perfmodel", "sm_count": 20}):
                call = Mock(return_value=("O", "H"))
                with forbid_queries(), patch("actlize_la.gdn_interface.import_module",
                        return_value=SimpleNamespace(**{entry: call})):
                    self.assertEqual(gdn_forward(*args, initial_state=state, backend=target,
                                     algorithm=algorithm, output_final_state=False, **options), ("O", "H"))
                kwargs = dict(initial_state=state, output_final_state=False)
                if algorithm == "fused_sm90":
                    kwargs["backend"] = target
                else:
                    kwargs["delivery"] = "scalar"
                call.assert_called_once_with(*args, **kwargs)

    def test_invalid_model_call_fails_before_loading_or_launching(self):
        with forbid_queries(), patch("actlize_la.gdn_interface.import_module") as loader:
            for options in ({"mode": "perfmodel", "sm_count": 20},
                            {"mode": "perfmodel", "backend": "ppu17"},
                            {"mode": "perfmodel", "backend": "ppu17", "sm_count": 0},
                            {"mode": "device", "backend": "ppu17", "sm_count": 20}):
                with self.assertRaises(ValueError):
                    gdn_forward(1, 2, 3, 4, 5, **options)
            loader.assert_not_called()


class SingleCallRunner(unittest.TestCase):
    def test_actual_runner_records_configured_identity_without_query(self):
        import torch
        runner = runpy.run_path(str(ROOT / "tools/run_sm90_gdn.py"))
        main = runner["main"]
        folder = Path("/workspace") / f"actlizeLA-perfmodel-test-{uuid4().hex}"
        folder.mkdir()
        extension = folder / "_gdn_fused_sm90.test.so"
        extension.write_bytes(b"host-only mocked extension; not executable")
        receipt = dict(complete=True, target="ppu17", mode="source-check", configuration="control",
                       extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                       source_sha256={"launch.cu": "retained"})
        (folder / "build.json").write_text(json.dumps(receipt))
        cpu = runner["fixture"](1, 1, 1, 1, -.1)
        q, k, v, g, beta = cpu
        want = runner["torch_recurrent_gated_delta_rule"](q, k, v, g, beta, output_final_state=True)
        native = Mock(return_value=want)
        cuda = SimpleNamespace(set_device=Mock(), synchronize=Mock(),
            get_device_properties=Mock(side_effect=AssertionError("device query is forbidden")))
        proxy = SimpleNamespace(cuda=cuda, set_num_threads=Mock(), device=lambda *a: torch.device("cpu"),
                                zeros_like=torch.zeros_like)
        argv = ["runner", "--extension", str(extension), "--backend", "ppu17", "--source-check",
                "--out", str(folder / "results"), "--length", "1", "--q-heads", "1", "--v-heads", "1",
                "--mode", "perfmodel", "--sm-count", "20"]
        with forbid_queries(), patch.object(sys, "argv", argv), patch.dict(main.__globals__,
                {"torch": proxy, "gdn_chunk_sm90": native}), redirect_stdout(io.StringIO()):
            main()
        native.assert_called_once()
        cuda.get_device_properties.assert_not_called()
        report = json.loads((folder / "results/result.json").read_text())
        self.assertEqual(report["mode"], "perfmodel")
        self.assertEqual(report["device_profile"], asdict(DeviceProfile("ppu17", 20, None, "configured")))
        self.assertEqual(report["performance"], "NOT_MEASURED")
        self.assertEqual(report["public_calls"], 1)
        self.assertTrue(report["source_check"])  # Build provenance is not rewritten by model mode.

    def test_invalid_runner_mode_inputs_fail_before_fixture_or_extension_loading(self):
        runner = runpy.run_path(str(ROOT / "tools/run_sm90_gdn.py"))
        main = runner["main"]
        base = ["runner", "--extension", "/workspace/nonexistent-test.so", "--backend", "ppu17",
                "--out", "/workspace/nonexistent-results"]
        for flags in (["--mode", "perfmodel"], ["--mode", "perfmodel", "--sm-count", "0"],
                      ["--sm-count", "20"]):
            with forbid_queries(), patch.object(sys, "argv", base + flags), patch.dict(main.__globals__,
                    {"fixture": Mock(side_effect=AssertionError("must not create fixture"))}), \
                    patch("sys.stderr", io.StringIO()) as error, self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertNotIn("extension missing", error.getvalue())


if __name__ == "__main__":
    unittest.main()
