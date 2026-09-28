#!/usr/bin/env python3
"""CPU tests for opt-in loading, timing order and paired admission."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
from types import SimpleNamespace
import unittest
from uuid import uuid4
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]
from actlize_la import gdn_wy_interface as api
import bench_ppu_wy_fla as benchmark
from bench_ppu_wy_fla import (comparison_summary, delivery_comparisons, order,
                              DELIVERY_ROLES, TILE_ROLES, STATE_ROLES, STAGE_ROLES,
                              PREPARE_ROWS_ROLES, experiment, resolve_samples)
from bench_ppu_wy_fla import AIU_ROLES, SPLIT_PREPARE_ROLES, STATE_PIPELINE_ROLES
from bench_ppu_gdn_fla import checked_pair, verdict


class Contracts(unittest.TestCase):
    def tearDown(self):
        api._backend.cache_clear()

    def test_opt_in_required(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GDN_QSA_WY_EXTENSION"):
                api._backend()

    def test_empty_path_not_directory_fallback(self):
        with patch.dict("os.environ", {"GDN_QSA_WY_EXTENSION": ""}, clear=True):
            with self.assertRaises(RuntimeError):
                api._backend()

    def test_six_orders_balance_positions(self):
        rows = [order(i) for i in range(6)]
        self.assertEqual(len(set(rows)), 6)
        for column in zip(*rows):
            for role in ("original", "wy", "fla"):
                self.assertEqual(column.count(role), 2)

    def test_delivery_orders_are_complete_and_balanced(self):
        rows = [order(i, DELIVERY_ROLES) for i in range(14)]
        self.assertEqual(len(set(rows)), 14)
        for row in rows:
            self.assertEqual(set(row), set(DELIVERY_ROLES))
        for column in zip(*rows):
            for role in DELIVERY_ROLES:
                self.assertEqual(column.count(role), 2)

    def test_tiled_orders_and_masks_are_a_distinct_balanced_family(self):
        names, roles = experiment(tile_ab=True)
        self.assertEqual(roles, TILE_ROLES)
        self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (8, 16, 32, 48, 56))
        self.assertEqual(len(roles), 8)
        rows = [order(i, roles) for i in range(resolve_samples(None, roles))]
        self.assertEqual(len(set(rows)), 16)
        for row in rows:
            self.assertEqual(set(row), set(roles))
        for column in zip(*rows):
            for role in roles:
                self.assertEqual(column.count(role), 2)
        with self.assertRaises(ValueError):
            experiment(delivery_ab=True, tile_ab=True)

    def test_missing_state_output_cell_cannot_shrink_the_denominator(self):
        def check_inventory():
            _, roles = experiment(tile_ab=True)
            self.assertEqual(set(roles), {"original", "wy", "fla", "wy-tiled-prepare",
                "wy-tiled-state", "wy-tiled-output", "wy-tiled-state-output", "wy-tiled-all"})
            self.assertEqual(len(roles), 8)
        check_inventory()
        missing = tuple(name for name in api.TILED_DELIVERIES if name != "tiled-state-output")
        with patch.object(benchmark, "TILED_DELIVERIES", missing):
            with self.assertRaises(AssertionError):
                check_inventory()

    def test_aiu_inventory_masks_and_orders(self):
        def inventory():
            names, roles = experiment(aiu_ab=True)
            self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (1520, 5616, 9712, 13808))
            self.assertEqual(roles, AIU_ROLES)
            self.assertEqual(len(roles), 7)
            return roles
        roles = inventory()
        self.assertEqual(resolve_samples(None, roles), 14)
        for column in zip(*(order(i, roles) for i in range(14))):
            for role in roles:
                self.assertEqual(column.count(role), 2)
        with patch.object(benchmark, "AIU_DELIVERIES", api.AIU_DELIVERIES[:-1]):
            with self.assertRaises(AssertionError):
                inventory()
        for name in ("delivery_ab", "tile_ab", "state_ab", "stage_ab", "prepare_rows_ab"):
            with self.assertRaises(ValueError):
                experiment(aiu_ab=True, **{name: True})

    def test_split_prepare_inventory_and_default_are_independent(self):
        names, roles = experiment(split_prepare_ab=True)
        self.assertEqual(names, ("aiu-state-output", "split-prepare"))
        self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (13808, 30192))
        self.assertEqual(roles, SPLIT_PREPARE_ROLES)
        self.assertEqual(len(roles), 5)
        self.assertEqual(resolve_samples(None, roles), 10)
        for column in zip(*(order(i, roles) for i in range(10))):
            for role in roles:
                self.assertEqual(column.count(role), 2)
        self.assertEqual(experiment()[1], ("original", "wy", "fla"))
        for flag in ("delivery_ab", "tile_ab", "state_ab", "stage_ab", "prepare_rows_ab", "aiu_ab"):
            with self.assertRaises(ValueError):
                experiment(split_prepare_ab=True, **{flag: True})

    def test_split_prepare_verdict_never_substitutes_an_old_control(self):
        arms = {role: dict(samples_us=[310., 315.]) for role in SPLIT_PREPARE_ROLES}
        arms["wy-split-prepare"]["samples_us"] = [320., 325.]
        result = delivery_comparisons(arms, "split-prepare", split_prepare_ab=True)
        self.assertEqual(set(result), {"original", "wy", "fla", "wy-aiu-state-output"})
        self.assertTrue(all(row["verdict"] == "CONTROL-WINS" for row in result.values()))
        arms["wy-split-prepare"]["samples_us"] = [300., 312.]
        self.assertTrue(all(row["verdict"] == "UNRESOLVED" for row in
            delivery_comparisons(arms, "split-prepare", split_prepare_ab=True).values()))
        del arms["wy-aiu-state-output"]
        with self.assertRaises(KeyError):
            delivery_comparisons(arms, "split-prepare", split_prepare_ab=True)

    def test_state_pipeline_inventory_masks_default_and_missing_arm(self):
        def inventory():
            names, roles = experiment(state_pipeline_ab=True)
            self.assertEqual(names, ("split-prepare", "state-pipeline"))
            self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (30192,62960))
            self.assertEqual(roles, STATE_PIPELINE_ROLES)
            self.assertEqual(len(roles), 5)
            return roles
        roles = inventory()
        for column in zip(*(order(i,roles) for i in range(resolve_samples(None,roles)))):
            for role in roles: self.assertEqual(column.count(role),2)
        self.assertEqual(experiment()[1],("original","wy","fla"))
        with patch.object(benchmark,"STATE_PIPELINE_DELIVERIES",("state-pipeline",)):
            with self.assertRaises(AssertionError): inventory()
        for flag in ("delivery_ab","tile_ab","state_ab","stage_ab","prepare_rows_ab","aiu_ab","split_prepare_ab"):
            with self.assertRaises(ValueError): experiment(state_pipeline_ab=True,**{flag:True})

    def test_pipeline_losing_verdict_keeps_split_incumbent(self):
        arms = {role:dict(samples_us=[300.,310.]) for role in STATE_PIPELINE_ROLES}
        arms["wy-state-pipeline"]["samples_us"]=[320.,330.]
        verdicts=delivery_comparisons(arms,"state-pipeline",state_pipeline_ab=True)
        self.assertEqual(set(verdicts),{"original","wy","fla","wy-split-prepare"})
        self.assertTrue(all(x["verdict"]=="CONTROL-WINS" for x in verdicts.values()))
        arms["wy-state-pipeline"]["samples_us"]=[290.,305.]
        self.assertTrue(all(x["verdict"]=="UNRESOLVED" for x in
            delivery_comparisons(arms,"state-pipeline",state_pipeline_ab=True).values()))
        del arms["wy-split-prepare"]
        with self.assertRaises(KeyError): delivery_comparisons(arms,"state-pipeline",state_pipeline_ab=True)

    def test_aiu_verdict_keeps_incumbent_and_both_single_changes(self):
        arms = {role: dict(samples_us=[500., 510.]) for role in AIU_ROLES}
        arms["wy-aiu-state-output"]["samples_us"] = [400., 410.]
        comparisons = delivery_comparisons(arms, "aiu-state-output", aiu_ab=True)
        self.assertEqual(set(comparisons), set(AIU_ROLES) - {"wy-aiu-state-output"})
        self.assertTrue(all(x["verdict"] == "CANDIDATE-WINS" for x in comparisons.values()))
        arms["wy-aiu-state-output"]["samples_us"] = [600., 610.]
        self.assertTrue(all(x["verdict"] == "CONTROL-WINS" for x in
                            delivery_comparisons(arms, "aiu-state-output", aiu_ab=True).values()))
        arms["wy-aiu-state-output"]["samples_us"] = [400., 610.]
        self.assertTrue(all(x["verdict"] == "UNRESOLVED" for x in
                            delivery_comparisons(arms, "aiu-state-output", aiu_ab=True).values()))
        del arms["wy-aiu-output"]
        with self.assertRaises(KeyError):
            delivery_comparisons(arms, "aiu-state-output", aiu_ab=True)

    def test_aiu_main_serializes_complete_family_metadata(self):
        directory = Path("/workspace") / f"gdn-wy-metadata-{uuid4().hex}"
        directory.mkdir()
        binding = directory / "synthetic.so"
        binding.write_bytes(b"test identity only; never loaded")
        output = directory / "comparison.json"
        argv = ["bench_ppu_wy_fla.py", "--extension", str(binding), "--wy-extension", str(binding),
                "--results", str(output), "--aiu-ab"]
        props = SimpleNamespace(name="PPU synthetic")
        with patch.object(sys, "argv", argv), patch.object(torch.cuda, "set_device"), \
                patch.object(torch.cuda, "get_device_properties", return_value=props), \
                patch.object(benchmark, "load_fla", return_value=(None, {})), \
                patch.object(benchmark, "compare", side_effect=lambda fn, gate, args, device: dict(g=gate)), \
                patch.dict("os.environ", {}, clear=True), redirect_stdout(io.StringIO()):
            benchmark.main()
        result = json.loads(output.read_text())
        self.assertTrue(result["delivery_ab"])
        self.assertTrue(result["aiu_ab"])
        self.assertEqual(result["roles"], list(AIU_ROLES))
        self.assertEqual(result["samples"], 14)
        self.assertEqual([case["g"] for case in result["cases"]], [-.1, -1.])

    def test_sample_count_tracks_actual_family_and_rejects_old_fourteen(self):
        self.assertEqual(resolve_samples(None, experiment()[1]), 12)
        self.assertEqual(resolve_samples(None, DELIVERY_ROLES), 14)
        self.assertEqual(resolve_samples(None, TILE_ROLES), 16)
        self.assertEqual(resolve_samples(32, TILE_ROLES), 32)
        for samples in (0, 7, 14, 15, 17, 24):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "multiple of 16"):
                resolve_samples(samples, TILE_ROLES)

    def test_state_experiment_inventory_and_balanced_orders(self):
        def check_inventory():
            names, roles = experiment(state_ab=True)
            self.assertEqual(set(roles), {"original", "wy", "fla", "wy-tiled-state-output",
                "wy-tiled-all", "wy-tiled-state-output-address", "wy-tiled-state-output-gates",
                "wy-tiled-state-output-both"})
            self.assertEqual(tuple(api.DELIVERIES[x] for x in names), (48, 56, 112, 176, 240))
            self.assertEqual(roles, STATE_ROLES)
            return roles
        roles = check_inventory()
        self.assertEqual(resolve_samples(None, roles), 16)
        rows = [order(i, roles) for i in range(16)]
        for column in zip(*rows):
            for role in roles:
                self.assertEqual(column.count(role), 2)
        with patch.object(benchmark, "STATE_DELIVERIES", api.STATE_DELIVERIES[:-1]):
            with self.assertRaises(AssertionError):
                check_inventory()
        for kwargs in (dict(delivery_ab=True), dict(tile_ab=True)):
            with self.assertRaises(ValueError):
                experiment(state_ab=True, **kwargs)
        with self.assertRaises(ValueError):
            resolve_samples(14, roles)

    def test_state_option_bits_reach_the_actual_python_abi(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        x, fake = torch.zeros(1), Fake()
        for suffix, expected in (("address", 112), ("gates", 176), ("both", 240)):
            delivery = f"tiled-state-output-{suffix}"
            def check():
                with patch.object(api, "_backend", return_value=fake):
                    api.gdn_chunk_wy(x, x, x, x, x, delivery=delivery)
                self.assertEqual(fake.args[-1], expected)
            check()
            # Dropping option bits keeps a numerically equal control; require
            # the selection itself, not just its answer, to differ.
            with patch.dict(api.DELIVERIES, {delivery: 48}):
                with self.assertRaises(AssertionError):
                    check()

    def test_state_combined_candidate_compares_both_single_changes(self):
        arms = {role: dict(samples_us=[500., 510.]) for role in STATE_ROLES}
        arms["wy-tiled-state-output-both"]["samples_us"] = [400., 410.]
        result = delivery_comparisons(arms, "tiled-state-output-both", state_ab=True)
        self.assertEqual(set(result), set(STATE_ROLES) - {"wy-tiled-state-output-both"})
        self.assertTrue(all(x['verdict'] == 'CANDIDATE-WINS' for x in result.values()))
        arms["wy-tiled-state-output-both"]["samples_us"] = [400., 550.]
        self.assertTrue(all(x['verdict'] == 'UNRESOLVED' for x in
            delivery_comparisons(arms, "tiled-state-output-both", state_ab=True).values()))
        del arms["wy-tiled-state-output-address"]
        with self.assertRaises(KeyError):
            delivery_comparisons(arms, "tiled-state-output-both", state_ab=True)

    def test_stage_inventory_masks_and_balanced_orders(self):
        def inventory():
            names, roles = experiment(stage_ab=True)
            self.assertEqual(set(roles), {"original", "wy", "fla", "wy-tiled-state-output",
                "wy-tiled-state-output-both", "wy-stage-address-prepare", "wy-stage-address-output",
                "wy-stage-address-both"})
            self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (48, 240, 496, 752, 1008))
            self.assertEqual(roles, STAGE_ROLES)
            return roles
        roles = inventory()
        self.assertEqual(resolve_samples(None, roles), 16)
        for column in zip(*(order(i, roles) for i in range(16))):
            for role in roles:
                self.assertEqual(column.count(role), 2)
        with patch.object(benchmark, "STAGE_DELIVERIES", api.STAGE_DELIVERIES[:-1]):
            with self.assertRaises(AssertionError):
                inventory()
        for kwargs in (dict(delivery_ab=True), dict(tile_ab=True), dict(state_ab=True)):
            with self.assertRaises(ValueError):
                experiment(stage_ab=True, **kwargs)
        with self.assertRaises(ValueError):
            resolve_samples(14, roles)

    def test_stage_bits_reach_the_python_abi_and_do_not_drop_state_both(self):
        class Fake:
            def forward(self, *args):
                self.mask = args[-1]
                return torch.ones(1), torch.ones(1)
        x, fake = torch.zeros(1), Fake()
        for suffix, expected in (("prepare", 496), ("output", 752), ("both", 1008)):
            name = f"stage-address-{suffix}"
            def check():
                with patch.object(api, "_backend", return_value=fake):
                    api.gdn_chunk_wy(x, x, x, x, x, delivery=name)
                self.assertEqual(fake.mask, expected)
                self.assertEqual(fake.mask & 255, 240)
            check()
            for dropped in (expected & 255, expected & ~192):
                with patch.dict(api.DELIVERIES, {name: dropped}), self.assertRaises(AssertionError):
                    check()

    def test_stage_combined_compares_mask240_and_both_singles(self):
        arms = {role: dict(samples_us=[400., 410.]) for role in STAGE_ROLES}
        candidate = arms["wy-stage-address-both"]
        candidate["samples_us"] = [350., 360.]
        result = delivery_comparisons(arms, "stage-address-both", stage_ab=True)
        self.assertEqual(set(result), set(STAGE_ROLES) - {"wy-stage-address-both"})
        self.assertTrue(all(x['verdict'] == 'CANDIDATE-WINS' for x in result.values()))
        candidate["samples_us"] = [350., 420.]
        self.assertTrue(all(x['verdict'] == 'UNRESOLVED' for x in
            delivery_comparisons(arms, "stage-address-both", stage_ab=True).values()))
        candidate["samples_us"] = [420., 430.]
        self.assertTrue(all(x['verdict'] == 'CONTROL-WINS' for x in
            delivery_comparisons(arms, "stage-address-both", stage_ab=True).values()))
        del arms["wy-tiled-state-output-both"]
        with self.assertRaises(KeyError):
            delivery_comparisons(arms, "stage-address-both", stage_ab=True)

    def test_prepare_rows_inventory_and_complete_balanced_cycle(self):
        def inventory():
            names, roles = experiment(prepare_rows_ab=True)
            self.assertEqual(set(roles), {"original", "wy", "fla", "wy-tiled-state-output",
                "wy-tiled-state-output-both", "wy-stage-address-prepare", "wy-prepare-rows-shared",
                "wy-prepare-rows-warp"})
            self.assertEqual(tuple(api.DELIVERIES[name] for name in names), (48,240,496,1520,2544))
            self.assertEqual(roles, PREPARE_ROWS_ROLES)
            return roles
        roles = inventory()
        self.assertEqual(resolve_samples(None, roles),16)
        for column in zip(*(order(i,roles) for i in range(16))):
            for role in roles:
                self.assertEqual(column.count(role),2)
        with patch.object(benchmark,"PREPARE_ROWS_DELIVERIES",api.PREPARE_ROWS_DELIVERIES[:-1]):
            with self.assertRaises(AssertionError):
                inventory()
        for kwargs in (dict(delivery_ab=True),dict(tile_ab=True),dict(state_ab=True),dict(stage_ab=True)):
            with self.assertRaises(ValueError):
                experiment(prepare_rows_ab=True,**kwargs)
        with self.assertRaises(ValueError):
            resolve_samples(14,roles)

    def test_prepare_rows_mode_bits_reach_real_abi_with_incumbent_intact(self):
        class Fake:
            def forward(self,*args):
                self.mask=args[-1]
                return torch.ones(1),torch.ones(1)
        x,fake=torch.zeros(1),Fake()
        for suffix,expected in (("shared",1520),("warp",2544)):
            name=f"prepare-rows-{suffix}"
            def check():
                with patch.object(api,"_backend",return_value=fake):
                    api.gdn_chunk_wy(x,x,x,x,x,delivery=name)
                self.assertEqual(fake.mask,expected)
                self.assertEqual(fake.mask & 1023,496)
            check()
            for wrong in (496,expected & ~256,expected | 512):
                with patch.dict(api.DELIVERIES,{name:wrong}),self.assertRaises(AssertionError):
                    check()

    def test_prepare_rows_compare_incumbent_and_each_other(self):
        arms={role:dict(samples_us=[400.,410.]) for role in PREPARE_ROWS_ROLES}
        candidate=arms['wy-prepare-rows-shared']
        candidate['samples_us']=[350.,360.]
        result=delivery_comparisons(arms,'prepare-rows-shared',prepare_rows_ab=True)
        self.assertEqual(set(result),set(PREPARE_ROWS_ROLES)-{'wy-prepare-rows-shared'})
        self.assertTrue(all(x['verdict']=='CANDIDATE-WINS' for x in result.values()))
        candidate['samples_us']=[350.,420.]
        self.assertTrue(all(x['verdict']=='UNRESOLVED' for x in
            delivery_comparisons(arms,'prepare-rows-shared',prepare_rows_ab=True).values()))
        candidate['samples_us']=[420.,430.]
        self.assertTrue(all(x['verdict']=='CONTROL-WINS' for x in
            delivery_comparisons(arms,'prepare-rows-shared',prepare_rows_ab=True).values()))
        del arms['wy-stage-address-prepare']
        with self.assertRaises(KeyError):
            delivery_comparisons(arms,'prepare-rows-shared',prepare_rows_ab=True)

    def test_state_output_keeps_scalar_prepare_at_the_real_python_abi(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        x, fake = torch.zeros(1), Fake()
        def check_selected_mask():
            with patch.object(api, "_backend", return_value=fake):
                api.gdn_chunk_wy(x, x, x, x, x, delivery="tiled-state-output")
            self.assertEqual(len(fake.args), 8)
            self.assertEqual(fake.args[-1], 48)
        check_selected_mask()
        for wrong_mask in (0, 16, 32, 56):
            with self.subTest(mask=wrong_mask), patch.dict(api.DELIVERIES, {"tiled-state-output": wrong_mask}):
                with self.assertRaises(AssertionError):
                    check_selected_mask()

    def test_state_output_verdict_compares_measured_pair_to_each_incumbent(self):
        arms = {role: dict(samples_us=times) for role, times in (
            ("wy", [700., 710.]), ("fla", [480., 490.]),
            ("original", [420., 430.]), ("wy-tiled-state", [450., 460.]),
            ("wy-tiled-all", [440., 446.]), ("wy-tiled-state-output", [432., 438.]))}
        result = delivery_comparisons(arms, "tiled-state-output")
        self.assertEqual(set(result), {"wy", "fla", "original", "wy-tiled-state", "wy-tiled-all"})
        self.assertEqual(result["original"]["verdict"], "CONTROL-WINS")
        self.assertEqual(result["wy-tiled-all"]["verdict"], "CANDIDATE-WINS")
        self.assertAlmostEqual(result["wy-tiled-all"]["descriptive_speedup"], 443. / 435.)
        # A low median cannot erase the observed overlap or a losing arm.
        arms["wy-tiled-state-output"]["samples_us"] = [432., 435., 480.]
        self.assertEqual(delivery_comparisons(arms, "tiled-state-output")["wy-tiled-all"]["verdict"],
                         "UNRESOLVED")
        arms["wy-tiled-state-output"]["samples_us"] = [760., 770.]
        self.assertTrue(all(x["verdict"] == "CONTROL-WINS"
                            for x in delivery_comparisons(arms, "tiled-state-output").values()))
        del arms["wy-tiled-all"]
        with self.assertRaises(KeyError):
            delivery_comparisons(arms, "tiled-state-output")

    def test_complete_comparison_runs_pair_and_rejects_raw_bit_drift(self):
        # Exercise actual admission/call binding/timing loops on CPU. Events
        # are synthetic: this tests plumbing, not any device timing claim.
        class Event:
            def __init__(self, **kwargs):
                if getattr(args, "admission_only", False):
                    raise AssertionError("numeric-only ACU admission created a timing event")
            def record(self):
                pass
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return 1.0
        cpu = tuple(torch.ones(1) for _ in range(5))
        want = (torch.ones(1), torch.ones(1))
        args = SimpleNamespace(delivery_ab=False, tile_ab=True, warmup=5, samples=16, launches=10)
        seen = []
        def run(plant=False, state_ab=False, stage_ab=False, prepare_rows_ab=False, aiu_ab=False, split_prepare_ab=False,
                admission_only=False, state_pipeline_ab=False):
            args.tile_ab = not (state_ab or stage_ab or prepare_rows_ab or aiu_ab or split_prepare_ab or state_pipeline_ab)
            args.state_ab = state_ab
            args.stage_ab = stage_ab
            args.prepare_rows_ab = prepare_rows_ab
            args.aiu_ab = aiu_ab
            args.split_prepare_ab = split_prepare_ab
            args.state_pipeline_ab = state_pipeline_ab
            args.admission_only = admission_only
            args.samples = 0 if admission_only else 10 if split_prepare_ab or state_pipeline_ab else 14 if aiu_ab else 16
            def wy(*inputs, delivery="scalar"):
                seen.append(delivery)
                # Within the unchanged 2% gate, but not scalar raw equality.
                selected = ("state-pipeline" if state_pipeline_ab else "split-prepare" if split_prepare_ab else "aiu-state-output" if aiu_ab else "prepare-rows-warp" if prepare_rows_ab else "stage-address-both" if stage_ab else
                            "tiled-state-output-both" if state_ab else "tiled-state-output")
                return (want[0] + .001, want[1]) if plant and delivery == selected else want
            with patch.object(benchmark.admission, "fixture", return_value=cpu), \
                    patch.object(benchmark.admission, "reference", return_value=want), \
                    patch.object(benchmark.admission, "gdn_chunk", return_value=want), \
                    patch.object(benchmark, "gdn_chunk_wy", side_effect=wy), \
                    patch.object(benchmark, "fla_call", return_value=lambda: want), \
                    patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "Event", Event), \
                    redirect_stdout(io.StringIO()):
                return benchmark.compare(None, -.1, args, torch.device("cpu"))
        result = run()
        self.assertEqual(set(result["arms"]), set(TILE_ROLES))
        for name in ("scalar", *api.TILED_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5 + 16 * 10)
        pair = result["arms"]["wy-tiled-state-output"]
        self.assertEqual(pair["delivery_mask"], 48)
        self.assertEqual(len(pair["samples_us"]), 16)
        self.assertEqual(len(pair["versus"]), 5)
        with self.assertRaisesRegex(AssertionError, "wy-tiled-state-output output/state bits differ"):
            run(plant=True)
        seen.clear()
        result = run(state_ab=True)
        self.assertEqual(set(result['arms']), set(STATE_ROLES))
        for name in ("scalar", *api.STATE_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5 + 16 * 10)
        self.assertEqual(result['arms']['wy-tiled-state-output-both']['delivery_mask'], 240)
        self.assertEqual(len(result['arms']['wy-tiled-state-output-both']['versus']), 7)
        with self.assertRaisesRegex(AssertionError, "wy-tiled-state-output-both output/state bits differ"):
            run(plant=True, state_ab=True)
        seen.clear()
        result = run(stage_ab=True)
        self.assertEqual(set(result['arms']), set(STAGE_ROLES))
        for name in ("scalar", *api.STAGE_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5 + 16 * 10)
        combined = result['arms']['wy-stage-address-both']
        self.assertEqual(combined['delivery_mask'], 1008)
        self.assertEqual(len(combined['versus']), 7)
        with self.assertRaisesRegex(AssertionError, "wy-stage-address-both output/state bits differ"):
            run(plant=True, stage_ab=True)
        seen.clear()
        result=run(prepare_rows_ab=True)
        self.assertEqual(set(result['arms']),set(PREPARE_ROWS_ROLES))
        for name in ('scalar',*api.PREPARE_ROWS_DELIVERIES):
            self.assertEqual(seen.count(name),8+5+16*10)
        self.assertEqual(result['arms']['wy-prepare-rows-warp']['delivery_mask'],2544)
        self.assertEqual(len(result['arms']['wy-prepare-rows-warp']['versus']),7)
        with self.assertRaisesRegex(AssertionError,'wy-prepare-rows-warp output/state bits differ'):
            run(plant=True,prepare_rows_ab=True)
        seen.clear()
        result = run(aiu_ab=True)
        self.assertEqual(set(result["arms"]), set(AIU_ROLES))
        for name in ("scalar", *api.AIU_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5 + 14 * 10)
        self.assertEqual(result["arms"]["wy-aiu-state-output"]["delivery_mask"], 13808)
        self.assertEqual(len(result["arms"]["wy-aiu-state-output"]["versus"]), 6)
        with self.assertRaisesRegex(AssertionError, "wy-aiu-state-output output/state bits differ"):
            run(plant=True, aiu_ab=True)
        seen.clear()
        result = run(split_prepare_ab=True)
        self.assertEqual(set(result["arms"]), set(SPLIT_PREPARE_ROLES))
        for name in ("scalar", *api.SPLIT_PREPARE_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5 + 10 * 10)
        self.assertEqual(result["arms"]["wy-split-prepare"]["delivery_mask"], 30192)
        self.assertEqual(len(result["arms"]["wy-split-prepare"]["versus"]), 4)
        with self.assertRaisesRegex(AssertionError, "wy-split-prepare output/state bits differ"):
            run(plant=True, split_prepare_ab=True)
        seen.clear()
        result = run(split_prepare_ab=True, admission_only=True)
        self.assertEqual(result["timing"], "NOT_RUN")
        self.assertEqual(result["scope"], "NUMERICS_ONLY_PERFORMANCE_NOT_MEASURED")
        for name in ("scalar", *api.SPLIT_PREPARE_DELIVERIES):
            self.assertEqual(seen.count(name), 8 + 5)
        for arm in result["arms"].values():
            self.assertEqual(arm["admitted_repeats"], 8)
            self.assertEqual(arm["samples_us"], [])
            self.assertNotIn("median_us", arm)
            self.assertNotIn("versus", arm)
        with self.assertRaisesRegex(AssertionError, "wy-split-prepare output/state bits differ"):
            run(plant=True, split_prepare_ab=True, admission_only=True)
        seen.clear()
        result=run(state_pipeline_ab=True,admission_only=True)
        self.assertEqual(set(result["arms"]),set(STATE_PIPELINE_ROLES))
        self.assertEqual(result["timing"],"NOT_RUN")
        for name in ("scalar",*api.STATE_PIPELINE_DELIVERIES): self.assertEqual(seen.count(name),8+5)
        for arm in result["arms"].values():
            self.assertEqual(arm["admitted_repeats"],8)
            self.assertEqual(arm["samples_us"],[])
            self.assertNotIn("median_us",arm)
        self.assertEqual(result["arms"]["wy-state-pipeline"]["delivery_mask"],62960)
        with self.assertRaisesRegex(AssertionError,"wy-state-pipeline output/state bits differ"):
            run(plant=True,state_pipeline_ab=True,admission_only=True)
        seen.clear()
        result=run(state_pipeline_ab=True)
        for name in ("scalar",*api.STATE_PIPELINE_DELIVERIES): self.assertEqual(seen.count(name),8+5+10*10)
        self.assertEqual(set(result["arms"]["wy-state-pipeline"]["versus"]),
                         {"original","wy","fla","wy-split-prepare"})

    def test_delivery_mask_is_consumed_not_silently_ignored(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        x = torch.zeros(1)
        fake = Fake()
        for name, mask in api.DELIVERIES.items():
            with patch.object(api, "_backend", return_value=fake):
                api.gdn_chunk_wy(x, x, x, x, x, delivery=name)
            self.assertEqual(len(fake.args), 7 if name == "scalar" else 8)
            if name != "scalar":
                self.assertEqual(fake.args[-1], mask)
        with patch.object(api, "_backend", side_effect=AssertionError("should not load")):
            with self.assertRaisesRegex(ValueError, "unknown WY delivery"):
                api.gdn_chunk_wy(x, x, x, x, x, delivery="typo")

    def test_wrong_values_are_failures(self):
        want = (torch.ones(3), torch.ones(4))
        checked_pair(want, want)
        for wrong in ((torch.zeros(3), want[1]), (want[0], torch.zeros(4)), (want[0], None),
                      (torch.full((3,), float("nan")), want[1])):
            with self.assertRaises(AssertionError):
                checked_pair(wrong, want)

    def test_both_verdict_directions_and_overlap(self):
        self.assertEqual(verdict([1, 2], [3, 4]), "OURS-WINS")
        self.assertEqual(verdict([3, 4], [1, 2]), "FLA-WINS")
        self.assertEqual(verdict([1, 3], [2, 4]), "UNRESOLVED")

    def test_fast_median_with_slow_samples_is_still_unresolved(self):
        arms = {role: dict(samples_us=times) for role, times in (
            ("original", [910.096, 915.720, 960.612]),
            ("wy", [713.744, 715.232, 726.492]),
            ("fla", [474.344, 480.806, 852.296]))}
        result = comparison_summary(arms)
        self.assertEqual(result["wy_vs_original"], "WY-WINS")
        self.assertEqual(result["wy_vs_fla"], "UNRESOLVED")
        self.assertAlmostEqual(result["wy_over_fla"], 715.232 / 480.806)
        self.assertEqual(result["ratio_scope"], "DESCRIPTIVE_MEDIANS_NOT_ADMISSION")
        # Removing the slow sample changes the answer: that is exactly why the
        # capture must not silently trim it to produce a desired winner.
        arms["fla"]["samples_us"] = [474.344, 480.806]
        self.assertEqual(comparison_summary(arms)["wy_vs_fla"], "FLA-WINS")

    def test_forwarding_and_no_state(self):
        class Fake:
            def forward(self, *args):
                self.args = args
                return torch.ones(1), torch.ones(1)
        fake = Fake()
        x = torch.zeros(3, 2).T
        with patch.object(api, "_backend", return_value=fake):
            out, state = api.gdn_chunk_wy(x, x, x, x, x, output_final_state=False)
        self.assertIsNone(state)
        self.assertTrue(all(t.is_contiguous() for t in fake.args[:5]))
        self.assertIsNone(fake.args[5])
        self.assertFalse(fake.args[6])


if __name__ == "__main__":
    unittest.main()
