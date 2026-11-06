"""One test inventory for the installed runner and its isolated subprocesses."""

import math

FLARE_TESTS = (
    "test_actlize_gdn_sequence_cuda",
    "test_actlize_mixing_cuda",
    "test_actlize_ple_cuda",
    "test_actlize_qsa_cuda",
)

HOST_TESTS = (
    "test_serving_contract", "test_aux_contract", "test_checked_gdn_contract",
    "test_serving_symbolic_handle", "test_gdn_core_contract",
)

CHECK_FILES = tuple(name + ".py" for name in HOST_TESTS) + (
    "serving_device.py", "aux_device.py", "serving_reference.py", "aux_reference.py",
    "test_ppu_default_device.py", "test_ppu_gdn_backend.py", "test_ppu_wy_backend.py",
    "residual_cpu.py", "wy_cpu.py",
)


def public_default_complete(data):
    """A PASS word cannot substitute for the public+prepared denominator."""
    expected = {(t, h, g) for t in (1, 63, 64, 65, 127, 128, 129, 2048, 8192)
                for h in (32, 48) for g in (-.1, -1.)}
    rows = data.get("cases", [])
    graphs = data.get("prepared_graphs", [])
    try:
        seen = {(row["shape"][1], row["shape"][3], row["gate"]) for row in rows}
        numeric = all(row["shape"] == [1, row["shape"][1], 16, row["shape"][3], 128]
                      and row["repeats"] == 8 and row["oracle"] == "NUMERIC/PASS"
                      and row["default_explicit"] == "RAW-BIT/PASS"
                      and row["wrong_state_transpose"] == "EXPECTED-RED/PASS"
                      and len(row["errors"]) == 2
                      and all(math.isfinite(e) and 0 <= e < .02 for e in row["errors"])
                      for row in rows)
        graph_keys = {(row["length"], row["initial"]) for row in graphs}
        graph_ok = all(row["batch"] == 2 and row["replay"] == 8
                       and row["gate_dtype"] == "fp32" and row["allocation"] == "STABLE"
                       and row["public_prepared"] == "RAW-BIT/PASS" for row in graphs)
        return (data["status"] == "PASS" and len(rows) == 36 and seen == expected and numeric
                and len(graphs) == 4 and graph_keys == {(t, h) for t in (64, 65) for h in (False, True)}
                and graph_ok and data["route"]["native_provider"] == "packaged-serving"
                and data["route"]["native_entry"] == "actlize_la_ppu10_forward_prepared")
    except (KeyError, TypeError, IndexError, ValueError):
        return False
