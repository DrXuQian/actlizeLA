#!/usr/bin/env python3
"""Bind the zero-operand consequence to the fc8cbac box failure's fixture.

This is a CPU numerical signature, NOT device replay. L006 proves the broken
register alias independently; the old device log did not print element values.
Import the real test's fixture, reference and unchanged 2% comparator rather
than reproducing any of them here. No extension or device is loaded.
"""
import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "ppu_device_admission", ROOT / "tests/test_ppu_gdn_backend.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)


def main():
    torch.set_num_threads(1)
    inputs = admission.fixture(2, 65, 1, 2, 0.0)
    want = admission.reference(inputs)
    zero = tuple(torch.zeros_like(x) for x in want)
    errs = [admission.error(a, b) for a, b in zip(zero, want)]
    observed = [0.9999995827674866, 1.0]
    if errs != observed:
        raise AssertionError(f"historical signature changed: {errs} != {observed}")
    admission.assert_pair(want, want)
    try:
        admission.assert_pair(zero, want)
    except AssertionError:
        pass
    else:
        raise AssertionError("zero-result negative escaped the device comparator")
    print(f"[PPU retile signature] device_baseline=fc8cbac "
          f"case=tail-gva-hillis input_sha={admission.digest(inputs)} "
          f"reference_sha={admission.digest(want)} zero_output_state_error={errs} "
          f"limit={admission.MAX_RELATIVE_ERROR} zero-result=EXPECTED-RED/PASS "
          "scope=CPU-SIGNATURE-NOT-DEVICE-REPLAY")


if __name__ == "__main__":
    main()
