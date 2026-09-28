#!/usr/bin/env python3
"""All63 tail sizes; use the existing independent oracle and RAW-BIT gate."""
import argparse
import os
from pathlib import Path
import torch
from test_ppu_residual_backend import admit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    if not args.extension.is_file():
        parser.error("mixed-tail extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if "PPU" not in torch.cuda.get_device_properties(0).name.upper():
        raise RuntimeError("mixed-tail admission requires PPU")
    count = 0
    for length in range(65, 128):
        for gate in (-.1, -1.):
            for initial in (False, True):
                admit((1, length, 1, 2), gate, initial, deliveries=["full-chunk", "mixed-tail"])
                count += 1
    for length in (2049, 2111):
        for gate in (-.1, -1.):
            for initial in (False, True):
                admit((1, length, 16, 32), gate, initial, deliveries=["full-chunk", "mixed-tail"])
                count += 1
    admit((2, 191, 2, 4), -.01, True, stress=True, deliveries=["mixed-tail"])
    count += 1
    if count != 261:
        raise AssertionError("mixed-tail device denominator changed")
    print("[mixed tail device] PASS cases=261 repeats=8 all63tails+long+stress RAW-BIT+2%-oracle routing=UNCHANGED")


if __name__ == "__main__":
    main()
